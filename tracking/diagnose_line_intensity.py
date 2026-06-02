"""
Analyse whether pixel intensity along the line connecting two detections
is a useful edge feature for distinguishing connection types.

For each edge category we measure:
  - Intra-frame edges (same volume):
      mean_intensity, min_intensity, midpoint_intensity, min/mean ratio
      (same-cell: uniformly high; different cells: high-low-high gap)
  - Cross-frame edges (different volumes):
      intensity at the TARGET detection's position in the SOURCE frame
      (same cell: moderate-high — cell was near B in frame t;
       parent→daughter: HIGH — daughter's future position is inside mother;
       wrong link: LOW — target location is empty in source frame)

Usage:
    python -m tracking.diagnose_line_intensity
"""

import os
import json
from collections import defaultdict

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from tracking.config import Config
from tracking.preprocess import EXPERIMENT_DIRS, sorted_tifs
from tracking.dataset import _scale


N_SAMPLE = 20    # points sampled along each line


def sample_line(vol, p1_zyx, p2_zyx, n=N_SAMPLE):
    """
    Sample pixel values along the 3D line from p1 to p2 in vol.
    Returns array of length n (float).
    """
    coords = np.linspace(p1_zyx, p2_zyx, n).T   # (3, n)
    # clip to volume bounds
    for i, s in enumerate(vol.shape):
        coords[i] = coords[i].clip(0, s - 1)
    vals = map_coordinates(vol.astype(np.float32), coords, order=1, mode='nearest')
    return vals


def sample_point(vol, zyx):
    """Trilinear interpolation at a single point."""
    coords = np.array([[zyx[0]], [zyx[1]], [zyx[2]]])
    for i, s in enumerate(vol.shape):
        coords[i] = coords[i].clip(0, s - 1)
    return float(map_coordinates(vol.astype(np.float32), coords, order=1, mode='nearest')[0])


def line_stats(vals):
    return dict(
        mean=float(vals.mean()),
        min=float(vals.min()),
        midpt=float(vals[len(vals)//2]),
        min_over_mean=float(vals.min() / (vals.mean() + 1e-6)),
    )


def report(name, stats_list):
    if not stats_list:
        print(f'  {name}: no samples')
        return
    keys = stats_list[0].keys()
    print(f'\n  {name}  (n={len(stats_list)})')
    for k in keys:
        vals = [s[k] for s in stats_list]
        print(f'    {k:20s}: '
              f'mean={np.mean(vals):.3f}  '
              f'median={np.median(vals):.3f}  '
              f'p10={np.percentile(vals,10):.3f}  '
              f'p90={np.percentile(vals,90):.3f}')


def main():
    cfg    = Config()
    exp_id = cfg.val_exps[0]
    dirs   = EXPERIMENT_DIRS[exp_id]

    det_files = sorted_tifs(os.path.join(cfg.data_root, dirs['det']))
    raw_files = sorted_tifs(os.path.join(cfg.data_root, dirs['raw']))
    n_frames  = len(det_files)

    mit_path = os.path.join(cfg.data_root, dirs['mitosis'])
    with open(mit_path) as f:
        mit = json.load(f)
    parents_gt = {int(k): v for k, v in mit['Parents'].items()}
    p2c_gt = {}
    for cid, info in mit['Children'].items():
        p2c_gt.setdefault(info['ParentID'], []).append(int(cid))

    cache_dir = os.path.join(cfg.cache_dir, exp_id)

    # Results per category
    res = defaultdict(list)

    # Normalise volumes to [0,1] at load time; cache last two frames
    vol_cache = {}
    def load_vol(t, source='det'):
        key = (t, source)
        if key not in vol_cache:
            path = det_files[t] if source == 'det' else raw_files[t]
            v = tifffile.imread(path).astype(np.float32)
            v = (v - v.min()) / (v.max() - v.min() + 1e-6)
            # keep only last 4 frames in cache
            if len(vol_cache) > 4:
                oldest = next(iter(vol_cache))
                del vol_cache[oldest]
            vol_cache[key] = v
        return vol_cache[key]

    print('Analysing frames...')

    for t in range(min(n_frames - 1, 355)):
        d0 = np.load(os.path.join(cache_dir, f'frame_{t:04d}.npz'))
        d1 = np.load(os.path.join(cache_dir, f'frame_{t+1:04d}.npz'))
        c0, ids0 = d0['centers'], d0['cell_ids']
        c1, ids1 = d1['centers'], d1['cell_ids']

        if len(c0) == 0 or len(c1) == 0:
            continue

        vol0 = load_vol(t,     'det')
        vol1 = load_vol(t + 1, 'det')

        # GT division at this transition
        gt_divs = {}
        for pid, pinfo in parents_gt.items():
            if pinfo['LastFrame'] == t:
                gt_divs[pid] = set(p2c_gt.get(pid, []))

        # -----------------------------------------------------------------
        # Intra-frame edges (frame t, same volume)
        # -----------------------------------------------------------------
        for i in range(len(c0)):
            for j in range(i + 1, len(c0)):
                # Skip if too far (only look at nearby pairs as in r_intra)
                dxy = np.linalg.norm(c0[i, 1:] - c0[j, 1:])
                if dxy > cfg.r_intra * 2:
                    continue
                vals = sample_line(vol0, c0[i], c0[j])
                st   = line_stats(vals)
                if ids0[i] > 0 and ids0[j] > 0 and ids0[i] == ids0[j]:
                    res['intra_same_cell'].append(st)
                elif ids0[i] > 0 and ids0[j] > 0:
                    res['intra_diff_cell'].append(st)

        # Daughter-pair intra-frame: two daughters in frame t+1 from same division
        for pid, dau_ids in gt_divs.items():
            dau_list = list(dau_ids)
            if len(dau_list) < 2:
                continue
            # Find positions in frame t+1
            pos = []
            for did in dau_list:
                idx = np.where(ids1 == did)[0]
                if len(idx):
                    pos.append(c1[idx[0]])
            if len(pos) >= 2:
                vals = sample_line(vol1, pos[0], pos[1])
                res['intra_daughter_pair'].append(line_stats(vals))

        # -----------------------------------------------------------------
        # Cross-frame edges: key metric = intensity at target's position in source frame
        # -----------------------------------------------------------------
        for i in range(len(c0)):
            if ids0[i] == 0:
                continue
            cid0 = int(ids0[i])

            # (A) Same cell across frames (normal tracking)
            idx1_same = np.where(ids1 == cid0)[0]
            if len(idx1_same):
                j = idx1_same[0]
                # intensity along line in source frame (t)
                src_line = line_stats(sample_line(vol0, c0[i], c1[j]))
                # intensity at TARGET position in source frame
                tgt_in_src = sample_point(vol0, c1[j])
                # intensity at SOURCE position in target frame
                src_in_tgt = sample_point(vol1, c0[i])
                res['cross_same_cell'].append({
                    **{f'src_line_{k}': v for k, v in src_line.items()},
                    'tgt_in_src': tgt_in_src,
                    'src_in_tgt': src_in_tgt,
                })

            # (B) Parent → daughter
            if cid0 in gt_divs:
                for did in gt_divs[cid0]:
                    idx1_dau = np.where(ids1 == did)[0]
                    if len(idx1_dau):
                        j = idx1_dau[0]
                        src_line = line_stats(sample_line(vol0, c0[i], c1[j]))
                        tgt_in_src = sample_point(vol0, c1[j])   # daughter pos in source
                        src_in_tgt = sample_point(vol1, c0[i])   # mother pos in target
                        res['cross_parent_daughter'].append({
                            **{f'src_line_{k}': v for k, v in src_line.items()},
                            'tgt_in_src': tgt_in_src,
                            'src_in_tgt': src_in_tgt,
                        })

        # (C) Random wrong cross-frame pairs (sample 30 random pairs as negative)
        rng = np.random.default_rng(t)
        idxs0_fg = np.where(ids0 > 0)[0]
        idxs1_fg = np.where(ids1 > 0)[0]
        if len(idxs0_fg) > 0 and len(idxs1_fg) > 0 and len(res['cross_wrong']) < 3000:
            for _ in range(10):
                i = int(rng.choice(idxs0_fg))
                j = int(rng.choice(idxs1_fg))
                if ids1[j] == ids0[i]:          # skip true same-cell pairs
                    continue
                if ids0[i] in gt_divs and ids1[j] in gt_divs[ids0[i]]:
                    continue                     # skip true parent-daughter
                tgt_in_src = sample_point(vol0, c1[j])
                src_in_tgt = sample_point(vol1, c0[i])
                res['cross_wrong'].append({
                    'tgt_in_src': tgt_in_src,
                    'src_in_tgt': src_in_tgt,
                })

        if t % 50 == 0:
            print(f'  frame {t:03d}  '
                  + '  '.join(f'{k}:{len(v)}' for k, v in res.items()))

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    print()
    print('=' * 65)
    print('  LINE INTENSITY FEATURE ANALYSIS  (detection probability map)')
    print('  All values normalised 0–1 per frame')
    print('=' * 65)

    # Intra-frame
    print('\n--- INTRA-FRAME EDGES (sampled in same volume) ---')
    print('  Key: min/mean ratio close to 1.0 = uniform (same cell)')
    print('       min/mean ratio low = gap in middle (membrane boundary)')
    report('intra_same_cell    (both detections = same GT cell)',
           res['intra_same_cell'])
    report('intra_daughter_pair(two daughters right after division)',
           res['intra_daughter_pair'])
    report('intra_diff_cell    (two different GT cells, nearby)',
           res['intra_diff_cell'])

    # Cross-frame
    print('\n--- CROSS-FRAME EDGES ---')
    print('  Key feature: tgt_in_src = intensity at B\'s position in frame t')
    print('    parent→daughter: daughter pos is INSIDE mother in frame t → HIGH')
    print('    same cell:       cell was near B in t            → moderate')
    print('    wrong link:      B is unrelated, random location  → LOW')
    report('cross_same_cell      (correct tracking link)',
           res['cross_same_cell'])
    report('cross_parent_daughter(mother → daughter)',
           res['cross_parent_daughter'])
    report('cross_wrong          (random wrong link)',
           res['cross_wrong'])

    print()
    print('=' * 65)
    print('SUMMARY:')
    def cat_median(cat, key):
        vals = [s[key] for s in res[cat] if key in s]
        return np.median(vals) if vals else float('nan')

    print(f'  Intra min/mean:  same_cell={cat_median("intra_same_cell","min_over_mean"):.3f}  '
          f'daughter_pair={cat_median("intra_daughter_pair","min_over_mean"):.3f}  '
          f'diff_cell={cat_median("intra_diff_cell","min_over_mean"):.3f}')
    print(f'  Cross tgt_in_src: same_cell={cat_median("cross_same_cell","tgt_in_src"):.3f}  '
          f'parent_dau={cat_median("cross_parent_daughter","tgt_in_src"):.3f}  '
          f'wrong={cat_median("cross_wrong","tgt_in_src"):.3f}')
    print('=' * 65)


if __name__ == '__main__':
    main()
