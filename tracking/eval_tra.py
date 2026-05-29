"""
Detection-based tracking evaluation using centroid matching.

Does NOT require segmentation overlap — GT and result cells are matched by
centroid distance (Hungarian assignment within a distance threshold).

Metrics reported:
  Det-P / Det-R / Det-F1  per-frame detection quality
  MOTA   = 1 - (FP + FN + IDSW) / sum_t(|GT_t|)
  MOTP   = mean matched centroid distance (isotropic pixels)
  IDF1   = 2*IDTP / (2*IDTP + IDFP + IDFN)   identity-preserving F1
  Div-P / Div-R / Div-F1  mitosis detection (parent centroid + frame matching)

Usage:
    python -m tracking.eval_tra \\
        --gt  cache/ctc_eval/01/01_GT/TRA \\
        --res cache/ctc_eval/01/01_RES \\
        --threshold 15
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_track_txt(path):
    """Return dict: tid -> {'first': int, 'last': int, 'parent': int}."""
    tracks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            L, B, E, P = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            tracks[L] = {'first': B, 'last': E, 'parent': P}
    return tracks


def centroids_from_mask(mask):
    """
    Return dict: label -> np.array([z, y, x]) for all nonzero labels.
    Uses argwhere + grouped mean — faster than repeated mask==lbl for many labels.
    """
    pos = np.argwhere(mask > 0)
    if len(pos) == 0:
        return {}
    lbls = mask[pos[:, 0], pos[:, 1], pos[:, 2]]
    order = np.argsort(lbls)
    pos   = pos[order]
    lbls  = lbls[order]
    split = np.where(np.diff(lbls))[0] + 1
    groups = np.split(pos, split)
    unique  = lbls[np.concatenate([[0], split])]
    return {int(u): g.mean(0) for u, g in zip(unique, groups)}


def iso_dist(c0, c1, z_anisotropy=0.5):
    """Isotropic Euclidean distance between two ZYX centroids."""
    d = c0 - c1
    d[0] *= z_anisotropy
    return float(np.linalg.norm(d))


def hungarian_match(gt_cents, res_cents, threshold, z_anisotropy=0.5):
    """
    Match GT centroids to result centroids within `threshold` isotropic pixels.
    Returns: list of (gt_label, res_label) matched pairs.
    """
    gt_lbls  = list(gt_cents.keys())
    res_lbls = list(res_cents.keys())
    if not gt_lbls or not res_lbls:
        return []

    # Cost matrix: isotropic distance (large cost = 1e9 if out of range)
    C = np.full((len(gt_lbls), len(res_lbls)), 1e9)
    for i, gl in enumerate(gt_lbls):
        for j, rl in enumerate(res_lbls):
            d = iso_dist(gt_cents[gl].copy(), res_cents[rl].copy(), z_anisotropy)
            if d <= threshold:
                C[i, j] = d

    ri, ci = linear_sum_assignment(C)
    pairs = []
    for r, c in zip(ri, ci):
        if C[r, c] <= threshold:
            pairs.append((gt_lbls[r], res_lbls[c]))
    return pairs


# ---------------------------------------------------------------------------
# Mitosis evaluation
# ---------------------------------------------------------------------------

def extract_divisions(tracks):
    """
    Return list of (parent_label, division_frame) from a track dict.
    division_frame = daughter's first frame - 1  (last frame the parent exists).
    Daughters are tracks with parent != 0.
    """
    events = []
    for tid, meta in tracks.items():
        if meta['parent'] != 0:
            div_frame = meta['first'] - 1   # frame parent last exists
            events.append((meta['parent'], div_frame))
    # Deduplicate: two daughters → one event
    return list(set(events))


def evaluate_mitosis(gt_tracks, res_tracks, gt_dir, res_dir,
                     threshold=15.0, frame_tol=1, z_anisotropy=0.5):
    """
    Match GT and result division events.

    A result division is a True Positive if:
      - The result parent cell centroid at division_frame is within `threshold`
        isotropic pixels of the GT parent cell centroid at (GT) division_frame.
      - The division frames differ by at most `frame_tol`.

    Returns dict with div_p, div_r, div_f1, tp, fp, fn.
    """
    gt_divs  = extract_divisions(gt_tracks)    # [(parent_lbl, frame), ...]
    res_divs = extract_divisions(res_tracks)

    if not gt_divs and not res_divs:
        print('  No division events in GT or result.')
        return dict(div_p=1.0, div_r=1.0, div_f1=1.0, tp=0, fp=0, fn=0)

    print(f'  GT divisions: {len(gt_divs)}   Result divisions: {len(res_divs)}')

    # Cache GT mask centroids per frame (only for frames involved in divisions)
    gt_frames_needed  = {f for _, f in gt_divs}
    res_frames_needed = {f for _, f in res_divs}
    all_frames_needed = gt_frames_needed | res_frames_needed

    gt_masks  = sorted(f for f in os.listdir(gt_dir)
                       if f.startswith('man_track') and f.endswith('.tif'))
    res_masks = sorted(f for f in os.listdir(res_dir)
                       if f.startswith('mask') and f.endswith('.tif'))

    gt_cents_cache  = {}
    res_cents_cache = {}
    for t in all_frames_needed:
        if t < len(gt_masks):
            m = tifffile.imread(os.path.join(gt_dir,  gt_masks[t]))
            gt_cents_cache[t]  = centroids_from_mask(m)
        if t < len(res_masks):
            m = tifffile.imread(os.path.join(res_dir, res_masks[t]))
            res_cents_cache[t] = centroids_from_mask(m)

    # Match GT divisions to result divisions
    matched_res = set()
    tp = fn = 0

    for gt_parent, gt_frame in gt_divs:
        gt_c = gt_cents_cache.get(gt_frame, {}).get(gt_parent)
        if gt_c is None:
            fn += 1
            continue

        best_dist = threshold + 1
        best_idx  = None

        for idx, (res_parent, res_frame) in enumerate(res_divs):
            if idx in matched_res:
                continue
            if abs(res_frame - gt_frame) > frame_tol:
                continue
            res_c = res_cents_cache.get(res_frame, {}).get(res_parent)
            if res_c is None:
                continue
            d = iso_dist(gt_c.copy(), res_c.copy(), z_anisotropy)
            if d < best_dist:
                best_dist = d
                best_idx  = idx

        if best_idx is not None:
            tp += 1
            matched_res.add(best_idx)
        else:
            fn += 1

    fp = len(res_divs) - len(matched_res)

    div_p  = tp / max(tp + fp, 1)
    div_r  = tp / max(tp + fn, 1)
    div_f1 = 2 * div_p * div_r / max(div_p + div_r, 1e-9)
    return dict(div_p=div_p, div_r=div_r, div_f1=div_f1, tp=tp, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(gt_dir, res_dir, threshold=15.0, z_anisotropy=0.5):
    gt_tracks  = read_track_txt(os.path.join(gt_dir,  'man_track.txt'))
    res_tracks = read_track_txt(os.path.join(res_dir, 'res_track.txt'))

    # Detect number of frames from GT mask files
    gt_masks = sorted(
        f for f in os.listdir(gt_dir)
        if f.startswith('man_track') and f.endswith('.tif')
    )
    res_masks = sorted(
        f for f in os.listdir(res_dir)
        if f.startswith('mask') and f.endswith('.tif')
    )
    n_frames = min(len(gt_masks), len(res_masks))
    print(f'Evaluating {n_frames} frames  (threshold={threshold} iso-px)')

    # -----------------------------------------------------------------------
    # Per-frame matching
    # -----------------------------------------------------------------------
    total_gt = total_fp = total_fn = total_idsw = 0
    total_dist = total_matched = 0

    # For IDF1: track-level identity accumulation
    # gt_track → res_track → count of frames they were matched together
    id_count = defaultdict(lambda: defaultdict(int))  # gt_tid -> res_tid -> frames
    gt_active_frames  = defaultdict(int)   # gt_tid  -> total frames it appears
    res_active_frames = defaultdict(int)   # res_tid -> total frames it appears

    prev_pairs = {}   # gt_label -> matched res_label from previous frame

    for t in range(n_frames):
        gt_mask  = tifffile.imread(os.path.join(gt_dir,  gt_masks[t]))
        res_mask = tifffile.imread(os.path.join(res_dir, res_masks[t]))

        gt_cents  = centroids_from_mask(gt_mask)
        res_cents = centroids_from_mask(res_mask)

        pairs = hungarian_match(gt_cents, res_cents, threshold, z_anisotropy)

        matched_gt  = {gl for gl, _  in pairs}
        matched_res = {rl for _,  rl in pairs}

        fp = len(res_cents) - len(matched_res)
        fn = len(gt_cents)  - len(matched_gt)
        total_gt  += len(gt_cents)
        total_fp  += fp
        total_fn  += fn

        # ID switches: GT cell matched to a different result track than last frame
        for gl, rl in pairs:
            if gl in prev_pairs and prev_pairs[gl] != rl:
                total_idsw += 1
            id_count[gl][rl] += 1
            total_dist    += iso_dist(gt_cents[gl].copy(), res_cents[rl].copy(),
                                      z_anisotropy)
            total_matched += 1

        for gl in gt_cents:
            gt_active_frames[gl] += 1
        for rl in res_cents:
            res_active_frames[rl] += 1

        prev_pairs = {gl: rl for gl, rl in pairs}

        if t % 50 == 0:
            p = len(matched_gt) / max(len(res_cents), 1)
            r = len(matched_gt) / max(len(gt_cents), 1)
            print(f'  T={t:3d}  GT={len(gt_cents)}  RES={len(res_cents)}  '
                  f'matched={len(pairs)}  FP={fp}  FN={fn}  '
                  f'P={p:.2f}  R={r:.2f}')

    # -----------------------------------------------------------------------
    # MOTA / MOTP
    # -----------------------------------------------------------------------
    mota = 1.0 - (total_fp + total_fn + total_idsw) / max(total_gt, 1)
    motp = total_dist / max(total_matched, 1)

    # -----------------------------------------------------------------------
    # IDF1
    # -----------------------------------------------------------------------
    # For each GT track, find the result track it's most associated with
    idtp = 0
    for gl, res_map in id_count.items():
        best_rl = max(res_map, key=res_map.get)
        idtp += res_map[best_rl]

    # IDFP = result detections not counted as IDTP
    # IDFN = GT detections not counted as IDTP
    idfp = total_matched - idtp   # matched but to "wrong" track from IDF perspective
    # also add unmatched result detections
    idfp += total_fp
    idfn = total_gt - idtp

    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    # Detection F1
    det_p = total_matched / max(total_matched + total_fp, 1)
    det_r = total_matched / max(total_matched + total_fn, 1)
    det_f1 = 2 * det_p * det_r / max(det_p + det_r, 1e-9)

    # -----------------------------------------------------------------------
    # Print
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Mitosis evaluation
    # -----------------------------------------------------------------------
    print('\nEvaluating divisions ...')
    div = evaluate_mitosis(gt_tracks, res_tracks, gt_dir, res_dir,
                           threshold, frame_tol=1, z_anisotropy=z_anisotropy)

    print()
    print('=' * 55)
    print(f'  Frames evaluated : {n_frames}')
    print(f'  Total GT dets    : {total_gt}')
    print(f'  Total FP         : {total_fp}')
    print(f'  Total FN         : {total_fn}')
    print(f'  ID switches      : {total_idsw}')
    print(f'  Matched          : {total_matched}')
    print('-' * 55)
    print(f'  Det-Precision    : {det_p:.4f}')
    print(f'  Det-Recall       : {det_r:.4f}')
    print(f'  Det-F1           : {det_f1:.4f}')
    print(f'  MOTA             : {mota:.4f}')
    print(f'  MOTP (iso-px)    : {motp:.2f}')
    print(f'  IDF1             : {idf1:.4f}')
    print('-' * 55)
    print(f'  Divisions GT/RES : {len(extract_divisions(gt_tracks))} / '
          f'{len(extract_divisions(res_tracks))}')
    print(f'  Div TP/FP/FN     : {div["tp"]} / {div["fp"]} / {div["fn"]}')
    print(f'  Div-Precision    : {div["div_p"]:.4f}')
    print(f'  Div-Recall       : {div["div_r"]:.4f}')
    print(f'  Div-F1           : {div["div_f1"]:.4f}')
    print('=' * 55)

    return dict(mota=mota, motp=motp, idf1=idf1,
                det_p=det_p, det_r=det_r, det_f1=det_f1,
                fp=total_fp, fn=total_fn, idsw=total_idsw,
                div_p=div['div_p'], div_r=div['div_r'], div_f1=div['div_f1'],
                div_tp=div['tp'], div_fp=div['fp'], div_fn=div['fn'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt',        required=True,
                        help='Path to GT TRA dir (containing man_track*.tif + man_track.txt)')
    parser.add_argument('--res',       required=True,
                        help='Path to RES dir (containing mask*.tif + res_track.txt)')
    parser.add_argument('--threshold',    type=float, default=15.0,
                        help='Centroid distance threshold in isotropic pixels (default 15)')
    parser.add_argument('--z_anisotropy', type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.gt, args.res, args.threshold, args.z_anisotropy)
