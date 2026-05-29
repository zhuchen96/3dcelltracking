"""
Coverage analysis: detection gaps and tracking gaps.

Reports:
  1. Detection gap  — GT cells with no detection in the cached NPZ
                      (hard ceiling; model can never recover these)
  2. Tracking gap   — GT cells never matched in the final output masks
                      (per-frame FN + per-track "never seen" count)

Usage:
    python -m tracking.coverage_analysis \\
        --exp 0515 \\
        --res cache/ctc_0515 \\
        [--threshold 15]
"""

import os
import argparse
import numpy as np
import tifffile
from collections import defaultdict

from tracking.config import Config
from tracking.eval_tra import centroids_from_mask, hungarian_match, read_track_txt
from tracking.preprocess import EXPERIMENT_DIRS


def run(exp_id, res_dir, threshold=15.0, z_anisotropy=0.5):
    cfg      = Config()
    dirs     = EXPERIMENT_DIRS[exp_id]
    cache_dir = os.path.join(cfg.cache_dir, exp_id)

    # GT masks come from the ctc_eval directory for this experiment
    # (adjust path if your GT lives elsewhere)
    gt_dirs = [
        os.path.join(cfg.cache_dir, 'ctc_eval', '2nd_with_mito', '01_GT', 'TRA'),
        os.path.join(cfg.cache_dir, 'ctc_eval', '1st_no_mito',   '01_GT', 'TRA'),
    ]
    gt_dir = next((d for d in gt_dirs if os.path.isdir(d)), None)
    if gt_dir is None:
        raise RuntimeError('Could not find GT TRA directory')

    gt_masks  = sorted(f for f in os.listdir(gt_dir)
                       if f.startswith('man_track') and f.endswith('.tif'))
    res_masks = sorted(f for f in os.listdir(res_dir)
                       if f.startswith('mask') and f.endswith('.tif'))
    frames    = sorted(f for f in os.listdir(cache_dir) if f.startswith('frame_'))
    n = min(len(gt_masks), len(res_masks), len(frames))

    gt_tracks  = read_track_txt(os.path.join(gt_dir, 'man_track.txt'))

    # --- 1. Detection gap (NPZ level) ---
    det_total_gt  = det_total_unc = 0
    det_unc_ids   = defaultdict(set)   # frame -> set of uncovered GT labels

    # --- 2. Tracking gap (output mask level) ---
    trk_total_gt  = trk_total_fn = 0
    gt_matched_frames = defaultdict(int)  # gt_label -> frames where it was matched
    gt_total_frames   = defaultdict(int)  # gt_label -> total frames it appears

    for t in range(n):
        # --- Detection gap ---
        npz     = np.load(os.path.join(cache_dir, frames[t]))
        det_ids = set(npz['cell_ids'].tolist()) - {0}
        gt_mask = tifffile.imread(os.path.join(gt_dir, gt_masks[t]))
        gt_ids  = set(np.unique(gt_mask).tolist()) - {0}
        unc     = gt_ids - det_ids
        det_total_gt  += len(gt_ids)
        det_total_unc += len(unc)
        for lbl in unc:
            det_unc_ids[t].add(lbl)

        # --- Tracking gap ---
        res_mask  = tifffile.imread(os.path.join(res_dir, res_masks[t]))
        gt_cents  = centroids_from_mask(gt_mask)
        res_cents = centroids_from_mask(res_mask)

        pairs       = hungarian_match(gt_cents, res_cents, threshold, z_anisotropy)
        matched_gt  = {gl for gl, _ in pairs}

        fn = len(gt_cents) - len(matched_gt)
        trk_total_gt += len(gt_cents)
        trk_total_fn += fn

        for lbl in gt_ids:
            gt_total_frames[lbl] += 1
        for lbl in matched_gt:
            gt_matched_frames[lbl] += 1

    # Tracks never matched in any frame
    never_matched = [lbl for lbl in gt_total_frames if gt_matched_frames[lbl] == 0]
    # Tracks matched in fewer than half their frames
    poorly_covered = [lbl for lbl in gt_total_frames
                      if 0 < gt_matched_frames[lbl] < gt_total_frames[lbl] // 2]

    print('=' * 55)
    print('  DETECTION GAP (NPZ detections vs GT mask labels)')
    print(f'    Total GT cell-instances : {det_total_gt}')
    print(f'    Undetected              : {det_total_unc}  '
          f'({100*det_total_unc/det_total_gt:.1f}%)')
    print()
    print('  TRACKING GAP (output masks vs GT mask labels)')
    print(f'    Total GT cell-instances : {trk_total_gt}')
    print(f'    Per-frame FN            : {trk_total_fn}  '
          f'({100*trk_total_fn/trk_total_gt:.1f}%)')
    print()
    print(f'  GT tracks total          : {len(gt_total_frames)}')
    print(f'  Tracks NEVER matched     : {len(never_matched)}  '
          f'({100*len(never_matched)/max(len(gt_total_frames),1):.1f}%)')
    print(f'  Tracks <50%% matched     : {len(poorly_covered)}  '
          f'({100*len(poorly_covered)/max(len(gt_total_frames),1):.1f}%)')
    print()
    print('  Breakdown of never-matched tracks:')
    for lbl in sorted(never_matched):
        m = gt_tracks.get(lbl, {})
        span = m.get('last', '?') - m.get('first', 0) + 1 if m else '?'
        print(f'    track {lbl:4d}  frames {m.get("first","?"):>4}–{m.get("last","?"):<4}  '
              f'(span={span}  parent={m.get("parent","?")})')
    print('=' * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',       default='0515')
    parser.add_argument('--res',       default=None,
                        help='result dir (default: cache/ctc_<exp>)')
    parser.add_argument('--threshold', type=float, default=15.0)
    args = parser.parse_args()

    cfg = Config()
    res = args.res or os.path.join(cfg.cache_dir, f'ctc_{args.exp}')
    run(args.exp, res, args.threshold)
