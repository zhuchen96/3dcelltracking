"""
Detection overlap analysis: compare predicted masks vs GT TRA masks frame by frame.

Measures:
  - FP detections: predicted cells whose voxels don't overlap any GT cell
  - FN detections: GT cells whose voxels have no predicted coverage

Usage:
    python -m tracking.diagnose_overlap
"""

import os
import glob
import numpy as np
import tifffile

GT_DIR   = '/srv/home/chen/3dtracking/cache/ctc_eval/2nd_with_mito/01_GT/TRA'
PRED_DIR = '/srv/home/chen/3dtracking/cache/ctc_0515'


def analyse_frame(gt_mask, pred_mask):
    """
    Returns (n_gt, n_pred, n_gt_detected, n_pred_fp) for one frame.
    A GT cell is "detected" if >= 1 predicted voxel overlaps it.
    A predicted cell is "FP" if 0 GT voxels overlap it.
    """
    gt_ids   = np.unique(gt_mask[gt_mask > 0])
    pred_ids = np.unique(pred_mask[pred_mask > 0])

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 0, 0, 0, 0

    # For each predicted cell: does it touch any GT region?
    n_pred_fp = 0
    pred_hit  = set()   # GT ids touched by at least one pred cell

    for pid in pred_ids:
        pmask = pred_mask == pid
        touched_gt = np.unique(gt_mask[pmask])
        touched_gt = touched_gt[touched_gt > 0]
        if len(touched_gt) == 0:
            n_pred_fp += 1
        else:
            pred_hit.update(touched_gt.tolist())

    n_gt_detected = len(pred_hit)
    return len(gt_ids), len(pred_ids), n_gt_detected, n_pred_fp


def main():
    gt_files   = sorted(glob.glob(os.path.join(GT_DIR,   'man_track*.tif')))
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, 'mask*.tif')))

    n_frames = min(len(gt_files), len(pred_files))
    print(f'Frames: {n_frames}  (GT={len(gt_files)}, pred={len(pred_files)})')

    total_gt       = 0
    total_pred     = 0
    total_detected = 0
    total_fp       = 0

    fp_frames  = []   # frames with any FP detection
    fn_frames  = []   # frames with any undetected GT cell

    for i in range(n_frames):
        gt   = tifffile.imread(gt_files[i])
        pred = tifffile.imread(pred_files[i])

        if gt.shape != pred.shape:
            print(f'  Frame {i}: shape mismatch GT={gt.shape} pred={pred.shape}')
            continue

        n_gt, n_pred, n_det, n_fp = analyse_frame(gt, pred)
        n_fn = n_gt - n_det

        total_gt       += n_gt
        total_pred     += n_pred
        total_detected += n_det
        total_fp       += n_fp

        if n_fp > 0:
            fp_frames.append((i, n_fp, n_pred))
        if n_fn > 0:
            fn_frames.append((i, n_fn, n_gt))

        if i % 50 == 0:
            print(f'  frame {i:03d}: GT={n_gt:3d}  pred={n_pred:3d}  '
                  f'detected={n_det:3d}  FP={n_fp:3d}  FN={n_fn:3d}')

    total_fn = total_gt - total_detected

    print()
    print('=' * 55)
    print(f'  Total GT cells       : {total_gt}')
    print(f'  Total pred cells     : {total_pred}')
    print(f'  GT detected (overlap): {total_detected}  '
          f'({100*total_detected/max(total_gt,1):.1f}%)')
    print(f'  GT undetected (FN)   : {total_fn}  '
          f'({100*total_fn/max(total_gt,1):.1f}%)')
    print(f'  Pred outside GT (FP) : {total_fp}  '
          f'({100*total_fp/max(total_pred,1):.1f}%)')
    print('=' * 55)

    if fp_frames:
        print(f'\nFrames with FP detections ({len(fp_frames)} frames):')
        for f, nfp, npred in fp_frames[:20]:
            print(f'  frame {f:03d}: {nfp} FP / {npred} pred total')
        if len(fp_frames) > 20:
            print(f'  ... and {len(fp_frames)-20} more frames')

    if fn_frames:
        print(f'\nFrames with undetected GT cells ({len(fn_frames)} frames):')
        for f, nfn, ngt in fn_frames[:20]:
            print(f'  frame {f:03d}: {nfn} undetected / {ngt} GT total')
        if len(fn_frames) > 20:
            print(f'  ... and {len(fn_frames)-20} more frames')


if __name__ == '__main__':
    main()
