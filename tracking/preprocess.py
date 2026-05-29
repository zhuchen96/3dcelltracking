"""
One-time preprocessing: extract detection centers + 3D patches per frame.
Uses per-frame ROI (from each frame's own mask) so ndimage.label only runs
on the small labeled sub-volume instead of the full 60M-voxel image.

Usage:
    python -m tracking.preprocess
"""

import os
import json
import numpy as np
import tifffile
from glob import glob
from scipy import ndimage


EXPERIMENT_DIRS = {
    '0501': dict(
        det='detections/0501',
        mask='masks/20250501_click_final',
        raw='raw_imgs/0501',
        mitosis='mitotic events/mitosis_info_0501.json',
    ),
    '0507': dict(
        det='detections/0507',
        mask='masks/20250507_click_final',
        raw='raw_imgs/0507',
        mitosis='mitotic events/mitosis_info_0507.json',
    ),
    '0515': dict(
        det='detections/0515',
        mask='masks/20250515_click_final',
        raw='raw_imgs/0515',
        mitosis='mitotic events/mitosis_info_0515.json',
    ),
}


def sorted_tifs(folder):
    return sorted(glob(os.path.join(folder, '*.tif')))


def frame_roi(mask, margin=None):
    """Bounding box (z1,y1,x1,z2,y2,x2) of nonzero voxels in one mask frame."""
    if margin is None:
        margin = np.array([2, 4, 4])
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    lo = np.maximum(0, coords.min(0) - margin)
    hi = coords.max(0) + margin + 1
    hi = np.minimum(hi, np.array(mask.shape))
    return tuple(lo.tolist() + hi.tolist())   # (z1,y1,x1, z2,y2,x2)


def extract_centers_in_roi(det, roi, min_size=30):
    """
    Find detection centers (zero-sphere centroids) within a tight ROI.
    Crops the detection array to the ROI before calling ndimage.label
    so the labeling runs on a small sub-volume.
    Returns (N, 3) float32 array of ZYX centers in GLOBAL image coords.
    """
    z1, y1, x1, z2, y2, x2 = roi
    crop = (det[z1:z2, y1:y2, x1:x2] == 0)      # small 3D bool array

    labeled, n = ndimage.label(crop)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    sizes = np.array(ndimage.sum(crop, labeled, range(1, n + 1)))
    centers = []
    for i in range(n):
        if sizes[i] >= min_size:
            pts = np.argwhere(labeled == i + 1)   # local coords
            local_center = pts.mean(0).astype(np.float32)
            # Convert to global image coords
            global_center = local_center + np.array([z1, y1, x1], dtype=np.float32)
            centers.append(global_center)

    return np.array(centers, dtype=np.float32) if centers else np.zeros((0, 3), dtype=np.float32)


def assign_labels(centers, mask, radius=2):
    """GT cell label for each center (most common nonzero label in neighborhood)."""
    labels = np.zeros(len(centers), dtype=np.int32)
    Z, Y, X = mask.shape
    for i, (z, y, x) in enumerate(centers.astype(int)):
        patch = mask[
            max(0, z - radius):min(Z, z + radius + 1),
            max(0, y - radius):min(Y, y + radius + 1),
            max(0, x - radius):min(X, x + radius + 1),
        ]
        vals, cnts = np.unique(patch.ravel(), return_counts=True)
        nz = vals > 0
        if nz.any():
            labels[i] = int(vals[nz][cnts[nz].argmax()])
    return labels


def extract_patch(raw, center, patch_size):
    """Normalised float32 patch (1, pz, py, px) centered at ZYX center."""
    pz, py, px = patch_size
    z, y, x = center.astype(int)
    Z, Y, X = raw.shape

    z0, z1 = z - pz // 2, z + pz // 2
    y0, y1 = y - py // 2, y + py // 2
    x0, x1 = x - px // 2, x + px // 2

    pd = [(max(0, -z0), max(0, z1 - Z)),
          (max(0, -y0), max(0, y1 - Y)),
          (max(0, -x0), max(0, x1 - X))]

    crop = raw[max(0,z0):min(Z,z1), max(0,y0):min(Y,y1), max(0,x0):min(X,x1)].astype(np.float32)
    if any(p[0] + p[1] > 0 for p in pd):
        crop = np.pad(crop, pd, mode='reflect')

    mu, sigma = crop.mean(), crop.std()
    return ((crop - mu) / (sigma + 1e-6))[np.newaxis]  # (1, pz, py, px)


def preprocess_experiment(exp_id, data_root, cache_dir, patch_size=(16, 24, 24)):
    dirs = EXPERIMENT_DIRS[exp_id]
    det_dir  = os.path.join(data_root, dirs['det'])
    mask_dir = os.path.join(data_root, dirs['mask'])
    raw_dir  = os.path.join(data_root, dirs['raw'])

    out_dir = os.path.join(cache_dir, exp_id)
    os.makedirs(out_dir, exist_ok=True)

    det_files  = sorted_tifs(det_dir)
    mask_files = sorted_tifs(mask_dir)
    raw_files  = sorted_tifs(raw_dir)
    n_frames   = len(det_files)

    print(f'[{exp_id}] {n_frames} frames')

    for t in range(n_frames):
        out_path = os.path.join(out_dir, f'frame_{t:04d}.npz')
        if os.path.exists(out_path):
            continue

        mask = tifffile.imread(mask_files[t])
        roi  = frame_roi(mask)

        if roi is None:
            np.savez_compressed(out_path,
                                centers=np.zeros((0, 3), dtype=np.float32),
                                cell_ids=np.zeros(0, dtype=np.int32),
                                patches=np.zeros((0, 1, *patch_size), dtype=np.float32))
            continue

        det     = tifffile.imread(det_files[t])
        centers = extract_centers_in_roi(det, roi)

        if len(centers) == 0:
            np.savez_compressed(out_path,
                                centers=np.zeros((0, 3), dtype=np.float32),
                                cell_ids=np.zeros(0, dtype=np.int32),
                                patches=np.zeros((0, 1, *patch_size), dtype=np.float32))
            continue

        raw      = tifffile.imread(raw_files[t])
        cell_ids = assign_labels(centers, mask)
        patches  = np.stack([extract_patch(raw, c, patch_size) for c in centers])

        np.savez_compressed(out_path, centers=centers, cell_ids=cell_ids, patches=patches)

        if t % 20 == 0:
            roi_str = f'z{roi[0]}-{roi[3]} y{roi[1]}-{roi[4]} x{roi[2]}-{roi[5]}'
            print(f'  [{exp_id}] frame {t}/{n_frames}  '
                  f'dets={len(centers)}  labeled={(cell_ids>0).sum()}  roi={roi_str}')

    print(f'[{exp_id}] Done.')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tracking.config import Config
    cfg = Config()
    os.makedirs(cfg.cache_dir, exist_ok=True)

    exps = sys.argv[1:] if len(sys.argv) > 1 else ['0501', '0507', '0515']
    for exp in exps:
        preprocess_experiment(exp, cfg.data_root, cfg.cache_dir, cfg.patch_size)
