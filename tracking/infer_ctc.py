"""
Inference → Cell Tracking Challenge (CTC) output (detection-based, no segmentation).

GT masks are used ONLY to filter background detections (cell_id == 0 → discard),
consistent with training. No GT information is used for tracking decisions.

For each tracked cluster, a small ellipsoid is drawn at the cluster's mean position
(mean of all merged detections). Tracking score can be evaluated without segmentation.

Output layout:
  <out_dir>/
    mask000.tif … maskNNN.tif   uint16, same ZYX shape as input images
    res_track.txt               L B E P per track (CTC format)

Algorithm per frame pair (t, t+1):
  1. Forward pass → per-edge same-cell probability.
  2. Intra-frame union-find: merge detections in the same frame with
     edge score ≥ threshold → cluster representative = mean position.
  3. Cross-frame mean-score matrix between clusters → Hungarian 1-to-1.
  4. Mitosis: if matched cluster-t also has a second unmatched cluster-(t+1)
     above threshold → 1-to-2 link (division).
  5. Propagate track IDs via per-detection identity (det_track arrays),
     using majority-vote to reconcile across pairs.

Usage:
    python -m tracking.infer_ctc --exp 0515 --ckpt cache/checkpoints/best.pt
"""

import os
import argparse
import json
from collections import Counter

import numpy as np
import torch
import tifffile
from scipy.optimize import linear_sum_assignment

from tracking.config import Config
from tracking.dataset import _build_graph, _scale
from tracking.model import TrackingNet
from tracking.infer import (intra_cluster, cluster_representative,
                             cross_frame_scores, match_cross_frame)
from tracking.preprocess import EXPERIMENT_DIRS, sorted_tifs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_fg(centers, cell_ids, patches):
    """Keep only foreground detections (cell_id > 0)."""
    m = cell_ids > 0
    return centers[m], cell_ids[m], patches[m]


def _cluster_local(clusters_global, offset):
    """Global node indices → local indices (subtract frame offset)."""
    return [[i - offset for i in cl] for cl in clusters_global]


def _cluster_tid(cluster_local, det_track):
    """Majority track_id among detections in this cluster (0 = unknown)."""
    tids = [int(det_track[i]) for i in cluster_local if det_track[i] > 0]
    return Counter(tids).most_common(1)[0][0] if tids else 0


def _draw_ellipsoid(mask, center_zyx, radius_zyx, value):
    """Paint an ellipsoid at center_zyx with given per-axis radii."""
    Z, Y, X = mask.shape
    z0, y0, x0 = int(round(center_zyx[0])), int(round(center_zyx[1])), int(round(center_zyx[2]))
    rz, ry, rx = radius_zyx

    z_lo = max(0, z0 - rz);  z_hi = min(Z, z0 + rz + 1)
    y_lo = max(0, y0 - ry);  y_hi = min(Y, y0 + ry + 1)
    x_lo = max(0, x0 - rx);  x_hi = min(X, x0 + rx + 1)

    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    inside = ((zz - z0) / rz) ** 2 + ((yy - y0) / ry) ** 2 + ((xx - x0) / rx) ** 2 <= 1.0
    mask[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][inside] = np.uint16(value)


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ctc_inference(cfg, exp_id, ckpt_path, out_dir,
                      radius_zyx=(3, 6, 6)):
    """
    radius_zyx: ellipsoid radii (Z, Y, X) in voxels drawn at each detection.
                Z is typically smaller due to anisotropy.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = TrackingNet(feat_dim=cfg.feat_dim, gnn_layers=cfg.gnn_layers).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dirs = EXPERIMENT_DIRS[exp_id]
    mit_path = os.path.join(cfg.data_root, dirs['mitosis'])
    with open(mit_path) as f:
        mit = json.load(f)
    parents_gt = {int(k): v for k, v in mit['Parents'].items()}
    p2c_gt = {}
    for cid, info in mit['Children'].items():
        p2c_gt.setdefault(info['ParentID'], []).append(int(cid))

    cache_dir = os.path.join(cfg.cache_dir, exp_id)
    n_frames  = len([f for f in os.listdir(cache_dir) if f.startswith('frame_')])

    # Get image shape from one mask file (shape only, no content used for tracking)
    ref_shape = tifffile.imread(
        sorted_tifs(os.path.join(cfg.data_root, dirs['mask']))[0]
    ).shape

    # det_track[t]: np.ndarray (N_fg_t,) — track_id per foreground detection at t
    # frame_clusters[t]: list of (tid, rep_zyx) — for writing masks
    det_track     = {}   # t -> array
    frame_clusters = {t: [] for t in range(n_frames)}

    track_meta = {}   # tid -> {'first', 'last', 'parent'}
    next_tid   = 1    # CTC labels: 1-based (0 = background)

    for t in range(n_frames - 1):
        d0 = np.load(os.path.join(cache_dir, f'frame_{t:04d}.npz'))
        d1 = np.load(os.path.join(cache_dir, f'frame_{t+1:04d}.npz'))

        # Load ALL detections (FG + BG) — no GT filter
        c0, ids0, p0 = d0['centers'], d0['cell_ids'], d0['patches']
        c1, ids1, p1 = d1['centers'], d1['cell_ids'], d1['patches']

        if len(c0) == 0 or len(c1) == 0:
            det_track.pop(t, None)
            continue

        N0, N1 = len(c0), len(c1)

        all_c   = np.vstack([c0, c1]).astype(np.float32)
        sc      = _scale(all_c, cfg.z_anisotropy)
        ff      = np.array([0]*N0 + [1]*N1, dtype=np.float32)
        pos     = np.column_stack([sc, ff]).astype(np.float32)
        patches = np.concatenate([p0, p1], axis=0)

        edge_index, edge_feat, _, _ = _build_graph(
            c0, ids0, c1, ids1, t, parents_gt, p2c_gt,
            cfg.r_intra, cfg.r_cross, cfg.z_anisotropy,
        )
        if edge_index.shape[1] == 0:
            det_track.pop(t, None)
            continue

        edge_logits, mit_logits, fg_logits, daughter_logits = model(
            torch.from_numpy(patches).to(device),
            torch.from_numpy(pos).to(device),
            torch.from_numpy(edge_index).to(device),
            torch.from_numpy(edge_feat).to(device),
        )
        scores          = torch.sigmoid(edge_logits).cpu().numpy()
        mit_scores      = torch.sigmoid(mit_logits).cpu().numpy()
        fg_scores       = torch.sigmoid(fg_logits).cpu().numpy()       # (N0+N1,)
        daughter_scores = torch.sigmoid(daughter_logits).cpu().numpy() # (N0+N1,)

        # Mark predicted-background nodes with frame flag -1 so intra_cluster
        # and cross_frame_scores naturally ignore them.
        ff_int = ff.astype(np.int32).copy()
        ff_int[fg_scores < cfg.fg_threshold] = -1

        # --- Intra-frame union-find (score-based, FG nodes only) ---
        cl0_g = intra_cluster(all_c, scores, edge_index, ff_int, 0, cfg.intra_threshold)
        cl1_g = intra_cluster(all_c, scores, edge_index, ff_int, 1, cfg.intra_threshold)

        if not cl0_g or not cl1_g:
            det_track.pop(t, None)
            continue

        cl0_loc = _cluster_local(cl0_g, 0)    # local indices 0..N0-1
        cl1_loc = _cluster_local(cl1_g, N0)   # local indices 0..N1-1

        # Representative positions (mean of merged detections)
        reps0 = cluster_representative(cl0_g, all_c)   # (K0, 3) ZYX
        reps1 = cluster_representative(cl1_g, all_c)

        # --- Determine track IDs for frame-t clusters ---
        prev_track = det_track.get(t)
        new_track0 = np.zeros(N0, dtype=np.int32)
        cl0_tid    = []
        used_tids  = set()   # prevent two clusters reusing the same tid in one pair

        for k0, cl in enumerate(cl0_loc):
            tid = _cluster_tid(cl, prev_track) if prev_track is not None else 0
            if tid == 0 or tid in used_tids:
                # New cluster (or duplicate tid from changed clustering): fresh track
                tid = next_tid; next_tid += 1
                track_meta[tid] = {'first': t, 'last': t, 'parent': 0}
                frame_clusters[t].append((tid, reps0[k0]))
            used_tids.add(tid)
            cl0_tid.append(tid)
            for i in cl:
                new_track0[i] = tid

        det_track[t] = new_track0

        # Per-cluster mitosis head score (mean over frame-t nodes in cluster)
        cl0_mit = np.array([
            mit_scores[list(cl)].mean() for cl in cl0_g
        ], dtype=np.float32)

        # Per-cluster daughter head score (mean over frame-t+1 nodes in cluster)
        cl1_daughter = np.array([
            daughter_scores[list(cl)].mean() for cl in cl1_g
        ], dtype=np.float32)

        # --- Cross-frame matching ---
        aff         = cross_frame_scores(cl0_g, cl1_g, scores, edge_index)
        assignments = match_cross_frame(
            aff, cl0_g, cl1_g, reps0, reps1,
            cfg.cross_threshold, cfg.r_cross, cfg.z_anisotropy,
            mitosis_threshold=cfg.mitosis_threshold,
            cluster0_mit_scores=cl0_mit,
            mitosis_head_threshold=cfg.mitosis_head_threshold,
            cluster1_daughter_scores=cl1_daughter,
            daughter_head_threshold=cfg.daughter_head_threshold,
        )

        # --- Assign track IDs to frame-(t+1) clusters ---
        new_track1 = np.zeros(N1, dtype=np.int32)
        matched_k1 = set()

        for k0, k1s in assignments:
            tid = cl0_tid[k0]
            if len(k1s) == 1:
                k1 = k1s[0]
                track_meta[tid]['last'] = t + 1
                for i in cl1_loc[k1]:
                    new_track1[i] = tid
                frame_clusters[t + 1].append((tid, reps1[k1]))
                matched_k1.add(k1)
            else:
                # Mitosis: parent ends at t, daughters start at t+1.
                # Explicitly set last=t in case a duplicate-tid normal match
                # already pushed it to t+1.
                track_meta[tid]['last'] = t
                for k1 in k1s:
                    dtid = next_tid; next_tid += 1
                    track_meta[dtid] = {'first': t + 1, 'last': t + 1, 'parent': tid}
                    for i in cl1_loc[k1]:
                        new_track1[i] = dtid
                    frame_clusters[t + 1].append((dtid, reps1[k1]))
                    matched_k1.add(k1)

        # Unmatched clusters at t+1 → new tracks
        for k1 in range(len(cl1_g)):
            if k1 not in matched_k1:
                tid = next_tid; next_tid += 1
                track_meta[tid] = {'first': t + 1, 'last': t + 1, 'parent': 0}
                for i in cl1_loc[k1]:
                    new_track1[i] = tid
                frame_clusters[t + 1].append((tid, reps1[k1]))

        det_track[t + 1] = new_track1

        if t % 20 == 0:
            n_mit = sum(1 for _, k1s in assignments if len(k1s) == 2)
            print(f'  pair {t}/{n_frames-1}: '
                  f'det={N0}/{N1}  cl={len(cl0_g)}/{len(cl1_g)}  '
                  f'matched={len(assignments)}  mitoses={n_mit}')

    # -----------------------------------------------------------------------
    # Write CTC mask TIFFs: draw ellipsoid at each cluster representative
    # -----------------------------------------------------------------------
    for t in range(n_frames):
        out_mask = np.zeros(ref_shape, dtype=np.uint16)
        for tid, rep in frame_clusters[t]:
            _draw_ellipsoid(out_mask, rep, radius_zyx, tid)
        tifffile.imwrite(
            os.path.join(out_dir, f'mask{t:03d}.tif'),
            out_mask, compression='lzw',
        )
        if t % 20 == 0:
            print(f'  mask {t:03d}/{n_frames}  detections={len(frame_clusters[t])}')

    # -----------------------------------------------------------------------
    # Write res_track.txt
    # -----------------------------------------------------------------------
    with open(os.path.join(out_dir, 'res_track.txt'), 'w') as f:
        for tid in sorted(track_meta):
            m = track_meta[tid]
            f.write(f"{tid} {m['first']} {m['last']} {m['parent']}\n")

    n_tracks = next_tid - 1
    n_mit    = sum(1 for m in track_meta.values() if m['parent'] > 0)
    print(f'\nDone. {n_tracks} tracks ({n_mit} mitosis daughters).')
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',    default='0515')
    parser.add_argument('--ckpt',   default='cache/checkpoints/best.pt')
    parser.add_argument('--out',    default=None)
    parser.add_argument('--rz',     type=int, default=3, help='ellipsoid Z radius (voxels)')
    parser.add_argument('--ry',     type=int, default=6, help='ellipsoid Y radius (voxels)')
    parser.add_argument('--rx',     type=int, default=6, help='ellipsoid X radius (voxels)')
    args = parser.parse_args()

    cfg = Config()
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(cfg.data_root, args.ckpt)
    out = args.out or os.path.join(cfg.cache_dir, f'ctc_{args.exp}')
    run_ctc_inference(cfg, args.exp, args.ckpt, out,
                      radius_zyx=(args.rz, args.ry, args.rx))
