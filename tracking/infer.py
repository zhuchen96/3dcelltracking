"""
Inference: run TrackingNet on one experiment and produce cell tracks.

Algorithm per frame pair (t, t+1):
  1. Forward pass → same-cell probability for every edge.
  2. Intra-frame clustering (each frame separately):
       - same-frame edges with score ≥ threshold → union-find merge
       - each cluster → one representative (mean position)
  3. Cross-frame assignment:
       - For each cluster at t, find candidate clusters at t+1 within r_cross.
       - Score = mean edge probability between their detections.
       - Allow 1-to-2 links (mitosis): if one cluster-t best-matches two
         cluster-(t+1) AND those two clusters are "different cell" → division.
       - Otherwise 1-to-1 Hungarian matching.
  4. Extend / start / end tracks accordingly.

Usage:
    python -m tracking.infer --exp 0515 --ckpt cache/checkpoints/best.pt
"""

import os
import argparse
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from tracking.config import Config
from tracking.dataset import TrackingDataset, _build_graph, _scale
from tracking.model import TrackingNet


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def intra_cluster(centers, scores, edge_index, frame_flags, frame_id, threshold):
    """
    Cluster detections within one frame using intra-frame edges.
    Returns: list of arrays, each containing detection indices in that cluster.
    """
    node_mask = (frame_flags == frame_id)
    # Global → local index mapping
    global_idx = np.where(node_mask)[0]
    n_local = len(global_idx)
    if n_local == 0:
        return []

    g2l = {g: l for l, g in enumerate(global_idx)}

    uf = UF(n_local)
    src, dst = edge_index[0], edge_index[1]

    for e in range(len(scores)):
        s, d = int(src[e]), int(dst[e])
        # Only intra-frame edges for this frame
        if frame_flags[s] != frame_id or frame_flags[d] != frame_id:
            continue
        if scores[e] >= threshold:
            uf.union(g2l[s], g2l[d])

    # Group by root
    clusters = {}
    for l, g in enumerate(global_idx):
        root = uf.find(l)
        clusters.setdefault(root, []).append(g)
    return list(clusters.values())


def cluster_representative(clusters, centers):
    """Mean position for each cluster."""
    reps = []
    for c in clusters:
        reps.append(centers[c].mean(0))
    return np.array(reps)  # (K, 3)


def cross_frame_scores(clusters0, clusters1, scores, edge_index):
    """
    Mean edge probability between each pair of clusters (t vs t+1).
    Returns: (K0, K1) float matrix.
    """
    K0, K1 = len(clusters0), len(clusters1)
    if K0 == 0 or K1 == 0:
        return np.zeros((K0, K1))

    set0 = [set(c) for c in clusters0]
    set1 = [set(c) for c in clusters1]

    src, dst = edge_index[0], edge_index[1]
    aff = np.zeros((K0, K1))
    cnt = np.zeros((K0, K1))

    for e in range(len(scores)):
        s, d = int(src[e]), int(dst[e])
        # We want src in frame-t clusters, dst in frame-(t+1) clusters
        # (edges are stored both ways, so we'll pick the right orientation below)
        for k0, s0 in enumerate(set0):
            if s in s0:
                for k1, s1 in enumerate(set1):
                    if d in s1:
                        aff[k0, k1] += scores[e]
                        cnt[k0, k1] += 1
                        break
                break

    mask = cnt > 0
    aff[mask] /= cnt[mask]
    return aff


def match_cross_frame(aff, clusters0, clusters1, reps0, reps1,
                      threshold, r_cross, z_anisotropy,
                      mitosis_threshold=None,
                      cluster0_mit_scores=None, mitosis_head_threshold=0.5,
                      cluster1_daughter_scores=None, daughter_head_threshold=0.1):
    """
    1-to-1 or 1-to-2 (mitosis) matching.

    Division detection uses a pre-pass that reserves both daughters BEFORE
    running the 1-to-1 Hungarian, preventing the second daughter from being
    consumed by an unrelated 1-to-1 match.

    A division candidate (k0, k1a, k1b) is accepted when:
      1. Mother head   : cluster0_mit_scores[k0] >= mitosis_head_threshold
      2. Edge topology : aff[k0, k1a] >= mitosis_threshold
                        aff[k0, k1b] >= mitosis_threshold
      3. Daughter head : (optional) both daughters >= daughter_head_threshold

    Args:
        threshold                : minimum affinity for normal 1-to-1 linking.
        mitosis_threshold        : affinity threshold for both daughter candidates.
                                   Defaults to threshold if not set.
        cluster0_mit_scores      : (K0,) per-cluster mother mitosis head scores.
        mitosis_head_threshold   : gate on mother head score.
        cluster1_daughter_scores : (K1,) per-cluster daughter head scores.
        daughter_head_threshold  : soft gate on both daughters' head scores.

    Returns list of (k0, [k1, ...]) assignment tuples.
    """
    if mitosis_threshold is None:
        mitosis_threshold = threshold

    K0, K1 = len(clusters0), len(clusters1)
    if K0 == 0 or K1 == 0:
        return []

    sc0 = reps0.copy(); sc0[:, 0] *= z_anisotropy
    sc1 = reps1.copy(); sc1[:, 0] *= z_anisotropy
    dist_matrix = np.linalg.norm(sc0[:, None] - sc1[None], axis=-1)

    cost = 1.0 - aff.copy()
    cost[dist_matrix > r_cross] = 1.0

    # --- Pre-pass: identify and reserve division assignments ---
    # Process mothers in decreasing mit_score order so the most confident
    # divisions claim their daughters before less certain ones.
    reserved_k0 = set()
    reserved_k1 = set()
    division_assignments = []

    if cluster0_mit_scores is not None:
        k0_order = sorted(range(K0), key=lambda x: -float(cluster0_mit_scores[x]))
        for k0 in k0_order:
            if float(cluster0_mit_scores[k0]) < mitosis_head_threshold:
                break
            # Find top-2 eligible daughters for this mother
            candidates = []
            for k1 in range(K1):
                if k1 in reserved_k1:
                    continue
                if dist_matrix[k0, k1] > r_cross:
                    continue
                if aff[k0, k1] < mitosis_threshold:
                    continue
                if (cluster1_daughter_scores is not None and
                        float(cluster1_daughter_scores[k1]) < daughter_head_threshold):
                    continue
                candidates.append((aff[k0, k1], k1))
            if len(candidates) >= 2:
                candidates.sort(reverse=True)
                k1a, k1b = candidates[0][1], candidates[1][1]
                division_assignments.append((k0, [k1a, k1b]))
                reserved_k0.add(k0)
                reserved_k1.add(k1a)
                reserved_k1.add(k1b)

    # --- 1-to-1 Hungarian on remaining (non-reserved) clusters ---
    rem_k0 = [k for k in range(K0) if k not in reserved_k0]
    rem_k1 = [k for k in range(K1) if k not in reserved_k1]

    assignments = list(division_assignments)
    if rem_k0 and rem_k1:
        rk0 = np.array(rem_k0)
        rk1 = np.array(rem_k1)
        sub_cost = cost[np.ix_(rk0, rk1)]
        row_ind, col_ind = linear_sum_assignment(sub_cost)
        for r, c in zip(row_ind, col_ind):
            k0, k1 = int(rk0[r]), int(rk1[c])
            if aff[k0, k1] >= threshold:
                assignments.append((k0, [k1]))

    return assignments


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(cfg, exp_id, ckpt_path, out_path=None):
    device = torch.device('cpu')

    model = TrackingNet(feat_dim=cfg.feat_dim, gnn_layers=cfg.gnn_layers).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    import json
    from tracking.preprocess import EXPERIMENT_DIRS, sorted_tifs, extract_centers, assign_labels, extract_patch
    import tifffile

    dirs       = EXPERIMENT_DIRS[exp_id]
    cache_dir  = os.path.join(cfg.cache_dir, exp_id)
    mit_path   = os.path.join(cfg.data_root, dirs['mitosis'])
    with open(mit_path) as f:
        mit = json.load(f)
    parents = {int(k): v for k, v in mit['Parents'].items()}
    p2c = {}
    for cid, info in mit['Children'].items():
        pid = info['ParentID']
        p2c.setdefault(pid, []).append(int(cid))

    frame_files = sorted(f for f in os.listdir(cache_dir) if f.startswith('frame_'))
    n_frames = len(frame_files)

    # tracks[track_id] = list of (frame, z, y, x)
    tracks = {}
    next_track_id = 0

    # active_tracks[cluster_representative_key] = track_id  — reset each frame
    # We use a list indexed by cluster index within a frame
    active = {}     # cluster index at t → track_id

    for t in range(n_frames - 1):
        data0 = np.load(os.path.join(cache_dir, f'frame_{t:04d}.npz'))
        data1 = np.load(os.path.join(cache_dir, f'frame_{t+1:04d}.npz'))

        c0, ids0, patches0 = data0['centers'], data0['cell_ids'], data0['patches']
        c1, ids1, patches1 = data1['centers'], data1['cell_ids'], data1['patches']

        if len(c0) == 0 or len(c1) == 0:
            active = {}
            continue

        N0 = len(c0)
        all_c = np.vstack([c0, c1]).astype(np.float32)
        sc    = _scale(all_c, cfg.z_anisotropy)
        frame_flags = np.array([0]*N0 + [len(c1)*0 + 1]*len(c1), dtype=np.float32)
        frame_flags[N0:] = 1
        positions = np.column_stack([sc, frame_flags]).astype(np.float32)

        all_patches = np.concatenate([patches0, patches1], axis=0)

        edge_index, edge_feat, _, _ = _build_graph(
            c0, ids0, c1, ids1, t, parents, p2c,
            cfg.r_intra, cfg.r_cross, cfg.z_anisotropy,
        )

        if edge_index.shape[1] == 0:
            active = {}
            continue

        patches_t  = torch.from_numpy(all_patches)
        pos_t      = torch.from_numpy(positions)
        ei_t       = torch.from_numpy(edge_index)
        ef_t       = torch.from_numpy(edge_feat)

        logits = model(patches_t, pos_t, ei_t, ef_t)
        scores = torch.sigmoid(logits).numpy()

        ff_int = frame_flags.astype(int)

        # Intra-frame clustering
        clusters0_list = intra_cluster(all_c, scores, edge_index, ff_int, 0, cfg.same_cell_threshold)
        clusters1_list = intra_cluster(all_c, scores, edge_index, ff_int, 1, cfg.same_cell_threshold)

        if not clusters0_list or not clusters1_list:
            active = {}
            continue

        reps0 = cluster_representative(clusters0_list, all_c)
        reps1 = cluster_representative(clusters1_list, all_c)

        # Adjust reps1 indices (they're offset by N0 in all_c)
        reps1_local = cluster_representative(clusters1_list, all_c)

        aff = cross_frame_scores(clusters0_list, clusters1_list, scores, edge_index)

        assignments = match_cross_frame(
            aff, clusters0_list, clusters1_list, reps0, reps1_local,
            cfg.same_cell_threshold, cfg.r_cross, cfg.z_anisotropy,
        )

        # --- propagate tracks ---
        # Initialise tracks for frame t=0
        if t == 0:
            for k0 in range(len(clusters0_list)):
                tid = next_track_id; next_track_id += 1
                pos = tuple(reps0[k0].tolist())
                tracks[tid] = [(t, *pos)]
                active[k0] = tid

        # Build new active map for t+1
        new_active = {}
        matched_k1 = set()

        for k0, k1s in assignments:
            tid = active.get(k0, None)
            if tid is None:
                tid = next_track_id; next_track_id += 1
                pos0 = tuple(reps0[k0].tolist())
                tracks[tid] = [(t, *pos0)]

            if len(k1s) == 1:
                k1 = k1s[0]
                pos1 = tuple(reps1[k1].tolist())
                tracks[tid].append((t + 1, *pos1))
                new_active[k1] = tid
                matched_k1.add(k1)
            else:
                # Mitosis: close parent track, start two daughter tracks
                tracks[tid].append((t, *tuple(reps0[k0].tolist())))  # mark last frame
                for k1 in k1s:
                    dtid = next_track_id; next_track_id += 1
                    pos1 = tuple(reps1[k1].tolist())
                    tracks[dtid] = [(t + 1, *pos1)]
                    new_active[k1] = dtid
                    matched_k1.add(k1)

        # Unmatched clusters at t+1 → new tracks
        for k1 in range(len(clusters1_list)):
            if k1 not in matched_k1:
                tid = next_track_id; next_track_id += 1
                pos1 = tuple(reps1[k1].tolist())
                tracks[tid] = [(t + 1, *pos1)]
                new_active[k1] = tid

        active = new_active

        if t % 50 == 0:
            print(f'  frame {t}/{n_frames}: {len(clusters0_list)} clusters-t, '
                  f'{len(clusters1_list)} clusters-(t+1), '
                  f'{len(assignments)} assignments')

    result = {str(tid): tpts for tid, tpts in tracks.items()}
    if out_path:
        import json
        with open(out_path, 'w') as f:
            json.dump(result, f)
        print(f'Tracks saved to {out_path}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',  default='0515')
    parser.add_argument('--ckpt', default='cache/checkpoints/best.pt')
    parser.add_argument('--out',  default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.ckpt and not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(cfg.data_root, args.ckpt)

    out = args.out or os.path.join(cfg.cache_dir, f'tracks_{args.exp}.json')
    run_inference(cfg, args.exp, args.ckpt, out)
