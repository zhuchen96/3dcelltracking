"""
Inference → Cell Tracking Challenge (CTC) output using SimpleTrackingNet.

Division detection uses sister scores (two daughters look similar to each other)
instead of the separate mitosis head used in TrackingNet.  No FG head gating —
score-gated message passing in the GNN suppresses background detections.

Division criterion for frame-t cluster k0:
  1. Both conn_aff[k0, k1a] >= mitosis_threshold
        conn_aff[k0, k1b] >= mitosis_threshold
  2. sister_score[k1a, k1b] >= sister_threshold
  → division (k0 → [k1a, k1b])

Best sister pair per mother is chosen by aff[k0,k1a] + aff[k0,k1b] + sister_score.
Assignments are greedy (highest-scoring division first).

Usage:
    python -m tracking.infer_ctc_simple --exp 0515 \\
        --ckpt cache/checkpoints_simple/best.pt \\
        --sister-threshold 0.5
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
from tracking.model import SimpleTrackingNet
from tracking.preprocess import EXPERIMENT_DIRS, sorted_tifs


# ---------------------------------------------------------------------------
# Union-Find (for intra-frame clustering)
# ---------------------------------------------------------------------------

class _UF:
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


def intra_cluster(centers, scores, edge_index, frame_flags, frame_id, threshold):
    """Cluster detections within one frame using high-scoring intra-frame edges."""
    node_mask  = (frame_flags == frame_id)
    global_idx = np.where(node_mask)[0]
    n_local    = len(global_idx)
    if n_local == 0:
        return []
    g2l = {g: l for l, g in enumerate(global_idx)}
    uf  = _UF(n_local)
    src, dst = edge_index[0], edge_index[1]
    for e in range(len(scores)):
        s, d = int(src[e]), int(dst[e])
        if frame_flags[s] != frame_id or frame_flags[d] != frame_id:
            continue
        if scores[e] >= threshold:
            uf.union(g2l[s], g2l[d])
    clusters = {}
    for l, g in enumerate(global_idx):
        clusters.setdefault(uf.find(l), []).append(g)
    return list(clusters.values())


def cluster_representative(clusters, centers):
    """Mean ZYX position for each cluster."""
    return np.array([centers[c].mean(0) for c in clusters])


def cross_frame_scores(clusters0, clusters1, scores, edge_index):
    """Mean edge probability between each pair of (frame-t, frame-t+1) clusters."""
    K0, K1 = len(clusters0), len(clusters1)
    if K0 == 0 or K1 == 0:
        return np.zeros((K0, K1))
    set0 = [set(c) for c in clusters0]
    set1 = [set(c) for c in clusters1]
    aff  = np.zeros((K0, K1))
    cnt  = np.zeros((K0, K1))
    src, dst = edge_index[0], edge_index[1]
    for e in range(len(scores)):
        s, d = int(src[e]), int(dst[e])
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


# ---------------------------------------------------------------------------
# Sister score matrix between frame-(t+1) cluster pairs
# ---------------------------------------------------------------------------

def sister_cluster_scores(clusters1, sister_scores, edge_index, N0):
    """
    Mean sister-head score for each pair of frame-(t+1) clusters.

    Only intra-t+1 edges (both nodes >= N0) contribute.
    Returns (K1, K1) symmetric float matrix.
    """
    K1 = len(clusters1)
    mat = np.zeros((K1, K1), dtype=np.float32)
    cnt = np.zeros((K1, K1), dtype=np.float32)
    if K1 == 0:
        return mat

    node2cl = {}
    for k, cl in enumerate(clusters1):
        for g in cl:
            node2cl[g] = k

    src, dst = edge_index[0], edge_index[1]
    for e in range(len(sister_scores)):
        s, d = int(src[e]), int(dst[e])
        if s < N0 or d < N0:
            continue
        ka = node2cl.get(s, -1)
        kb = node2cl.get(d, -1)
        if ka < 0 or kb < 0 or ka == kb:
            continue
        mat[ka, kb] += sister_scores[e]
        cnt[ka, kb] += 1

    mask = cnt > 0
    mat[mask] /= cnt[mask]
    return mat


# ---------------------------------------------------------------------------
# Cross-frame matching with sister-based division detection
# ---------------------------------------------------------------------------

def match_cross_frame_simple(aff, sister_mat, clusters0, clusters1, reps0, reps1,
                              threshold, mitosis_threshold, sister_threshold,
                              r_cross, z_anisotropy):
    """
    1-to-1 (normal link) or 1-to-2 (division) assignment.

    Division pre-pass runs before Hungarian to reserve daughters.
    Division candidates scored by aff[k0,k1a] + aff[k0,k1b] + sister_mat[k1a,k1b].
    Greedy assignment: highest-scoring candidate wins.

    Returns list of (k0, [k1, ...]) tuples.
    """
    K0, K1 = len(clusters0), len(clusters1)
    if K0 == 0 or K1 == 0:
        return []

    sc0 = reps0.copy(); sc0[:, 0] *= z_anisotropy
    sc1 = reps1.copy(); sc1[:, 0] *= z_anisotropy
    dist_matrix = np.linalg.norm(sc0[:, None] - sc1[None], axis=-1)

    cost = 1.0 - aff.copy()
    cost[dist_matrix > r_cross] = 1.0

    # --- Division pre-pass ---
    div_candidates = []
    for k0 in range(K0):
        eligible = [
            k1 for k1 in range(K1)
            if (aff[k0, k1] >= mitosis_threshold and
                dist_matrix[k0, k1] <= r_cross)
        ]
        if len(eligible) < 2:
            continue
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                k1a, k1b = eligible[i], eligible[j]
                s = sister_mat[k1a, k1b]
                if s >= sister_threshold:
                    score = aff[k0, k1a] + aff[k0, k1b] + s
                    div_candidates.append((score, k0, k1a, k1b))

    div_candidates.sort(reverse=True)

    reserved_k0 = set()
    reserved_k1 = set()
    division_assignments = []
    for score, k0, k1a, k1b in div_candidates:
        if k0 in reserved_k0 or k1a in reserved_k1 or k1b in reserved_k1:
            continue
        division_assignments.append((k0, [k1a, k1b]))
        reserved_k0.add(k0)
        reserved_k1.add(k1a)
        reserved_k1.add(k1b)

    # --- 1-to-1 Hungarian on remaining clusters ---
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
# Drawing helper
# ---------------------------------------------------------------------------

def _draw_ellipsoid(mask, center_zyx, radius_zyx, value):
    Z, Y, X = mask.shape
    z0 = int(round(center_zyx[0]))
    y0 = int(round(center_zyx[1]))
    x0 = int(round(center_zyx[2]))
    rz, ry, rx = radius_zyx
    z_lo = max(0, z0 - rz);  z_hi = min(Z, z0 + rz + 1)
    y_lo = max(0, y0 - ry);  y_hi = min(Y, y0 + ry + 1)
    x_lo = max(0, x0 - rx);  x_hi = min(X, x0 + rx + 1)
    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    inside = (((zz - z0) / rz) ** 2 +
              ((yy - y0) / ry) ** 2 +
              ((xx - x0) / rx) ** 2) <= 1.0
    mask[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][inside] = np.uint16(value)


def _cluster_local(clusters_global, offset):
    return [[i - offset for i in cl] for cl in clusters_global]


def _cluster_tid(cluster_local, det_track):
    # Bounds-check guards against phantom node indices that exceed det_track size.
    tids = [int(det_track[i]) for i in cluster_local
            if i < len(det_track) and det_track[i] > 0]
    return Counter(tids).most_common(1)[0][0] if tids else 0


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ctc_inference_simple(cfg, exp_id, ckpt_path, out_dir,
                              sister_threshold=0.5,
                              mitosis_threshold=None,
                              radius_zyx=(3, 6, 6)):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = SimpleTrackingNet(
        feat_dim=cfg.feat_dim, gnn_layers=cfg.gnn_layers, in_channels=cfg.in_channels
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f'Loaded:           {ckpt_path}')
    if mitosis_threshold is None:
        mitosis_threshold = cfg.mitosis_threshold
    print(f'fg_threshold:     {cfg.fg_threshold}')
    print(f'intra_threshold:  {cfg.intra_threshold}')
    print(f'cross_threshold:  {cfg.cross_threshold}')
    print(f'mitosis_threshold:{mitosis_threshold}')
    print(f'sister_threshold: {sister_threshold}')
    print(f'min_track_length: {cfg.min_track_length}')

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

    ref_shape = tifffile.imread(
        sorted_tifs(os.path.join(cfg.data_root, dirs['mask']))[0]
    ).shape

    det_track      = {}
    frame_clusters = {t: [] for t in range(n_frames)}
    track_meta     = {}
    next_tid       = 1

    # Phantom node state
    # phantom_pool[tid] = {pos, h_feat, vel, frames_remaining}
    phantom_pool    = {}
    prev_track_reps = {}   # tid -> last known ZYX position (for velocity)

    for t in range(n_frames - 1):
        d0 = np.load(os.path.join(cache_dir, f'frame_{t:04d}.npz'))
        d1 = np.load(os.path.join(cache_dir, f'frame_{t+1:04d}.npz'))

        c0, ids0, p0 = d0['centers'], d0['cell_ids'], d0['patches']
        c1, ids1, p1 = d1['centers'], d1['cell_ids'], d1['patches']

        if len(c0) == 0 or len(c1) == 0:
            det_track.pop(t, None)
            phantom_pool.clear()
            continue

        N0_real, N1 = len(c0), len(c1)

        # ---- Inject phantom nodes into frame-t source ----
        ph_list = list(phantom_pool.items())   # [(tid, ph_data), ...]
        P       = len(ph_list)
        ph_tids = [tid for tid, _ in ph_list]

        if P > 0:
            ph_proj = np.array([ph['pos'] + ph['vel']
                                for _, ph in ph_list], dtype=np.float32)
            c0_aug  = np.vstack([c0, ph_proj])
            p0_aug  = np.zeros((N0_real + P, *p0.shape[1:]), dtype=p0.dtype)
            p0_aug[:N0_real] = p0
            # ids: phantoms get 0 (no GT id; labels not used at inference)
            ids0_aug = np.zeros(N0_real + P, dtype=ids0.dtype)
            ids0_aug[:N0_real] = ids0
        else:
            c0_aug, p0_aug, ids0_aug = c0, p0, ids0

        N0 = len(c0_aug)

        # ---- Build graph ----
        edge_index, edge_feat, _, _ = _build_graph(
            c0_aug, ids0_aug, c1, ids1, t, parents_gt, p2c_gt,
            cfg.r_intra, cfg.r_cross, cfg.z_anisotropy,
        )
        if edge_index.shape[1] == 0:
            det_track.pop(t, None)
            continue

        all_c = np.vstack([c0_aug, c1]).astype(np.float32)
        sc    = _scale(all_c, cfg.z_anisotropy)
        ff    = np.array([0]*N0 + [1]*N1, dtype=np.float32)
        pos   = np.column_stack([sc, ff]).astype(np.float32)
        patches_all = np.concatenate([p0_aug, p1], axis=0)

        # ---- Encode → override phantom rows → GNN + classify ----
        h_cnn = model.encode(
            torch.from_numpy(patches_all[:, :cfg.in_channels]).to(device),
            torch.from_numpy(pos).to(device),
        )
        for ph_i, (_, ph) in enumerate(ph_list):
            h_cnn[N0_real + ph_i] = ph['h_feat'].to(device)

        conn_logits, sister_logits, fg_logits = model.forward_from_features(
            h_cnn,
            torch.from_numpy(edge_index).to(device),
            torch.from_numpy(edge_feat).to(device),
        )
        conn_scores   = torch.sigmoid(conn_logits).cpu().numpy()
        sister_scores = torch.sigmoid(sister_logits).cpu().numpy()
        fg_scores     = torch.sigmoid(fg_logits).cpu().numpy()

        # FG filter: real nodes use threshold; phantom nodes always pass
        ff_int = ff.astype(np.int32).copy()
        ff_int[fg_scores < cfg.fg_threshold] = -1
        for ph_i in range(P):
            ff_int[N0_real + ph_i] = 0

        # ---- Clustering ----
        cl0_g = intra_cluster(all_c, conn_scores, edge_index, ff_int, 0, cfg.intra_threshold)
        cl1_g = intra_cluster(all_c, conn_scores, edge_index, ff_int, 1, cfg.intra_threshold)

        if not cl0_g or not cl1_g:
            det_track.pop(t, None)
            continue

        cl0_loc = _cluster_local(cl0_g, 0)
        cl1_loc = _cluster_local(cl1_g, N0)
        reps0   = cluster_representative(cl0_g, all_c)
        reps1   = cluster_representative(cl1_g, all_c)

        # Global node index → phantom track ID (for phantom-containing clusters)
        ph_node_to_tid = {N0_real + i: ph_tids[i] for i in range(P)}

        # ---- Track IDs for frame-t clusters ----
        prev_track = det_track.get(t)
        new_track0 = np.zeros(N0_real, dtype=np.int32)
        cl0_tid    = []
        used_tids  = set()

        for k0, cl_loc in enumerate(cl0_loc):
            # Phantom in this cluster overrides the track ID lookup
            ph_tid = next(
                (ph_node_to_tid[g] for g in cl_loc if g in ph_node_to_tid), 0
            )
            if ph_tid > 0:
                tid = ph_tid
            elif prev_track is not None:
                tid = _cluster_tid(cl_loc, prev_track)
            else:
                tid = 0

            if tid == 0 or tid in used_tids:
                tid = next_tid; next_tid += 1
                track_meta[tid] = {'first': t, 'last': t, 'parent': 0}
                frame_clusters[t].append((tid, reps0[k0]))
            used_tids.add(tid)
            cl0_tid.append(tid)
            for i in cl_loc:
                if i < N0_real:
                    new_track0[i] = tid

        det_track[t] = new_track0

        # Cache mean h_cnn per frame-t cluster (needed to create phantoms below)
        cl0_h = {k0: h_cnn[list(cl)].mean(0).detach().cpu()
                 for k0, cl in enumerate(cl0_g)}

        # ---- Affinity matrices ----
        aff        = cross_frame_scores(cl0_g, cl1_g, conn_scores, edge_index)
        sister_mat = sister_cluster_scores(cl1_g, sister_scores, edge_index, N0)

        # ---- Match ----
        assignments = match_cross_frame_simple(
            aff, sister_mat, cl0_g, cl1_g, reps0, reps1,
            cfg.cross_threshold, mitosis_threshold,
            sister_threshold,
            cfg.r_cross, cfg.z_anisotropy,
        )

        assigned_k0 = {k0 for k0, _ in assignments}

        # ---- Update phantom pool ----

        # Phantoms whose cluster matched → track resumed, remove from pool
        for k0, _ in assignments:
            for node in cl0_g[k0]:
                if node in ph_node_to_tid:
                    phantom_pool.pop(ph_node_to_tid[node], None)

        # Phantoms that appeared in no cluster (filtered as BG or out of range)
        # → decrement directly; they'll be cleaned up below
        ph_in_cluster = {ph_node_to_tid[n]
                         for cl in cl0_g for n in cl if n in ph_node_to_tid}
        for tid in list(phantom_pool):
            if tid not in ph_in_cluster and tid in [t2 for t2, _ in ph_list]:
                phantom_pool[tid]['frames_remaining'] -= 1

        # Unmatched clusters → create or update phantoms
        for k0 in range(len(cl0_g)):
            if k0 in assigned_k0:
                continue
            tid      = cl0_tid[k0]
            pos_now  = reps0[k0].copy()

            if tid in phantom_pool:
                # Existing phantom still unmatched → update position, decrement
                old_pos = phantom_pool[tid]['pos'].copy()
                phantom_pool[tid]['pos']              = pos_now
                phantom_pool[tid]['vel']              = pos_now - old_pos
                phantom_pool[tid]['frames_remaining'] -= 1
            else:
                # Real cluster ending → create new phantom
                vel = pos_now - prev_track_reps.get(tid, pos_now)
                phantom_pool[tid] = {
                    'pos': pos_now,
                    'h_feat': cl0_h[k0],
                    'vel': vel,
                    'frames_remaining': cfg.phantom_max_frames,
                }

        # Expire dead phantoms
        for tid in [tid for tid, ph in phantom_pool.items()
                    if ph['frames_remaining'] <= 0]:
            del phantom_pool[tid]

        # Update velocity reference positions for next frame
        prev_track_reps = {cl0_tid[k0]: reps0[k0] for k0 in range(len(cl0_g))}

        # ---- Assign track IDs to frame-(t+1) clusters ----
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
                # Division: parent ends at t, daughters start at t+1
                track_meta[tid]['last'] = t
                for k1 in k1s:
                    dtid = next_tid; next_tid += 1
                    track_meta[dtid] = {'first': t + 1, 'last': t + 1, 'parent': tid}
                    for i in cl1_loc[k1]:
                        new_track1[i] = dtid
                    frame_clusters[t + 1].append((dtid, reps1[k1]))
                    matched_k1.add(k1)

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
            print(f'  pair {t:3d}/{n_frames-1}: '
                  f'det={N0_real:3d}+{P}ph/{N1:3d}  '
                  f'cl={len(cl0_g):3d}/{len(cl1_g):3d}  '
                  f'matched={len(assignments):3d}  mitoses={n_mit}  '
                  f'pool={len(phantom_pool)}')

    # -----------------------------------------------------------------------
    # Minimum track length filter
    # -----------------------------------------------------------------------
    if cfg.min_track_length > 1:
        keep = {tid for tid, m in track_meta.items()
                if (m['last'] - m['first'] + 1 >= cfg.min_track_length)
                or m['parent'] > 0}
        keep |= {track_meta[tid]['parent']
                 for tid in keep if track_meta[tid]['parent'] > 0}
        track_meta = {tid: m for tid, m in track_meta.items() if tid in keep}
        frame_clusters = {t: [(tid, rep) for tid, rep in fc if tid in keep]
                          for t, fc in frame_clusters.items()}
        n_removed = next_tid - 1 - len(track_meta)
        print(f'  Removed {n_removed} short tracks (length < {cfg.min_track_length}), '
              f'{len(track_meta)} remain.')

    # -----------------------------------------------------------------------
    # Write mask TIFFs
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
    parser.add_argument('--ckpt',   default='cache/checkpoints_simple/best.pt')
    parser.add_argument('--out',    default=None)
    parser.add_argument('--sister-threshold', type=float, default=0.5,
                        help='sister score threshold for division detection (default 0.5)')
    parser.add_argument('--mitosis-threshold', type=float, default=None,
                        help='conn affinity threshold for each daughter (default: cfg.mitosis_threshold)')
    parser.add_argument('--rz', type=int, default=3, help='ellipsoid Z radius (voxels)')
    parser.add_argument('--ry', type=int, default=6, help='ellipsoid Y radius (voxels)')
    parser.add_argument('--rx', type=int, default=6, help='ellipsoid X radius (voxels)')
    args = parser.parse_args()

    cfg = Config()
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(cfg.data_root, args.ckpt)
    out = args.out or os.path.join(cfg.cache_dir, f'ctc_simple_{args.exp}')
    run_ctc_inference_simple(cfg, args.exp, args.ckpt, out,
                             sister_threshold=args.sister_threshold,
                             mitosis_threshold=args.mitosis_threshold,
                             radius_zyx=(args.rz, args.ry, args.rx))
