"""
Find the optimal three-gate mitosis thresholds that maximise TP divisions
while keeping FP = 0.

Three gates (all must pass to call a division):
  1. Mother head  : cluster0 mit_score       >= th_mit
  2. Daughter aff : min aff to both daughters >= th_aff
  3. Daughter head: min dau_score of both     >= th_dau

Algorithm:
  - Run the model once on the val set (0515) to collect per-candidate scores.
  - For each potential mother cluster k0 (any with >=2 reachable daughters):
      record (mit_score, min_aff_top2, min_dau_top2, is_gt_tp)
  - Grid-search (th_mit, th_aff, th_dau):
      count FP and TP; keep only FP=0 combos; maximise TP.

Usage:
    python -m tracking.tune_mito_thresholds
"""

import os
import json
from collections import defaultdict, Counter

import numpy as np
import torch

from tracking.config import Config
from tracking.dataset import _build_graph, _scale
from tracking.model import TrackingNet
from tracking.infer import intra_cluster, cluster_representative, cross_frame_scores
from tracking.preprocess import EXPERIMENT_DIRS


# ---------------------------------------------------------------------------

def majority_cell_id(cluster_local_indices, cell_ids_arr):
    cids = [int(cell_ids_arr[i]) for i in cluster_local_indices if cell_ids_arr[i] > 0]
    if not cids:
        return 0
    return Counter(cids).most_common(1)[0][0]


@torch.no_grad()
def collect_candidates(cfg, model, device, exp_id, gt_only=False):
    """
    For every frame pair in exp_id, find all potential division candidates
    (any cluster that has >=2 daughters within r_cross by affinity) and
    record their three gate scores plus the GT label.

    Returns list of dicts with keys:
        t, mit, min_aff, min_dau, is_gt_tp
    Also returns n_gt_total (total GT division events in this experiment).
    """
    dirs = EXPERIMENT_DIRS[exp_id]
    mit_path = os.path.join(cfg.data_root, dirs['mitosis'])
    with open(mit_path) as f:
        mit = json.load(f)
    parents_gt = {int(k): v for k, v in mit['Parents'].items()}
    p2c_gt = {}
    for cid, info in mit['Children'].items():
        p2c_gt.setdefault(info['ParentID'], []).append(int(cid))

    # GT divisions indexed by the frame where the parent's LAST frame is t
    gt_by_t = defaultdict(list)   # t -> [(parent_cid, {daughter_cids})]
    for pid, pinfo in parents_gt.items():
        daus = p2c_gt.get(pid, [])
        if len(daus) >= 2:
            gt_by_t[pinfo['LastFrame']].append((pid, set(daus)))
    n_gt_total = sum(len(v) for v in gt_by_t.values())

    cache_dir = os.path.join(cfg.cache_dir, exp_id)
    frame_files = sorted(f for f in os.listdir(cache_dir) if f.startswith('frame_'))
    n_frames = len(frame_files)

    candidates = []
    gt_tp_seen = set()   # (t, parent_cid) already counted as reachable

    for t in range(n_frames - 1):
        d0 = np.load(os.path.join(cache_dir, f'frame_{t:04d}.npz'))
        d1 = np.load(os.path.join(cache_dir, f'frame_{t+1:04d}.npz'))
        c0, ids0, p0 = d0['centers'], d0['cell_ids'], d0['patches']
        c1, ids1, p1 = d1['centers'], d1['cell_ids'], d1['patches']

        if len(c0) == 0 or len(c1) == 0:
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
            continue

        edge_logits, mit_logits, fg_logits, dau_logits = model(
            torch.from_numpy(patches).to(device),
            torch.from_numpy(pos).to(device),
            torch.from_numpy(edge_index).to(device),
            torch.from_numpy(edge_feat).to(device),
        )
        scores     = torch.sigmoid(edge_logits).cpu().numpy()
        mit_scores = torch.sigmoid(mit_logits).cpu().numpy()
        fg_scores  = torch.sigmoid(fg_logits).cpu().numpy()
        dau_scores = torch.sigmoid(dau_logits).cpu().numpy()

        ff_int = ff.astype(np.int32).copy()
        if gt_only:
            # Use GT cell_id > 0 as foreground mask (oracle mode, no FP background)
            all_ids = np.concatenate([ids0, ids1])
            ff_int[all_ids == 0] = -1
        else:
            ff_int[fg_scores < cfg.fg_threshold] = -1

        cl0_g = intra_cluster(all_c, scores, edge_index, ff_int, 0, cfg.intra_threshold)
        cl1_g = intra_cluster(all_c, scores, edge_index, ff_int, 1, cfg.intra_threshold)
        if not cl0_g or not cl1_g:
            continue

        reps0 = cluster_representative(cl0_g, all_c)
        reps1 = cluster_representative(cl1_g, all_c)

        cl0_mit  = np.array([mit_scores[list(cl)].mean() for cl in cl0_g])
        cl1_dau  = np.array([dau_scores[list(cl)].mean() for cl in cl1_g])
        aff      = cross_frame_scores(cl0_g, cl1_g, scores, edge_index)

        # Isotropic distance matrix
        sc0 = reps0.copy(); sc0[:, 0] *= cfg.z_anisotropy
        sc1 = reps1.copy(); sc1[:, 0] *= cfg.z_anisotropy
        dist = np.linalg.norm(sc0[:, None] - sc1[None], axis=-1)  # (K0, K1)

        # Majority cell-id per cluster
        # cl0_g has global indices 0..N0-1  → index ids0 directly
        # cl1_g has global indices N0..N0+N1-1 → subtract N0 to index ids1
        cl0_cid = [majority_cell_id(cl,            ids0) for cl in cl0_g]
        cl1_cid = [majority_cell_id([i-N0 for i in cl], ids1) for cl in cl1_g]

        gt_divs = gt_by_t.get(t, [])  # [(parent_cid, {dau_cids})]

        K0, K1 = len(cl0_g), len(cl1_g)
        for k0 in range(K0):
            # Collect all daughters within r_cross (no threshold applied here)
            cands_k1 = [
                (float(aff[k0, k1]), k1)
                for k1 in range(K1)
                if dist[k0, k1] <= cfg.r_cross
            ]
            if len(cands_k1) < 2:
                continue

            cands_k1.sort(reverse=True)
            (aff_a, k1a), (aff_b, k1b) = cands_k1[0], cands_k1[1]
            min_aff = min(aff_a, aff_b)
            min_dau = min(float(cl1_dau[k1a]), float(cl1_dau[k1b]))
            mit_sc  = float(cl0_mit[k0])

            # GT-TP check: k0's cell is a GT divider AND top-2 daughters are both GT daughters
            parent_cid = cl0_cid[k0]
            dau_a_cid  = cl1_cid[k1a]
            dau_b_cid  = cl1_cid[k1b]

            is_gt_tp = False
            for pid, gt_daus in gt_divs:
                if (parent_cid == pid
                        and dau_a_cid in gt_daus
                        and dau_b_cid in gt_daus):
                    is_gt_tp = True
                    gt_tp_seen.add((t, pid))
                    break

            candidates.append(dict(t=t, mit=mit_sc, min_aff=min_aff,
                                   min_dau=min_dau, is_gt_tp=is_gt_tp))

        if t % 50 == 0:
            print(f'  frame {t:03d}/{n_frames}  candidates so far: {len(candidates)}')

    n_gt_reachable = len(gt_tp_seen)
    return candidates, n_gt_total, n_gt_reachable


def grid_search(candidates, n_gt_total, n_gt_reachable):
    tp_cands = [c for c in candidates if     c['is_gt_tp']]
    fp_cands = [c for c in candidates if not c['is_gt_tp']]

    tp_mit = np.array([c['mit']     for c in tp_cands])
    tp_aff = np.array([c['min_aff'] for c in tp_cands])
    tp_dau = np.array([c['min_dau'] for c in tp_cands])

    fp_mit = np.array([c['mit']     for c in fp_cands]) if fp_cands else np.zeros(0)
    fp_aff = np.array([c['min_aff'] for c in fp_cands]) if fp_cands else np.zeros(0)
    fp_dau = np.array([c['min_dau'] for c in fp_cands]) if fp_cands else np.zeros(0)

    th_mit_vals = np.round(np.arange(0.40, 1.00, 0.02), 3)
    th_aff_vals = np.round(np.arange(0.20, 1.00, 0.02), 3)
    th_dau_vals = np.round(np.arange(0.05, 1.00, 0.05), 3)

    best_tp = -1
    best_combos = []

    total = len(th_mit_vals) * len(th_aff_vals) * len(th_dau_vals)
    print(f'Grid search: {total:,} combinations...')

    for th_mit in th_mit_vals:
        # Pre-filter FP candidates by mit threshold
        if len(fp_mit) > 0:
            fp_mit_pass = fp_mit >= th_mit
        else:
            fp_mit_pass = np.zeros(0, dtype=bool)

        tp_mit_pass = tp_mit >= th_mit

        for th_aff in th_aff_vals:
            if len(fp_mit) > 0:
                fp_ma_pass = fp_mit_pass & (fp_aff >= th_aff)
            else:
                fp_ma_pass = np.zeros(0, dtype=bool)
            tp_ma_pass = tp_mit_pass & (tp_aff >= th_aff)

            for th_dau in th_dau_vals:
                # FP count
                n_fp = int((fp_ma_pass & (fp_dau >= th_dau)).sum()) if len(fp_mit) > 0 else 0
                if n_fp > 0:
                    continue

                n_tp = int((tp_ma_pass & (tp_dau >= th_dau)).sum())

                if n_tp > best_tp:
                    best_tp = n_tp
                    best_combos = [(th_mit, th_aff, th_dau)]
                elif n_tp == best_tp and n_tp > 0:
                    best_combos.append((th_mit, th_aff, th_dau))

    return best_tp, best_combos


def main(ckpt_override=None, gt_only=False):
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt = ckpt_override or os.path.join(cfg.cache_dir, 'checkpoints', 'best.pt')
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(cfg.data_root, ckpt)
    model = TrackingNet(feat_dim=cfg.feat_dim, gnn_layers=cfg.gnn_layers).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f'Loaded: {ckpt}')

    exp_id = cfg.val_exps[0]   # '0515'
    print(f'Val experiment: {exp_id}')
    print(f'GT-only mode  : {gt_only}')

    candidates, n_gt_total, n_gt_reachable = collect_candidates(
        cfg, model, device, exp_id, gt_only=gt_only)

    n_fp_cands = sum(1 for c in candidates if not c['is_gt_tp'])
    n_tp_cands = sum(1 for c in candidates if     c['is_gt_tp'])

    print()
    print(f'Candidates collected : {len(candidates)}')
    print(f'  GT-TP (reachable)  : {n_tp_cands}  ({n_gt_reachable}/{n_gt_total} GT divisions reachable)')
    print(f'  FP                 : {n_fp_cands}')
    print()

    # Print score distributions for GT-TP vs FP
    if n_tp_cands:
        tp = [c for c in candidates if c['is_gt_tp']]
        print('GT-TP score ranges:')
        print(f'  mit     : {min(c["mit"]     for c in tp):.3f} – {max(c["mit"]     for c in tp):.3f}')
        print(f'  min_aff : {min(c["min_aff"] for c in tp):.3f} – {max(c["min_aff"] for c in tp):.3f}')
        print(f'  min_dau : {min(c["min_dau"] for c in tp):.3f} – {max(c["min_dau"] for c in tp):.3f}')
    if n_fp_cands:
        fp = [c for c in candidates if not c['is_gt_tp']]
        print('FP score ranges:')
        print(f'  mit     : {min(c["mit"]     for c in fp):.3f} – {max(c["mit"]     for c in fp):.3f}')
        print(f'  min_aff : {min(c["min_aff"] for c in fp):.3f} – {max(c["min_aff"] for c in fp):.3f}')
        print(f'  min_dau : {min(c["min_dau"] for c in fp):.3f} – {max(c["min_dau"] for c in fp):.3f}')
    print()

    best_tp, best_combos = grid_search(candidates, n_gt_total, n_gt_reachable)

    print()
    print('=' * 60)
    print('  RESULTS')
    print('=' * 60)
    print(f'  GT divisions total        : {n_gt_total}')
    print(f'  GT divisions reachable    : {n_gt_reachable}  (top-2 daughters within r_cross)')
    print(f'  Best TP achievable (FP=0) : {best_tp}')
    if n_gt_total > 0:
        print(f'  Div-Recall at best        : {best_tp}/{n_gt_total} = {best_tp/n_gt_total:.3f}')
    print()

    if best_combos:
        # Recommend: most permissive (lowest thresholds = lowest sum) for generalisability
        best_combos_arr = np.array(best_combos)
        # Sort: primary by -TP (all equal), secondary by sum of thresholds (ascending)
        order = np.argsort(best_combos_arr.sum(axis=1))
        most_permissive = best_combos_arr[order[0]]
        most_strict     = best_combos_arr[order[-1]]

        print(f'  Most permissive (lowest thresholds):')
        print(f'    mitosis_head_threshold  = {most_permissive[0]:.2f}')
        print(f'    mitosis_threshold       = {most_permissive[1]:.2f}')
        print(f'    daughter_head_threshold = {most_permissive[2]:.2f}')
        print()
        print(f'  Most strict (highest thresholds):')
        print(f'    mitosis_head_threshold  = {most_strict[0]:.2f}')
        print(f'    mitosis_threshold       = {most_strict[1]:.2f}')
        print(f'    daughter_head_threshold = {most_strict[2]:.2f}')
        print()
        print(f'  Total FP=0 combinations found: {len(best_combos)}')
    else:
        print('  No threshold combination achieves FP=0 with any TP.')
        print('  The model needs retraining to separate GT-TP from FP candidates.')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None,
                        help='checkpoint path (default: cache/checkpoints/best.pt)')
    parser.add_argument('--gt-only', action='store_true',
                        help='use GT cell_id>0 as FG mask instead of predicted FG head')
    args = parser.parse_args()
    main(ckpt_override=args.ckpt, gt_only=args.gt_only)
