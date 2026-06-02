"""
Diagnose which gate blocks each GT division event.

Usage:
    python -m tracking.diagnose_mitosis
"""

import json, os
import numpy as np
import torch

from tracking.config import Config
from tracking.model import TrackingNet
from tracking.dataset import _build_graph, _scale
from tracking.infer import intra_cluster, cross_frame_scores

cfg = Config()
exp = '0515'
ckpt = 'cache/checkpoints/best.pt'
cache_dir = os.path.join(cfg.cache_dir, exp)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = TrackingNet(feat_dim=cfg.feat_dim, gnn_layers=cfg.gnn_layers).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

with open(os.path.join(cfg.data_root, 'mitotic_events/mitosis_info_0515.json')) as f:
    mit = json.load(f)
parents  = {int(k): v for k, v in mit['Parents'].items()}
children = {int(k): v for k, v in mit['Children'].items()}
p2c = {}
for cid, info in children.items():
    p2c.setdefault(info['ParentID'], []).append(int(cid))

results = []

for pid, pinfo in parents.items():
    t = pinfo['LastFrame']
    dids = p2c.get(pid, [])
    if len(dids) < 2:
        continue

    d0 = np.load(os.path.join(cache_dir, f'frame_{t:04d}.npz'))
    d1 = np.load(os.path.join(cache_dir, f'frame_{t+1:04d}.npz'))
    c0, ids0, p0 = d0['centers'], d0['cell_ids'], d0['patches']
    c1, ids1, p1 = d1['centers'], d1['cell_ids'], d1['patches']

    if len(c0) == 0 or len(c1) == 0:
        continue

    N0 = len(c0)
    all_c   = np.vstack([c0, c1]).astype(np.float32)
    sc      = _scale(all_c, cfg.z_anisotropy)
    ff      = np.array([0]*N0 + [1]*len(c1), dtype=np.float32)
    pos     = np.column_stack([sc, ff]).astype(np.float32)
    patches = np.concatenate([p0, p1], axis=0)

    edge_index, edge_feat, _, _ = _build_graph(
        c0, ids0, c1, ids1, t, parents, p2c,
        cfg.r_intra, cfg.r_cross, cfg.z_anisotropy)

    if edge_index.shape[1] == 0:
        continue

    with torch.no_grad():
        edge_logits, mit_logits, fg_logits, daughter_logits = model(
            torch.from_numpy(patches).to(device),
            torch.from_numpy(pos).to(device),
            torch.from_numpy(edge_index).to(device),
            torch.from_numpy(edge_feat).to(device),
            fg_gt=None,
        )
    scores     = torch.sigmoid(edge_logits).cpu().numpy()
    mit_scores = torch.sigmoid(mit_logits).cpu().numpy()
    dau_scores = torch.sigmoid(daughter_logits).cpu().numpy()
    fg_scores  = torch.sigmoid(fg_logits).cpu().numpy()

    ff_int = ff.astype(np.int32).copy()
    ff_int[fg_scores < cfg.fg_threshold] = -1

    cl0 = intra_cluster(all_c, scores, edge_index, ff_int, 0, cfg.intra_threshold)
    cl1 = intra_cluster(all_c, scores, edge_index, ff_int, 1, cfg.intra_threshold)
    if not cl0 or not cl1:
        continue

    aff = cross_frame_scores(cl0, cl1, scores, edge_index)

    m_idx = np.where(ids0 == pid)[0]
    if len(m_idx) == 0:
        continue
    m_node    = int(m_idx[0])
    mother_cl = next((k for k, cl in enumerate(cl0) if m_node in cl), None)
    if mother_cl is None:
        continue  # mother filtered out by FG head

    mit_score = mit_scores[m_node]

    d_info = []
    for did in dids:
        d_idx = np.where(ids1 == did)[0]
        if len(d_idx) == 0:
            d_info.append({'found': False, 'did': did})
            continue
        d_node    = int(d_idx[0]) + N0
        dau_score = dau_scores[d_node]
        dau_cl    = next((k for k, cl in enumerate(cl1) if d_node in cl), None)
        edge_aff  = float(aff[mother_cl, dau_cl]) if dau_cl is not None else 0.0
        d_info.append({'found': True, 'did': did,
                       'dau_score': float(dau_score),
                       'dau_cl': dau_cl,
                       'edge_aff': edge_aff})

    results.append({'pid': pid, 't': t,
                    'mit_score': float(mit_score),
                    'mother_cl': mother_cl,
                    'daughters': d_info})

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
mit_thr = cfg.mitosis_head_threshold   # 0.3
dau_thr = 0.1
aff_thr = cfg.mitosis_threshold        # 0.4

blocked_by_mit  = 0
blocked_by_aff  = 0
blocked_by_dau  = 0
blocked_by_miss = 0
detected        = 0

print(f'\nGT divisions analyzed: {len(results)}')
print(f'Thresholds — mit_head: {mit_thr}  edge_aff: {aff_thr}  dau_head: {dau_thr}')
print()
print(f"{'t':>4}  {'pid':>6}  {'mit':>5}  {'d1_aff':>6}  {'d1_dau':>6}  {'d2_aff':>6}  {'d2_dau':>6}  status")
print('-' * 78)

for r in sorted(results, key=lambda x: x['t']):
    mit = r['mit_score']
    ds  = r['daughters']
    t   = r['t']
    pid = r['pid']

    if any(not d['found'] for d in ds):
        blocked_by_miss += 1
        print(f"{t:4d}  {pid:6d}  {mit:5.3f}  {'--':>6}  {'--':>6}  {'--':>6}  {'--':>6}  MISSING_IN_CACHE")
        continue

    affs = [d['edge_aff'] for d in ds]
    daus = [d['dau_score'] for d in ds]

    if mit < mit_thr:
        status = 'BLOCK:mother_head'
        blocked_by_mit += 1
    elif any(a < aff_thr for a in affs):
        status = 'BLOCK:edge_aff'
        blocked_by_aff += 1
    elif any(d < dau_thr for d in daus):
        status = 'BLOCK:dau_head'
        blocked_by_dau += 1
    else:
        status = 'OK'
        detected += 1

    d1 = ds[0]
    d2 = ds[1] if len(ds) > 1 else ds[0]
    print(f"{t:4d}  {pid:6d}  {mit:5.3f}  "
          f"{d1['edge_aff']:6.3f}  {d1['dau_score']:6.3f}  "
          f"{d2['edge_aff']:6.3f}  {d2['dau_score']:6.3f}  {status}")

print()
print(f"Detected  (all gates pass) : {detected}")
print(f"Blocked by mother head     : {blocked_by_mit}  (mit < {mit_thr})")
print(f"Blocked by edge affinity   : {blocked_by_aff}  (any daughter aff < {aff_thr})")
print(f"Blocked by daughter head   : {blocked_by_dau}  (any daughter dau < {dau_thr})")
print(f"Missing in cache           : {blocked_by_miss}")
