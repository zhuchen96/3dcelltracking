"""
TrackingDataset: loads preprocessed frame triplets and builds detection graphs.

Each sample is a triplet (t-1, t, t+1) centred on frame t.
Nodes  = all detections in all three frames.
Edges  = intra-frame pairs (r_intra) + adjacent cross-frame pairs (r_cross).
         No edges between prev (t-1) and next (t+1).
Labels = 1 if same cell (or parent→daughter), 0 otherwise.

Sister edges: intra-next edges where both nodes are daughters of the same
parent that divides at frame t (the centre frame).

Position encoding: 5-D  [z_scaled, y, x, is_prev, is_next]
  prev nodes:   is_prev=1, is_next=0
  centre nodes: is_prev=0, is_next=0
  next nodes:   is_prev=0, is_next=1
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_mitosis(path):
    with open(path) as f:
        data = json.load(f)
    parents  = {int(k): v for k, v in data['Parents'].items()}
    children = {int(k): v for k, v in data['Children'].items()}
    p2c = {}
    for cid, info in children.items():
        pid = info['ParentID']
        p2c.setdefault(pid, []).append(cid)
    return parents, children, p2c


def _scale(centers, z_anisotropy):
    s = centers.copy()
    s[:, 0] *= z_anisotropy
    return s


# ---------------------------------------------------------------------------
# Legacy pair graph builder (kept for backward compatibility with old code)
# ---------------------------------------------------------------------------

def _build_graph(centers0, ids0, centers1, ids1, t, parents, p2c,
                 r_intra, r_cross, z_anisotropy):
    """Frame-pair (t, t+1) graph. Kept for backward compatibility."""
    N0, N1 = len(centers0), len(centers1)
    N = N0 + N1
    all_c  = np.vstack([centers0, centers1]).astype(np.float32)
    all_id = np.concatenate([ids0, ids1])
    frame  = np.array([0]*N0 + [1]*N1, dtype=np.int32)
    sc     = _scale(all_c, z_anisotropy)
    pos_cross = set()
    for pid, pinfo in parents.items():
        if pinfo['LastFrame'] == t:
            for did in p2c.get(pid, []):
                pos_cross.add((pid, did))
    diff  = sc[:, None, :] - sc[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    same_fr = (frame[:, None] == frame[None, :])
    in_range = (same_fr & (dists <= r_intra)) | (~same_fr & (dists <= r_cross))
    np.fill_diagonal(in_range, False)
    src_arr, dst_arr = np.where(in_range)
    if len(src_arr) == 0:
        empty = np.zeros(0, dtype=np.float32)
        return (np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32),
                empty, empty.astype(bool))
    d  = all_c[src_arr] - all_c[dst_arr]
    d[:, 0] *= z_anisotropy
    sf = (frame[src_arr] == frame[dst_arr]).astype(np.float32)
    edge_feat = np.column_stack([d, sf]).astype(np.float32)
    id_s = all_id[src_arr]; id_d = all_id[dst_arr]
    same_frame_edge = sf.astype(bool)
    labels = np.zeros(len(src_arr), dtype=np.float32)
    valid_mask = np.zeros(len(src_arr), dtype=bool)
    for e in range(len(src_arr)):
        is_, id_ = int(id_s[e]), int(id_d[e])
        if is_ == 0 or id_ == 0:
            continue
        valid_mask[e] = True
        if same_frame_edge[e]:
            labels[e] = float(is_ == id_)
        else:
            fi = int(frame[src_arr[e]])
            pid_cell, did_cell = (is_, id_) if fi == 0 else (id_, is_)
            labels[e] = float((pid_cell == did_cell) or ((pid_cell, did_cell) in pos_cross))
    edge_index = np.stack([src_arr, dst_arr], axis=0).astype(np.int64)
    return edge_index, edge_feat, labels, valid_mask


# ---------------------------------------------------------------------------
# Triplet graph builder
# ---------------------------------------------------------------------------

def _build_triplet_graph(centers_prev, ids_prev,
                          centers_curr, ids_curr,
                          centers_next, ids_next,
                          t_center, parents, p2c,
                          r_intra, r_cross, z_anisotropy):
    """
    Three-frame graph for triplet (t-1, t, t+1) with t_center = t.

    Frame tags: 0 = prev (t-1), 1 = center (t), 2 = next (t+1).
    Edges:  intra-frame within each frame (r_intra),
            cross-frame between adjacent pairs only (r_cross).
            No prev↔next edges.

    Returns
    -------
    edge_index    : (2, E) int64
    edge_feat     : (E, 4) float32  [dz_scaled, dy, dx, is_intra]
    labels        : (E,) float32    1 = same cell / parent→daughter
    valid_mask    : (E,) bool       True when both endpoints have GT id
    sister_labels : (E,) float32    1 = daughter pair of same parent at t_center
    valid_sister  : (E,) bool       True for intra-next GT edges
    """
    N_p, N_c, N_n = len(centers_prev), len(centers_curr), len(centers_next)
    N = N_p + N_c + N_n

    all_c  = np.vstack([centers_prev, centers_curr, centers_next]).astype(np.float32)
    all_id = np.concatenate([ids_prev, ids_curr, ids_next])
    frame  = np.array([0]*N_p + [1]*N_c + [2]*N_n, dtype=np.int32)
    sc     = _scale(all_c, z_anisotropy)

    # Only adjacent frame pairs (no prev↔next edges)
    frame_adj = (np.abs(frame[:, None] - frame[None, :]) <= 1)
    np.fill_diagonal(frame_adj, False)

    diff  = sc[:, None, :] - sc[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    same_fr  = (frame[:, None] == frame[None, :])
    in_range = frame_adj & (
        (same_fr  & (dists <= r_intra)) |
        (~same_fr & (dists <= r_cross))
    )

    src_arr, dst_arr = np.where(in_range)

    if len(src_arr) == 0:
        empty = np.zeros(0, dtype=np.float32)
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 4), dtype=np.float32),
                empty, empty.astype(bool),
                empty, empty.astype(bool))

    d        = all_c[src_arr] - all_c[dst_arr]
    d[:, 0] *= z_anisotropy
    is_intra  = (frame[src_arr] == frame[dst_arr]).astype(np.float32)
    edge_feat = np.column_stack([d, is_intra]).astype(np.float32)

    # Parent→daughter sets
    pos_prev_curr = set()   # parents dividing at t-1 → daughters at t
    for pid, pinfo in parents.items():
        if pinfo['LastFrame'] == t_center - 1:
            for did in p2c.get(pid, []):
                pos_prev_curr.add((pid, did))

    pos_curr_next = set()   # parents dividing at t → daughters at t+1
    for pid, pinfo in parents.items():
        if pinfo['LastFrame'] == t_center:
            for did in p2c.get(pid, []):
                pos_curr_next.add((pid, did))

    id_s = all_id[src_arr]
    id_d = all_id[dst_arr]
    fs   = frame[src_arr]
    fd   = frame[dst_arr]

    labels     = np.zeros(len(src_arr), dtype=np.float32)
    valid_mask = np.zeros(len(src_arr), dtype=bool)

    for e in range(len(src_arr)):
        is_, id_ = int(id_s[e]), int(id_d[e])
        if is_ == 0 or id_ == 0:
            continue
        valid_mask[e] = True
        fse, fde = int(fs[e]), int(fd[e])
        if fse == fde:                        # intra-frame
            labels[e] = float(is_ == id_)
        elif fse == 0 and fde == 1:           # prev→curr
            labels[e] = float(is_ == id_ or (is_, id_) in pos_prev_curr)
        elif fse == 1 and fde == 0:           # curr→prev
            labels[e] = float(is_ == id_ or (id_, is_) in pos_prev_curr)
        elif fse == 1 and fde == 2:           # curr→next
            labels[e] = float(is_ == id_ or (is_, id_) in pos_curr_next)
        elif fse == 2 and fde == 1:           # next→curr
            labels[e] = float(is_ == id_ or (id_, is_) in pos_curr_next)

    edge_index = np.stack([src_arr, dst_arr], axis=0).astype(np.int64)

    # Sister labels: intra-next (frame=2) edges
    dau_to_parent = {}
    for pid, pinfo in parents.items():
        if pinfo['LastFrame'] == t_center:
            for did in p2c.get(pid, []):
                dau_to_parent[did] = pid

    sister_labels = np.zeros(len(src_arr), dtype=np.float32)
    valid_sister  = np.zeros(len(src_arr), dtype=bool)
    for e in range(len(src_arr)):
        if int(fs[e]) == 2 and int(fd[e]) == 2:
            cid_s, cid_d = int(all_id[src_arr[e]]), int(all_id[dst_arr[e]])
            if cid_s > 0 and cid_d > 0:
                valid_sister[e] = True
                if (cid_s in dau_to_parent and cid_d in dau_to_parent
                        and dau_to_parent[cid_s] == dau_to_parent[cid_d]):
                    sister_labels[e] = 1.0

    return edge_index, edge_feat, labels, valid_mask, sister_labels, valid_sister


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrackingDataset(Dataset):
    """
    Each item is a frame triplet (t-1, t, t+1) centred on frame t.
    t ranges over [1, n-2] so all three frames always exist.
    """

    MITOSIS_PATHS = {
        '0001': 'mitotic_events/mitosis_info_0001.json',
        '0002': 'mitotic_events/mitosis_info_0002.json',
        '0003': 'mitotic_events/mitosis_info_0003.json',
        '0004': 'mitotic_events/mitosis_info_0004.json',
        '0501': 'mitotic_events/mitosis_info_0501.json',
        '0507': 'mitotic_events/mitosis_info_0507.json',
        '0515': 'mitotic_events/mitosis_info_0515.json',
        '0517': 'mitotic_events/mitosis_info_0517.json',
        '0522': 'mitotic_events/mitosis_info_0522.json',
        '0528': 'mitotic_events/mitosis_info_0528.json',
        '0605': 'mitotic_events/mitosis_info_0605.json',
    }

    def __init__(self, exp_ids, data_root, cache_dir,
                 r_intra=20., r_cross=30., z_anisotropy=0.5,
                 aug_drop_prob=0.0):
        self.cache_dir     = cache_dir
        self.data_root     = data_root
        self.r_intra       = r_intra
        self.r_cross       = r_cross
        self.z_anisotropy  = z_anisotropy
        self.aug_drop_prob = aug_drop_prob

        self.samples = []   # (exp_id, t_center)
        self.mitosis = {}   # exp_id -> (parents, children, p2c)

        for exp in exp_ids:
            mit_path = os.path.join(data_root, self.MITOSIS_PATHS[exp])
            self.mitosis[exp] = _load_mitosis(mit_path)

            exp_cache = os.path.join(cache_dir, exp)
            frames = sorted(f for f in os.listdir(exp_cache) if f.startswith('frame_'))
            n = len(frames)
            # Centre frames: 1 to n-2 so prev and next always exist
            for t in range(1, n - 1):
                self.samples.append((exp, t))

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, exp, t):
        path = os.path.join(self.cache_dir, exp, f'frame_{t:04d}.npz')
        data = np.load(path)
        return data['centers'], data['cell_ids'], data['patches']

    def __getitem__(self, idx):
        exp, t = self.samples[idx]   # t is the centre frame
        parents, children, p2c = self.mitosis[exp]

        c_prev, ids_prev, patches_prev = self._load_frame(exp, t - 1)
        c_curr, ids_curr, patches_curr = self._load_frame(exp, t)
        c_next, ids_next, patches_next = self._load_frame(exp, t + 1)

        if len(c_curr) == 0:
            return None

        # Handle genuinely empty flanking frames
        pshape = patches_curr.shape[1:]
        if len(c_prev) == 0:
            c_prev     = np.zeros((0, 3),  dtype=np.float32)
            ids_prev   = np.zeros(0,        dtype=ids_curr.dtype)
            patches_prev = np.zeros((0,) + pshape, dtype=patches_curr.dtype)
        if len(c_next) == 0:
            c_next     = np.zeros((0, 3),  dtype=np.float32)
            ids_next   = np.zeros(0,        dtype=ids_curr.dtype)
            patches_next = np.zeros((0,) + pshape, dtype=patches_curr.dtype)

        N_p, N_c, N_n = len(c_prev), len(c_curr), len(c_next)

        edge_index, edge_feat, labels, valid, sister_labels, valid_sister = \
            _build_triplet_graph(
                c_prev, ids_prev, c_curr, ids_curr, c_next, ids_next,
                t, parents, p2c,
                self.r_intra, self.r_cross, self.z_anisotropy,
            )

        # 5-D positions: [z_scaled, y, x, is_prev, is_next]
        sc_prev = _scale(c_prev, self.z_anisotropy)
        sc_curr = _scale(c_curr, self.z_anisotropy)
        sc_next = _scale(c_next, self.z_anisotropy)

        def _pos(sc, is_p, is_n):
            N = len(sc)
            if N == 0:
                return np.zeros((0, 5), dtype=np.float32)
            return np.column_stack([sc,
                                    np.full(N, is_p, dtype=np.float32),
                                    np.full(N, is_n, dtype=np.float32)])

        positions = np.vstack([
            _pos(sc_prev, 1., 0.),
            _pos(sc_curr, 0., 0.),
            _pos(sc_next, 0., 1.),
        ]).astype(np.float32)

        all_patches = np.concatenate([patches_prev, patches_curr, patches_next], axis=0)
        all_ids     = np.concatenate([ids_prev, ids_curr, ids_next])
        frame_flags = np.array([0]*N_p + [1]*N_c + [2]*N_n, dtype=np.int32)

        # FG labels (all three frames)
        fg_labels = (all_ids > 0).astype(np.float32)
        valid_fg  = np.ones(N_p + N_c + N_n, dtype=bool)

        # --- Detection dropout augmentation ---
        # Protect dividing parents and their daughters so sister-edge signal is preserved.
        if self.aug_drop_prob > 0.0:
            dividing_prev = {pid for pid, pinfo in parents.items()
                             if pinfo['LastFrame'] == t - 1}
            dividing_curr = {pid for pid, pinfo in parents.items()
                             if pinfo['LastFrame'] == t}
            daugh_curr    = {did for pid in dividing_prev for did in p2c.get(pid, [])}
            daugh_next    = {did for pid in dividing_curr for did in p2c.get(pid, [])}

            protected = np.zeros(N_p + N_c + N_n, dtype=bool)
            for i, cid in enumerate(ids_prev):
                if int(cid) in dividing_prev:
                    protected[i] = True
            for i, cid in enumerate(ids_curr):
                if int(cid) in dividing_curr or int(cid) in daugh_curr:
                    protected[N_p + i] = True
            for i, cid in enumerate(ids_next):
                if int(cid) in daugh_next:
                    protected[N_p + N_c + i] = True

            drop = ((all_ids > 0) & ~protected &
                    (np.random.rand(N_p + N_c + N_n) < self.aug_drop_prob))
            if drop.any():
                drop_edge = drop[edge_index[0]] | drop[edge_index[1]]
                valid[drop_edge]        = False
                valid_sister[drop_edge] = False
                fg_labels[drop]         = 0.0

        return dict(
            patches       = torch.from_numpy(all_patches),
            positions     = torch.from_numpy(positions),
            edge_index    = torch.from_numpy(edge_index),
            edge_feat     = torch.from_numpy(edge_feat),
            labels        = torch.from_numpy(labels),
            valid         = torch.from_numpy(valid),
            fg_labels     = torch.from_numpy(fg_labels),
            valid_fg      = torch.from_numpy(valid_fg),
            sister_labels = torch.from_numpy(sister_labels),
            valid_sister  = torch.from_numpy(valid_sister),
            frame_flags   = torch.from_numpy(frame_flags),
            N_prev        = N_p,
            N_curr        = N_c,
            N_next        = N_n,
            exp           = exp,
            t             = t,
        )


def collate_fn(batch):
    """Drop None items (empty frames)."""
    return [b for b in batch if b is not None]
