"""
TrackingDataset: loads preprocessed frame pairs and builds detection graphs.

Each sample is a frame pair (t, t+1) from one experiment.
Nodes  = all detections in both frames.
Edges  = pairs within spatial radius (intra-frame or cross-frame).
Labels = 1 if same cell, 0 otherwise.

The mitosis logic is encoded purely in labels:
  - parent(t) → daughter(t+1) : label 1  (same as "same cell")
  - daughter_A(t+1) ↔ daughter_B(t+1) : label 0  (different cells)
At inference: a parent with two label-1 cross-frame edges to daughters
that are label-0 to each other ⟹ division event.
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
    # parent_id -> list of child_ids
    p2c = {}
    for cid, info in children.items():
        pid = info['ParentID']
        p2c.setdefault(pid, []).append(cid)
    return parents, children, p2c


def _scale(centers, z_anisotropy):
    s = centers.copy()
    s[:, 0] *= z_anisotropy
    return s


def _build_graph(centers0, ids0, centers1, ids1, t, parents, p2c,
                 r_intra, r_cross, z_anisotropy):
    """
    Build all graph edges and labels for a frame pair (t, t+1).

    Returns:
        edge_index : (2, E) int64
        edge_feat  : (E, 4) float32  [dz, dy, dx, same_frame]
        labels     : (E,)  float32   1=same cell, 0=different
        valid_mask : (E,)  bool      True when both nodes have known GT
    """
    N0, N1 = len(centers0), len(centers1)
    N = N0 + N1

    all_c  = np.vstack([centers0, centers1]).astype(np.float32)   # (N,3)
    all_id = np.concatenate([ids0, ids1])                          # (N,)
    frame  = np.array([0]*N0 + [1]*N1, dtype=np.int32)

    sc = _scale(all_c, z_anisotropy)                               # isotropic coords

    # Positive parent→daughter pairs at this transition (parent LastFrame == t)
    pos_cross = set()
    for pid, pinfo in parents.items():
        if pinfo['LastFrame'] == t:
            for did in p2c.get(pid, []):
                pos_cross.add((pid, did))  # (parent_cell_id, daughter_cell_id)

    # --- vectorised distance matrix ---
    diff  = sc[:, None, :] - sc[None, :, :]          # (N, N, 3)
    dists = np.linalg.norm(diff, axis=-1)             # (N, N)

    same_fr = (frame[:, None] == frame[None, :])      # (N, N) bool

    in_range = (
        (same_fr  & (dists <= r_intra)) |
        (~same_fr & (dists <= r_cross))
    )
    np.fill_diagonal(in_range, False)

    src_arr, dst_arr = np.where(in_range)             # (E,) each direction

    if len(src_arr) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 4), dtype=np.float32),
                empty, empty.astype(bool))

    # Edge features: [dz, dy, dx, same_frame]
    d = all_c[src_arr] - all_c[dst_arr]               # (E, 3) raw delta ZYX
    d[:, 0] *= z_anisotropy
    sf = (frame[src_arr] == frame[dst_arr]).astype(np.float32)
    edge_feat = np.column_stack([d, sf]).astype(np.float32)

    # --- labels ---
    id_s = all_id[src_arr]
    id_d = all_id[dst_arr]
    same_frame_edge = sf.astype(bool)

    labels     = np.zeros(len(src_arr), dtype=np.float32)
    valid_mask = np.zeros(len(src_arr), dtype=bool)

    for e in range(len(src_arr)):
        is_  = id_s[e]
        id_  = id_d[e]
        if is_ == 0 or id_ == 0:
            valid_mask[e] = False
            continue
        valid_mask[e] = True

        if same_frame_edge[e]:
            labels[e] = float(is_ == id_)
        else:
            # Orient so src=frame-t, dst=frame-(t+1)
            fi = int(frame[src_arr[e]])
            if fi == 0:
                pid_cell, did_cell = is_, id_
            else:
                pid_cell, did_cell = id_, is_
            labels[e] = float(
                (pid_cell == did_cell) or
                ((pid_cell, did_cell) in pos_cross)
            )

    edge_index = np.stack([src_arr, dst_arr], axis=0).astype(np.int64)
    return edge_index, edge_feat, labels, valid_mask


class TrackingDataset(Dataset):
    """
    Each item is one frame pair (t, t+1) from one experiment.
    Skips pairs where either frame has zero detections.
    """

    MITOSIS_PATHS = {
        '0501': 'mitotic_events/mitosis_info_0501.json',
        '0507': 'mitotic_events/mitosis_info_0507.json',
        '0515': 'mitotic_events/mitosis_info_0515.json',
    }

    def __init__(self, exp_ids, data_root, cache_dir,
                 r_intra=20., r_cross=30., z_anisotropy=0.5):
        self.cache_dir     = cache_dir
        self.data_root     = data_root
        self.r_intra       = r_intra
        self.r_cross       = r_cross
        self.z_anisotropy  = z_anisotropy

        self.samples = []    # (exp_id, t)
        self.mitosis = {}    # exp_id -> (parents, children, p2c)

        for exp in exp_ids:
            mit_path = os.path.join(data_root, self.MITOSIS_PATHS[exp])
            self.mitosis[exp] = _load_mitosis(mit_path)

            exp_cache = os.path.join(cache_dir, exp)
            frames = sorted(f for f in os.listdir(exp_cache) if f.startswith('frame_'))
            n = len(frames)
            for t in range(n - 1):
                self.samples.append((exp, t))

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, exp, t):
        """Load all detections (FG + BG) for a frame."""
        path = os.path.join(self.cache_dir, exp, f'frame_{t:04d}.npz')
        data = np.load(path)
        return data['centers'], data['cell_ids'], data['patches']

    def __getitem__(self, idx):
        exp, t = self.samples[idx]
        parents, children, p2c = self.mitosis[exp]

        c0, ids0, patches0 = self._load_frame(exp, t)
        c1, ids1, patches1 = self._load_frame(exp, t + 1)

        if len(c0) == 0 or len(c1) == 0:
            return None

        edge_index, edge_feat, labels, valid = _build_graph(
            c0, ids0, c1, ids1, t, parents, p2c,
            self.r_intra, self.r_cross, self.z_anisotropy
        )

        N0, N1 = len(c0), len(c1)
        all_c  = np.vstack([c0, c1]).astype(np.float32)
        sc     = _scale(all_c, self.z_anisotropy)
        frame_flags = np.array([0]*N0 + [1]*N1, dtype=np.float32)
        positions = np.column_stack([sc, frame_flags]).astype(np.float32)

        all_patches = np.concatenate([patches0, patches1], axis=0)

        # --- Mitosis node labels ---
        # A node is positive (label=1) if its cell_id is a parent whose
        # LastFrame == t (i.e., it divides at this frame transition).
        # Only frame-t nodes (indices 0..N0-1) can be dividing parents.
        dividing_ids = {
            pid for pid, pinfo in parents.items()
            if pinfo['LastFrame'] == t
        }
        mitosis_labels = np.zeros(N0 + N1, dtype=np.float32)
        for i, cid in enumerate(ids0):
            if int(cid) in dividing_ids:
                mitosis_labels[i] = 1.0
        # Frame-(t+1) nodes are never "parent at t", so their labels stay 0.
        # valid_mitosis: only GT-labeled frame-t nodes contribute to the loss
        valid_mitosis = np.zeros(N0 + N1, dtype=bool)
        valid_mitosis[:N0] = (ids0 > 0)

        # --- Foreground node labels ---
        all_ids = np.concatenate([ids0, ids1])
        fg_labels = (all_ids > 0).astype(np.float32)
        valid_fg  = np.ones(N0 + N1, dtype=bool)

        # --- Daughter node labels ---
        # A frame-(t+1) node is a daughter if its cell_id appears in the
        # children list of a parent whose LastFrame == t.
        daughter_ids = set()
        for pid, pinfo in parents.items():
            if pinfo['LastFrame'] == t:
                for did in p2c.get(pid, []):
                    daughter_ids.add(did)
        daughter_labels = np.zeros(N0 + N1, dtype=np.float32)
        for i, cid in enumerate(ids1):
            if int(cid) in daughter_ids:
                daughter_labels[N0 + i] = 1.0
        # Frame-t nodes are never daughters; only GT frame-(t+1) nodes contribute.
        valid_daughter = np.zeros(N0 + N1, dtype=bool)
        valid_daughter[N0:] = (ids1 > 0)

        return dict(
            patches          = torch.from_numpy(all_patches),
            positions        = torch.from_numpy(positions),
            edge_index       = torch.from_numpy(edge_index),
            edge_feat        = torch.from_numpy(edge_feat),
            labels           = torch.from_numpy(labels),
            valid            = torch.from_numpy(valid),
            mitosis_labels   = torch.from_numpy(mitosis_labels),
            valid_mitosis    = torch.from_numpy(valid_mitosis),
            fg_labels        = torch.from_numpy(fg_labels),
            valid_fg         = torch.from_numpy(valid_fg),
            daughter_labels  = torch.from_numpy(daughter_labels),
            valid_daughter   = torch.from_numpy(valid_daughter),
            frame_flags      = torch.from_numpy(frame_flags).long(),
            N0               = N0,
            exp              = exp,
            t                = t,
        )


def collate_fn(batch):
    """Drop None items (empty frames)."""
    batch = [b for b in batch if b is not None]
    return batch  # list of dicts; each processed individually in training loop
