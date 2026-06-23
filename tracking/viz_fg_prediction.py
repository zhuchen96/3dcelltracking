"""
Visualise foreground/background predictions overlaid on a raw MIP.

Each detected cell centre is drawn as a dot coloured by model prediction vs GT:
  Green  (TP): predicted FG=1, GT cell (cell_id > 0)          → correctly kept
  Gray   (TN): predicted FG=0, no GT cell (cell_id == 0)       → correctly removed
  Red    (FN): predicted FG=0, GT cell                          → wrongly removed  ← highlight
  Orange (FP): predicted FG=1, no GT cell                       → wrongly kept     ← highlight

Usage:
    python -m tracking.viz_fg_prediction \
        --exp   0004 \
        --ckpt  cache/checkpoints_simple/triplet_largepatch/best.pt \
        --frame 50 \
        --out   results/fg_viz_t050.png
"""

import os
import argparse
import numpy as np
import torch
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tracking.config import Config
from tracking.model import SimpleTrackingNet
from tracking.preprocess import EXPERIMENT_DIRS, sorted_tifs
from tracking.dataset import _build_triplet_graph, _scale


@torch.no_grad()
def viz_fg(cfg, exp_id, ckpt_path, frame_idx, out_path, ds=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleTrackingNet(
        feat_dim=cfg.feat_dim, gnn_layers=cfg.gnn_layers, in_channels=cfg.in_channels
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dirs      = EXPERIMENT_DIRS[exp_id]
    cache_dir = os.path.join(cfg.cache_dir, exp_id)
    raw_dir   = os.path.join(cfg.data_root, dirs['raw'])
    raw_tifs  = sorted_tifs(raw_dir)

    t = frame_idx

    # ---- Load current + context + next frames ----
    def load(idx):
        d = np.load(os.path.join(cache_dir, f'frame_{idx:04d}.npz'))
        return d['centers'], d['patches'], d['cell_ids']

    c0, p0, ids0 = load(t)

    if t > 0:
        c_ctx, p_ctx, _ = load(t - 1)
    else:
        c_ctx = np.zeros((0, 3), dtype=np.float32)
        p_ctx = np.zeros((0,) + p0.shape[1:], dtype=p0.dtype)

    n_frames = len(raw_tifs)
    if t + 1 < n_frames:
        c1, p1, _ = load(t + 1)
    else:
        c1 = np.zeros((0, 3), dtype=np.float32)
        p1 = np.zeros((0,) + p0.shape[1:], dtype=p0.dtype)

    N_prev, N_curr, N_next = len(c_ctx), len(c0), len(c1)

    # ---- Build triplet graph (need edge_index for GNN) ----
    ids_dummy = np.zeros(max(N_prev, N_next), dtype=ids0.dtype)
    import json
    mit_path = os.path.join(cfg.data_root, dirs['mitosis'])
    with open(mit_path) as f:
        mit = json.load(f)
    parents_gt = {int(k): v for k, v in mit['Parents'].items()}
    p2c_gt = {}
    for cid, info in mit['Children'].items():
        p2c_gt.setdefault(info['ParentID'], []).append(int(cid))

    edge_index, edge_feat, _, _, _, _ = _build_triplet_graph(
        c_ctx,  ids_dummy[:N_prev],
        c0,     ids0,
        c1,     ids_dummy[:N_next],
        t, parents_gt, p2c_gt,
        cfg.r_intra, cfg.r_cross, cfg.z_anisotropy,
    )

    # ---- 5-D position encoding ----
    all_c = np.vstack([c_ctx, c0, c1]).astype(np.float32) if N_prev + N_next > 0 else c0.astype(np.float32)
    if N_prev > 0 or N_next > 0:
        all_c = np.vstack([c_ctx, c0, c1 if N_next > 0 else np.zeros((0, 3), np.float32)])
    else:
        all_c = c0.astype(np.float32)

    N_total = N_prev + N_curr + N_next
    all_c_full = np.zeros((N_total, 3), dtype=np.float32)
    all_c_full[:N_prev]              = c_ctx
    all_c_full[N_prev:N_prev+N_curr] = c0
    all_c_full[N_prev+N_curr:]       = c1 if N_next > 0 else np.zeros((0, 3), np.float32)

    sc  = _scale(all_c_full, cfg.z_anisotropy)
    pos = np.zeros((N_total, 5), dtype=np.float32)
    pos[:, :3] = sc
    pos[:N_prev, 3] = 1.0
    pos[N_prev + N_curr:, 4] = 1.0

    # ---- Encode all nodes ----
    p_ctx_sl = p_ctx[:, :cfg.in_channels] if N_prev > 0 else np.zeros((0,) + p0.shape[1:], p0.dtype)
    p1_sl    = p1[:, :cfg.in_channels]    if N_next > 0 else np.zeros((0,) + p0.shape[1:], p0.dtype)
    patches_all = np.concatenate([p_ctx_sl, p0[:, :cfg.in_channels], p1_sl], axis=0)

    h_cnn_raw = model.encode_cnn(torch.from_numpy(patches_all).to(device))
    h_cnn     = h_cnn_raw + model.pos_enc(torch.from_numpy(pos).to(device))

    if edge_index.shape[1] > 0:
        _, _, fg_logits = model.forward_from_features(
            h_cnn,
            torch.from_numpy(edge_index).to(device),
            torch.from_numpy(edge_feat).to(device),
        )
    else:
        fg_logits = model.fg_head(h_cnn)

    fg_scores = torch.sigmoid(fg_logits).cpu().numpy()

    # Curr-frame only (indices N_prev … N_prev+N_curr-1)
    fg_curr   = fg_scores[N_prev: N_prev + N_curr]
    fg_pred   = (fg_curr >= cfg.fg_threshold).astype(int)   # 1=FG, 0=BG
    gt_fg     = (ids0 > 0).astype(int)                       # 1=real cell, 0=no GT

    tp = (fg_pred == 1) & (gt_fg == 1)   # green
    tn = (fg_pred == 0) & (gt_fg == 0)   # gray
    fn = (fg_pred == 0) & (gt_fg == 1)   # red   ← wrong
    fp = (fg_pred == 1) & (gt_fg == 0)   # orange ← wrong

    print(f'Frame {t}:  TP={tp.sum()}  TN={tn.sum()}  FN={fn.sum()}  FP={fp.sum()}')

    # ---- Load raw and MIP ----
    raw  = tifffile.imread(raw_tifs[t])   # (Z, Y, X)
    mip  = raw.max(axis=0).astype(np.float32)
    lo, hi = np.percentile(mip, 1), np.percentile(mip, 99.5)
    mip8 = np.clip((mip - lo) / max(hi - lo, 1), 0, 1)

    H, W = mip8.shape
    Hd = (H // ds) & ~1
    Wd = (W // ds) & ~1

    dpi = 120
    fig, ax = plt.subplots(figsize=(Wd / dpi, Hd / dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.imshow(np.stack([mip8[::ds, ::ds]] * 3, -1), origin='upper',
              interpolation='nearest', aspect='equal')
    ax.set_xlim(0, Wd - 1); ax.set_ylim(Hd - 1, 0)

    # Plot each category
    def scatter(mask, color, zorder, size, marker='o', lw=0):
        if mask.any():
            ys = c0[mask, 1] / ds
            xs = c0[mask, 2] / ds
            ax.scatter(xs, ys, s=size, c=color, marker=marker,
                       linewidths=lw, edgecolors='white' if lw > 0 else 'none',
                       zorder=zorder, alpha=0.85)

    scatter(tn, 'gray',    2, 8)
    scatter(tp, '#00FF88', 3, 12)
    scatter(fp, 'orange',  4, 20, lw=0.8)
    scatter(fn, 'red',     5, 20, lw=0.8)

    # FG score text for errors
    for i in np.where(fn)[0]:
        ax.text(c0[i, 2] / ds + 2, c0[i, 1] / ds - 2,
                f'{fg_curr[i]:.2f}', color='red', fontsize=4, zorder=6)
    for i in np.where(fp)[0]:
        ax.text(c0[i, 2] / ds + 2, c0[i, 1] / ds - 2,
                f'{fg_curr[i]:.2f}', color='orange', fontsize=4, zorder=6)

    # Legend
    legend_elems = [
        mpatches.Patch(color='#00FF88', label=f'TP foreground  (n={tp.sum()})'),
        mpatches.Patch(color='gray',    label=f'TN background  (n={tn.sum()})'),
        mpatches.Patch(color='red',     label=f'FN missed cell (n={fn.sum()})'),
        mpatches.Patch(color='orange',  label=f'FP false det   (n={fp.sum()})'),
    ]
    ax.legend(handles=legend_elems, loc='upper left',
              fontsize=max(4, 7 // ds), framealpha=0.6, markerscale=0.8)
    ax.set_title(f'FG prediction  t={t}  threshold={cfg.fg_threshold}',
                 color='white', fontsize=8, pad=2,
                 bbox=dict(facecolor='black', alpha=0.5, pad=2))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',   default='0004')
    parser.add_argument('--ckpt',  default='cache/checkpoints_simple/triplet_largepatch/best.pt')
    parser.add_argument('--frame', type=int, default=50)
    parser.add_argument('--out',   default=None)
    parser.add_argument('--ds',    type=int, default=2, help='spatial downsample factor')
    args = parser.parse_args()

    cfg = Config()
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(cfg.data_root, args.ckpt)
    out = args.out or f'results/fg_viz_{args.exp}_t{args.frame:03d}.png'
    viz_fg(cfg, args.exp, args.ckpt, args.frame, out, ds=args.ds)
