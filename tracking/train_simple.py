"""
Training script for SimpleTrackingNet.

Single objective: predict whether two detections are connected
(same cell intra-frame, same cell cross-frame, or parent→daughter).
No auxiliary heads.

Usage:
    python -m tracking.train_simple
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader

from tracking.config import Config
from tracking.dataset import TrackingDataset, collate_fn
from tracking.model import SimpleTrackingNet


def cosine_metric_loss(h_cnn, edge_index, edge_feat, labels, valid, margin=0.3):
    """
    Cosine embedding loss on cross-frame edges only.
    Same-cell pairs (label=1): push cosine sim → 1.
    Different-cell pairs (label=0): push cosine sim < margin.
    Applied to pre-GNN h_cnn so the CNN encoder learns discriminative identity features.
    """
    is_cross = edge_feat[:, 3] < 0.5          # same_frame flag == 0
    mask = valid & is_cross
    if mask.sum() == 0:
        return h_cnn.sum() * 0.0
    src, dst = edge_index[0][mask], edge_index[1][mask]
    h_norm = F.normalize(h_cnn, dim=-1)
    target = 2 * labels[mask] - 1             # label 1→+1, label 0→-1
    return F.cosine_embedding_loss(h_norm[src], h_norm[dst], target, margin=margin)


def focal_bce(logits, labels, valid, pos_weight, gamma=2.0):
    if valid.sum() == 0:
        return logits.sum() * 0.0
    l = logits[valid]
    y = labels[valid]
    bce = nn.functional.binary_cross_entropy_with_logits(
        l, y,
        pos_weight=torch.tensor(pos_weight, device=l.device),
        reduction='none',
    )
    pt    = torch.where(y == 1, torch.sigmoid(l), 1 - torch.sigmoid(l))
    focal = (1 - pt) ** gamma
    return (focal * bce).mean()


@torch.no_grad()
def eval_edges(model, dataset, device, cfg):
    model.eval()
    use_amp = device.type == 'cuda'
    tp = fp = fn = 0
    s_tp = s_fp = s_fn = 0
    for item in dataset:
        if item is None:
            continue
        patches    = item['patches'].to(device)
        positions  = item['positions'].to(device)
        edge_index = item['edge_index'].to(device)
        edge_feat  = item['edge_feat'].to(device)
        if edge_index.shape[1] == 0:
            continue

        N_p, N_c, N_n = item['N_prev'], item['N_curr'], item['N_next']
        h_raw = _encode_frames(model, patches[:, :cfg.in_channels],
                               N_p, N_c, N_n, cfg.feat_dim, device, use_amp)
        h_cnn = h_raw + model.pos_enc(positions)
        conn_logits, sister_logits, fg_logits = model.forward_from_features(h_cnn, edge_index, edge_feat)

        valid = item['valid'].to(device)
        if valid.sum() > 0:
            pred = (torch.sigmoid(conn_logits[valid]) >= cfg.cross_threshold).float()
            true = item['labels'].to(device)[valid]
            tp += ((pred == 1) & (true == 1)).sum().item()
            fp += ((pred == 1) & (true == 0)).sum().item()
            fn += ((pred == 0) & (true == 1)).sum().item()

        vs = item['valid_sister'].to(device)
        if vs.sum() > 0:
            sp = (torch.sigmoid(sister_logits[vs]) >= 0.5).float()
            st = item['sister_labels'].to(device)[vs]
            s_tp += ((sp == 1) & (st == 1)).sum().item()
            s_fp += ((sp == 1) & (st == 0)).sum().item()
            s_fn += ((sp == 0) & (st == 1)).sum().item()

    def prf(a, b, c):
        p = a / (a + b + 1e-8); r = a / (a + c + 1e-8)
        return p, r, 2*p*r/(p+r+1e-8)
    p, r, f1    = prf(tp,   fp,   fn)
    sp, sr, sf1 = prf(s_tp, s_fp, s_fn)
    return dict(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn,
                sister_p=sp, sister_r=sr, sister_f1=sf1)


def _encode_frames(model, patches, N_p, N_c, N_n, feat_dim, device, use_amp):
    """Encode CNN one frame at a time to cap the gradient-checkpoint recompute size."""
    def _enc(p):
        if p.shape[0] == 0:
            return torch.zeros(0, feat_dim, device=device)
        with amp.autocast('cuda', enabled=use_amp):
            return model.encode_cnn(p)
    h = torch.cat([_enc(patches[:N_p]),
                   _enc(patches[N_p:N_p + N_c]),
                   _enc(patches[N_p + N_c:])])
    return h.float()


def train(cfg: Config, run_name: str = 'default'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_ds = TrackingDataset(
        cfg.train_exps, cfg.data_root, cfg.cache_dir,
        cfg.r_intra, cfg.r_cross, cfg.z_anisotropy,
        aug_drop_prob=cfg.aug_drop_prob,
    )
    val_ds = TrackingDataset(
        cfg.val_exps, cfg.data_root, cfg.cache_dir,
        cfg.r_intra, cfg.r_cross, cfg.z_anisotropy,
        aug_drop_prob=0.0,
    )
    print(f'Train triplets: {len(train_ds)},  Val triplets: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    model = SimpleTrackingNet(
        feat_dim=cfg.feat_dim,
        gnn_layers=cfg.gnn_layers,
        in_channels=cfg.in_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    warmup_epochs = cfg.warmup_epochs
    if warmup_epochs > 0:
        def _lr_lambda(ep):   # ep is 0-indexed
            if ep < warmup_epochs:
                return (ep + 1) / warmup_epochs
            progress = (ep - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    ckpt_dir = os.path.join(cfg.cache_dir, 'checkpoints_simple', run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, 'log.jsonl')

    best_combined = 0.0
    use_amp = device.type == 'cuda'
    scaler = amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        if use_amp:
            torch.cuda.empty_cache()

        for batch in train_loader:
            for item in batch:
                patches    = item['patches'].to(device)
                positions  = item['positions'].to(device)
                edge_index = item['edge_index'].to(device)
                edge_feat  = item['edge_feat'].to(device)
                if edge_index.shape[1] == 0:
                    continue

                labels_d    = item['labels'].to(device)
                valid_d     = item['valid'].to(device)

                N_p, N_c, N_n = item['N_prev'], item['N_curr'], item['N_next']
                h_raw = _encode_frames(model, patches[:, :cfg.in_channels],
                                       N_p, N_c, N_n, cfg.feat_dim, device, use_amp)
                h_cnn = h_raw + model.pos_enc(positions)
                conn_logits, sister_logits, fg_logits = model.forward_from_features(
                    h_cnn, edge_index, edge_feat)

                conn_loss = focal_bce(conn_logits, labels_d, valid_d, cfg.pos_weight)
                sister_loss = focal_bce(
                    sister_logits,
                    item['sister_labels'].to(device),
                    item['valid_sister'].to(device),
                    cfg.sister_pos_weight,
                )
                fg_loss = focal_bce(
                    fg_logits,
                    item['fg_labels'].to(device),
                    item['valid_fg'].to(device),
                    cfg.fg_pos_weight,
                )
                m_loss = cosine_metric_loss(h_cnn, edge_index, edge_feat, labels_d, valid_d)
                loss = (conn_loss
                        + cfg.sister_loss_weight * sister_loss
                        + cfg.fg_loss_weight * fg_loss
                        + cfg.metric_loss_weight * m_loss)
                if loss.requires_grad:
                    optim.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                    losses.append(loss.item())

        sched.step()
        avg_loss = np.mean(losses) if losses else float('nan')

        if epoch % 5 == 0 or epoch == 1:
            metrics = eval_edges(model, val_ds, device, cfg)
            combined = metrics['f1'] + metrics['sister_f1']
            print(f'Epoch {epoch:4d} | loss {avg_loss:.4f} | '
                  f'conn F1={metrics["f1"]:.3f} | '
                  f'sister F1={metrics["sister_f1"]:.3f} '
                  f'(P={metrics["sister_p"]:.3f} R={metrics["sister_r"]:.3f}) | '
                  f'combined={combined:.3f}')
            with open(log_path, 'a') as fh:
                fh.write(json.dumps(dict(epoch=epoch, loss=avg_loss, **metrics)) + '\n')
            if combined > best_combined:
                best_combined = combined
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))
        else:
            print(f'Epoch {epoch:4d} | loss {avg_loss:.4f}')

    print(f'\nBest combined (conn+sister) F1: {best_combined:.4f}')
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last.pt'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='default')
    parser.add_argument('--aug-drop-prob',      type=float, default=None,
                        help='override cfg.aug_drop_prob')
    parser.add_argument('--sister-loss-weight', type=float, default=None,
                        help='override cfg.sister_loss_weight')
    parser.add_argument('--warmup-epochs',      type=int,   default=None,
                        help='override cfg.warmup_epochs (0 = no warmup)')
    parser.add_argument('--in-channels',        type=int,   default=None,
                        help='override cfg.in_channels (1 or 2)')
    args = parser.parse_args()

    cfg = Config()
    if args.aug_drop_prob      is not None: cfg.aug_drop_prob      = args.aug_drop_prob
    if args.sister_loss_weight is not None: cfg.sister_loss_weight = args.sister_loss_weight
    if args.warmup_epochs      is not None: cfg.warmup_epochs      = args.warmup_epochs
    if args.in_channels        is not None: cfg.in_channels        = args.in_channels

    print(f'aug_drop_prob={cfg.aug_drop_prob}  sister_loss_weight={cfg.sister_loss_weight}  '
          f'warmup_epochs={cfg.warmup_epochs}  in_channels={cfg.in_channels}')
    train(cfg, run_name=args.run_name)
