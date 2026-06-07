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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tracking.config import Config
from tracking.dataset import TrackingDataset, collate_fn
from tracking.model import SimpleTrackingNet


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

        conn_logits, sister_logits, fg_logits = model(patches, positions, edge_index, edge_feat)

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
    print(f'Train pairs: {len(train_ds)},  Val pairs: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)

    model = SimpleTrackingNet(
        feat_dim=cfg.feat_dim,
        gnn_layers=cfg.gnn_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    ckpt_dir = os.path.join(cfg.cache_dir, 'checkpoints_simple', run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, 'log.jsonl')

    best_f1 = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            for item in batch:
                patches    = item['patches'].to(device)
                positions  = item['positions'].to(device)
                edge_index = item['edge_index'].to(device)
                edge_feat  = item['edge_feat'].to(device)
                if edge_index.shape[1] == 0:
                    continue

                conn_logits, sister_logits, fg_logits = model(
                    patches, positions, edge_index, edge_feat)

                conn_loss = focal_bce(
                    conn_logits,
                    item['labels'].to(device),
                    item['valid'].to(device),
                    cfg.pos_weight,
                )
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
                loss = (conn_loss
                        + cfg.sister_loss_weight * sister_loss
                        + cfg.fg_loss_weight * fg_loss)
                if loss.requires_grad:
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optim.step()
                    losses.append(loss.item())

        sched.step()
        avg_loss = np.mean(losses) if losses else float('nan')

        if epoch % 5 == 0 or epoch == 1:
            metrics = eval_edges(model, val_ds, device, cfg)
            f1 = metrics['f1']
            print(f'Epoch {epoch:4d} | loss {avg_loss:.4f} | '
                  f'conn F1={f1:.3f} | '
                  f'sister F1={metrics["sister_f1"]:.3f} '
                  f'(P={metrics["sister_p"]:.3f} R={metrics["sister_r"]:.3f})')
            with open(log_path, 'a') as fh:
                fh.write(json.dumps(dict(epoch=epoch, loss=avg_loss, **metrics)) + '\n')
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pt'))
        else:
            print(f'Epoch {epoch:4d} | loss {avg_loss:.4f}')

    print(f'\nBest val F1: {best_f1:.4f}')
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last.pt'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='default',
                        help='subdirectory under checkpoints_simple/ for this run')
    args = parser.parse_args()
    cfg = Config()
    train(cfg, run_name=args.run_name)
