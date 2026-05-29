"""
Training script for TrackingNet.

Usage:
    python -m tracking.train
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tracking.config import Config
from tracking.dataset import TrackingDataset, collate_fn
from tracking.model import TrackingNet


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_bce(logits, labels, valid, pos_weight, gamma=2.0):
    """Focal BCE loss on valid subset. pos_weight upweights the positive class."""
    if valid.sum() == 0:
        return logits.sum() * 0.0

    l = logits[valid]
    y = labels[valid]

    bce = nn.functional.binary_cross_entropy_with_logits(
        l, y,
        pos_weight=torch.tensor(pos_weight, device=l.device),
        reduction='none',
    )
    p     = torch.sigmoid(l)
    pt    = torch.where(y == 1, p, 1 - p)
    focal = (1 - pt) ** gamma
    return (focal * bce).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_edges(model, dataset, device, cfg):
    """Compute edge P/R/F1 and mitosis P/R/F1 on a dataset."""
    model.eval()
    tp = fp = fn = 0
    mit_tp = mit_fp = mit_fn = 0
    for item in dataset:
        if item is None:
            continue
        out = forward_item(model, item, device)
        if out is None:
            continue
        edge_logits, mit_logits = out

        valid = item['valid'].to(device)
        if valid.sum() > 0:
            pred = (torch.sigmoid(edge_logits[valid]) >= cfg.cross_threshold).float()
            true = item['labels'].to(device)[valid]
            tp += ((pred == 1) & (true == 1)).sum().item()
            fp += ((pred == 1) & (true == 0)).sum().item()
            fn += ((pred == 0) & (true == 1)).sum().item()

        vm = item['valid_mitosis'].to(device)
        if vm.sum() > 0:
            mp   = (torch.sigmoid(mit_logits[vm]) >= cfg.mitosis_head_threshold).float()
            mt   = item['mitosis_labels'].to(device)[vm]
            mit_tp += ((mp == 1) & (mt == 1)).sum().item()
            mit_fp += ((mp == 1) & (mt == 0)).sum().item()
            mit_fn += ((mp == 0) & (mt == 1)).sum().item()

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    mp   = mit_tp / (mit_tp + mit_fp + 1e-8)
    mr   = mit_tp / (mit_tp + mit_fn + 1e-8)
    mf1  = 2 * mp * mr / (mp + mr + 1e-8)
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn,
                mit_p=mp, mit_r=mr, mit_f1=mf1)


def forward_item(model, item, device):
    """Run one forward pass; returns (edge_logits, mitosis_logits) or None."""
    patches    = item['patches'].to(device)
    positions  = item['positions'].to(device)
    edge_index = item['edge_index'].to(device)
    edge_feat  = item['edge_feat'].to(device)

    if edge_index.shape[1] == 0:
        return None
    return model(patches, positions, edge_index, edge_feat)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- data ----
    print('Loading datasets...')
    train_ds = TrackingDataset(
        cfg.train_exps, cfg.data_root, cfg.cache_dir,
        cfg.r_intra, cfg.r_cross, cfg.z_anisotropy,
    )
    val_ds = TrackingDataset(
        cfg.val_exps, cfg.data_root, cfg.cache_dir,
        cfg.r_intra, cfg.r_cross, cfg.z_anisotropy,
    )
    print(f'  Train pairs: {len(train_ds)},  Val pairs: {len(val_ds)}')

    # DataLoader with shuffle; collate just drops Nones, we iterate the list
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)

    # ---- model ----
    model = TrackingNet(
        feat_dim=cfg.feat_dim,
        gnn_layers=cfg.gnn_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,}')

    optim  = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    ckpt_dir = os.path.join(cfg.cache_dir, 'checkpoints_cpu')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, 'log.jsonl')

    best_f1  = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            for item in batch:          # batch size is 1; item is a single dict
                out = forward_item(model, item, device)
                if out is None:
                    continue
                edge_logits, mit_logits = out

                edge_loss = focal_bce(
                    edge_logits,
                    item['labels'].to(device),
                    item['valid'].to(device),
                    cfg.pos_weight,
                )
                mit_loss = focal_bce(
                    mit_logits,
                    item['mitosis_labels'].to(device),
                    item['valid_mitosis'].to(device),
                    cfg.mitosis_pos_weight,
                )
                loss = edge_loss + cfg.mitosis_loss_weight * mit_loss

                if loss.requires_grad:
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optim.step()
                    losses.append(loss.item())

        sched.step()
        avg_loss = np.mean(losses) if losses else float('nan')

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = eval_edges(model, val_ds, device, cfg)
            f1 = val_metrics['f1']
            print(f'Epoch {epoch:4d} | loss {avg_loss:.4f} | '
                  f'edge P={val_metrics["precision"]:.3f} '
                  f'R={val_metrics["recall"]:.3f} '
                  f'F1={f1:.3f} | '
                  f'mit P={val_metrics["mit_p"]:.3f} '
                  f'R={val_metrics["mit_r"]:.3f} '
                  f'F1={val_metrics["mit_f1"]:.3f}')

            log_entry = dict(epoch=epoch, loss=avg_loss, **val_metrics)
            with open(log_path, 'a') as fh:
                fh.write(json.dumps(log_entry) + '\n')

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'best.pt'))
        else:
            print(f'Epoch {epoch:4d} | loss {avg_loss:.4f}')

    print(f'\nBest val F1: {best_f1:.4f}')
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last.pt'))


if __name__ == '__main__':
    cfg = Config()
    train(cfg)
