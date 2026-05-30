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
    """Compute edge/mitosis/daughter/fg P/R/F1 on a dataset."""
    model.eval()
    tp = fp = fn = 0
    mit_tp = mit_fp = mit_fn = 0
    dau_tp = dau_fp = dau_fn = 0
    fg_tp  = fg_fp  = fg_fn  = 0
    for item in dataset:
        if item is None:
            continue
        out = forward_item(model, item, device)
        if out is None:
            continue
        edge_logits, mit_logits, fg_logits, daughter_logits = out

        valid = item['valid'].to(device)
        if valid.sum() > 0:
            pred = (torch.sigmoid(edge_logits[valid]) >= cfg.cross_threshold).float()
            true = item['labels'].to(device)[valid]
            tp += ((pred == 1) & (true == 1)).sum().item()
            fp += ((pred == 1) & (true == 0)).sum().item()
            fn += ((pred == 0) & (true == 1)).sum().item()

        vm = item['valid_mitosis'].to(device)
        if vm.sum() > 0:
            mp = (torch.sigmoid(mit_logits[vm]) >= cfg.mitosis_head_threshold).float()
            mt = item['mitosis_labels'].to(device)[vm]
            mit_tp += ((mp == 1) & (mt == 1)).sum().item()
            mit_fp += ((mp == 1) & (mt == 0)).sum().item()
            mit_fn += ((mp == 0) & (mt == 1)).sum().item()

        vd = item['valid_daughter'].to(device)
        if vd.sum() > 0:
            dp = (torch.sigmoid(daughter_logits[vd]) >= cfg.daughter_head_threshold).float()
            dt = item['daughter_labels'].to(device)[vd]
            dau_tp += ((dp == 1) & (dt == 1)).sum().item()
            dau_fp += ((dp == 1) & (dt == 0)).sum().item()
            dau_fn += ((dp == 0) & (dt == 1)).sum().item()

        vf = item['valid_fg'].to(device)
        if vf.sum() > 0:
            fp2 = (torch.sigmoid(fg_logits[vf]) >= cfg.fg_threshold).float()
            ft  = item['fg_labels'].to(device)[vf]
            fg_tp += ((fp2 == 1) & (ft == 1)).sum().item()
            fg_fp += ((fp2 == 1) & (ft == 0)).sum().item()
            fg_fn += ((fp2 == 0) & (ft == 1)).sum().item()

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_ + 1e-8)
        r = tp_ / (tp_ + fn_ + 1e-8)
        return p, r, 2 * p * r / (p + r + 1e-8)

    prec, rec, f1  = prf(tp, fp, fn)
    mp,   mr,  mf1 = prf(mit_tp, mit_fp, mit_fn)
    dp,   dr,  df1 = prf(dau_tp, dau_fp, dau_fn)
    fp_,  fr,  ff1 = prf(fg_tp,  fg_fp,  fg_fn)
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn,
                mit_p=mp,  mit_r=mr,  mit_f1=mf1,
                dau_p=dp,  dau_r=dr,  dau_f1=df1,
                fg_p=fp_,  fg_r=fr,   fg_f1=ff1)


def forward_item(model, item, device):
    """Run one forward pass; returns (edge_logits, mitosis_logits, fg_logits) or None."""
    patches    = item['patches'].to(device)
    positions  = item['positions'].to(device)
    edge_index = item['edge_index'].to(device)
    edge_feat  = item['edge_feat'].to(device)
    fg_gt      = item['fg_labels'].to(device)   # GT mask for GNN edge filtering

    if edge_index.shape[1] == 0:
        return None
    return model(patches, positions, edge_index, edge_feat, fg_gt=fg_gt)


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

    ckpt_dir = os.path.join(cfg.cache_dir, 'checkpoints')
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
                edge_logits, mit_logits, fg_logits, daughter_logits = out

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
                daughter_loss = focal_bce(
                    daughter_logits,
                    item['daughter_labels'].to(device),
                    item['valid_daughter'].to(device),
                    cfg.daughter_pos_weight,
                )
                fg_loss = focal_bce(
                    fg_logits,
                    item['fg_labels'].to(device),
                    item['valid_fg'].to(device),
                    cfg.fg_pos_weight,
                )
                loss = (edge_loss
                        + cfg.mitosis_loss_weight  * mit_loss
                        + cfg.daughter_loss_weight * daughter_loss
                        + cfg.fg_loss_weight       * fg_loss)

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
                  f'edge F1={f1:.3f} | '
                  f'mit F1={val_metrics["mit_f1"]:.3f} | '
                  f'dau F1={val_metrics["dau_f1"]:.3f} | '
                  f'fg F1={val_metrics["fg_f1"]:.3f}')

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
