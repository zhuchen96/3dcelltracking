"""
TrackingNet: 3D-patch CNN encoder + MPNN + edge classifier.

Flow:
  patches (N, 1, pz, py, px)  →  Encoder3D  →  node feats (N, D)
  node feats + positions       →  MPNN×L     →  refined feats (N, D)
  node pairs + edge feats      →  EdgeMLP    →  logits (E,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D CNN patch encoder
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)


class Encoder3D(nn.Module):
    """Input: (B, 1, pz, py, px)  →  Output: (B, feat_dim)"""

    def __init__(self, feat_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResBlock3D(32), nn.MaxPool3d(2))   # /2
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            ResBlock3D(64), nn.MaxPool3d(2),                            # /4
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            ResBlock3D(128),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Message-passing GNN layer (MPNN)
# ---------------------------------------------------------------------------

class MPNNLayer(nn.Module):
    """
    Single message-passing layer.
    message(i→j) = MLP([h_i ‖ h_j ‖ e_ij])
    h_j'         = LayerNorm(h_j + FFN(mean_i message(i→j)))
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, edge_index, edge_feat):
        # h:         (N, D)
        # edge_index:(2, E)
        # edge_feat: (E, F)
        src, dst = edge_index[0], edge_index[1]
        N = h.shape[0]

        # Compute messages
        msgs = self.msg(torch.cat([h[src], h[dst], edge_feat], dim=-1))  # (E, D)

        # Mean-aggregate per destination node
        agg   = torch.zeros_like(h)
        count = torch.zeros(N, 1, device=h.device)
        idx   = dst.unsqueeze(1).expand_as(msgs)
        agg.scatter_add_(0, idx, msgs)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(len(dst), 1, device=h.device))
        agg = agg / count.clamp(min=1)

        # Update
        return self.norm(h + self.update(torch.cat([h, agg], dim=-1)))


# ---------------------------------------------------------------------------
# Edge classifier
# ---------------------------------------------------------------------------

class EdgeClassifier(nn.Module):
    """[h_i ‖ h_j ‖ e_ij] → scalar logit per edge."""

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        hidden = node_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h, edge_index, edge_feat):
        src, dst = edge_index[0], edge_index[1]
        x = torch.cat([h[src], h[dst], edge_feat], dim=-1)
        return self.mlp(x).squeeze(-1)  # (E,)


# ---------------------------------------------------------------------------
# Node-level binary heads
# ---------------------------------------------------------------------------

class _NodeHead(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2), nn.GELU(),
            nn.Linear(node_dim // 2, node_dim // 4), nn.GELU(),
            nn.Linear(node_dim // 4, 1),
        )

    def forward(self, h):
        return self.mlp(h).squeeze(-1)   # (N,) logits


class ForegroundHead(_NodeHead):
    """P(detection is a real foreground cell). Runs on pre-GNN CNN features."""


class MitosisHead(_NodeHead):
    """P(cell is a dividing parent at this frame). Runs on GNN-refined features."""


class DaughterHead(_NodeHead):
    """P(cell just appeared as a division daughter). Runs on GNN-refined features."""


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TrackingNet(nn.Module):
    """
    End-to-end tracking network.

    Args:
        feat_dim  : CNN + GNN feature dimension
        gnn_layers: number of MPNN layers
        edge_dim  : dimension of edge features (4: dz, dy, dx, same_frame)
        pos_dim   : dimension of position encoding input (4: z, y, x, frame)
    """

    def __init__(self, feat_dim=128, gnn_layers=2, edge_dim=4, pos_dim=4):
        super().__init__()
        self.encoder         = Encoder3D(feat_dim)
        self.pos_enc         = nn.Linear(pos_dim, feat_dim)
        self.gnn             = nn.ModuleList(
            [MPNNLayer(feat_dim, edge_dim) for _ in range(gnn_layers)]
        )
        self.classifier      = EdgeClassifier(feat_dim, edge_dim)
        self.foreground_head = ForegroundHead(feat_dim)
        self.mitosis_head    = MitosisHead(feat_dim)
        self.daughter_head   = DaughterHead(feat_dim)

    def _encode(self, patches, positions):
        """CNN encoder + position encoding — shared first stage."""
        return self.encoder(patches) + self.pos_enc(positions)

    def forward(self, patches, positions, edge_index, edge_feat, fg_gt=None):
        """
        Full forward pass (training + inference).

        Args:
            fg_gt : (N,) float32 GT FG labels (1=real cell, 0=BG).
                    When provided (training), GNN sees only FG-FG edges so BG
                    detections don't pollute context-based heads.
                    When None (inference), predicted FG scores are used instead.

        Returns:
            edge_logits      : (E,)  same-cell probability logits
            mitosis_logits   : (N,)  per-node dividing-parent logits  (frame-t nodes)
            fg_logits        : (N,)  per-node foreground logits (pre-GNN)
            daughter_logits  : (N,)  per-node division-daughter logits (frame-t+1 nodes)
        """
        h_cnn = self._encode(patches, positions)
        fg_logits = self.foreground_head(h_cnn)   # appearance only, before GNN

        # Build FG mask: use GT labels during training, predicted scores at inference
        if fg_gt is not None:
            fg_mask = fg_gt.bool()
        else:
            fg_mask = torch.sigmoid(fg_logits) >= 0.5

        h = h_cnn
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            fg_e  = fg_mask[src] & fg_mask[dst]   # keep only FG-FG edges for GNN
            fg_ei = edge_index[:, fg_e]
            fg_ef = edge_feat[fg_e]
            for layer in self.gnn:
                h = layer(h, fg_ei, fg_ef)

        edge_logits      = self.classifier(h, edge_index, edge_feat)
        mitosis_logits   = self.mitosis_head(h)
        daughter_logits  = self.daughter_head(h)
        return edge_logits, mitosis_logits, fg_logits, daughter_logits
