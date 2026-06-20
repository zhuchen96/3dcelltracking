"""
SimpleTrackingNet: 3D-patch CNN encoder + score-gated MPNN + edge classifiers.

Flow:
  patches (N, 1, pz, py, px)  →  Encoder3D  →  node feats h_cnn (N, D)
  h_cnn + positions            →  MPNN×L     →  refined feats h (N, D)
  node pairs + edge feats      →  EdgeMLP    →  conn_logits / sister_logits (E,)
  h_cnn                        →  NodeMLP    →  fg_logits (N,)

For phantom-node inference, call encode() → override phantom rows → forward_from_features().
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
    """Input: (B, in_channels, pz, py, px)  →  Output: (B, feat_dim)
    Channel 0: raw intensity.  Channel 1 (optional): exp(-det/5) detection probability."""

    def __init__(self, feat_dim=128, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1, bias=False),
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
    h_j'         = LayerNorm(h_j + FFN(sum_i message(i→j) ‖ log(count+1)))

    Sum aggregation preserves the number of incoming messages, which is
    essential for division detection: a dividing cell has 2 strong connections
    while a normal cell has 1.  log-count makes this explicit without dominating.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, edge_index, edge_feat):
        src, dst = edge_index[0], edge_index[1]
        N = h.shape[0]

        msgs = self.msg(torch.cat([h[src], h[dst], edge_feat], dim=-1))

        agg   = torch.zeros_like(h)
        count = torch.zeros(N, 1, device=h.device)
        idx   = dst.unsqueeze(1).expand_as(msgs)
        agg.scatter_add_(0, idx, msgs)
        count.scatter_add_(0, dst.unsqueeze(1),
                           torch.ones(len(dst), 1, device=h.device))

        log_count = torch.log(count + 1)
        return self.norm(h + self.update(torch.cat([h, agg, log_count], dim=-1)))


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
        return self.mlp(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Node-level foreground head
# ---------------------------------------------------------------------------

class ForegroundHead(nn.Module):
    """P(detection is a real foreground cell). Applied to pre-GNN CNN features."""

    def __init__(self, node_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2), nn.GELU(),
            nn.Linear(node_dim // 2, node_dim // 4), nn.GELU(),
            nn.Linear(node_dim // 4, 1),
        )

    def forward(self, h):
        return self.mlp(h).squeeze(-1)


# ---------------------------------------------------------------------------
# SimpleTrackingNet
# ---------------------------------------------------------------------------

class SimpleTrackingNet(nn.Module):
    """
    Simplified tracking network with score-gated message passing.

    Connection edges (intra-frame same-cell + cross-frame same-cell + parent→daughter)
    share a single binary classifier.  Sister edges (two daughters of the same
    division at frame t+1) have a separate head used for division detection.

    At inference, phantom nodes (projected positions with cached CNN features) can
    be injected: call encode() → override phantom rows in h_cnn → forward_from_features().
    """

    def __init__(self, feat_dim=128, gnn_layers=2, edge_dim=4, pos_dim=4, in_channels=1):
        super().__init__()
        self.encoder     = Encoder3D(feat_dim, in_channels=in_channels)
        self.pos_enc     = nn.Linear(pos_dim, feat_dim)
        self.fg_head     = ForegroundHead(feat_dim)
        self.gnn         = nn.ModuleList(
            [MPNNLayer(feat_dim, edge_dim) for _ in range(gnn_layers)]
        )
        self.classifier  = EdgeClassifier(feat_dim, edge_dim)
        self.sister_head = EdgeClassifier(feat_dim, edge_dim)

    def encode(self, patches, positions):
        """CNN encoder + positional encoding → pre-GNN node features (N, D)."""
        return self.encoder(patches) + self.pos_enc(positions)

    def forward_from_features(self, h_cnn, edge_index, edge_feat):
        """
        GNN + classifiers from pre-computed features.

        Used at inference when phantom nodes have their h_cnn rows overridden
        with cached features from the last real detection frame.
        """
        fg_logits = self.fg_head(h_cnn)
        h = h_cnn
        for i, layer in enumerate(self.gnn):
            if i == 0 or edge_index.shape[1] == 0:
                h = layer(h, edge_index, edge_feat)
            else:
                with torch.no_grad():
                    prelim = torch.sigmoid(self.classifier(h, edge_index, edge_feat))
                mask = prelim > 0.3
                h = layer(h, edge_index[:, mask], edge_feat[mask])
        connected_logits = self.classifier(h, edge_index, edge_feat)
        sister_logits    = self.sister_head(h, edge_index, edge_feat)
        return connected_logits, sister_logits, fg_logits

    def forward(self, patches, positions, edge_index, edge_feat):
        """Standard forward pass; returns (conn_logits, sister_logits, fg_logits)."""
        h_cnn = self.encode(patches, positions)
        return self.forward_from_features(h_cnn, edge_index, edge_feat)
