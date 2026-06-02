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
    h_j'         = LayerNorm(h_j + FFN(sum_i message(i→j) ‖ log(count+1)))

    Sum aggregation (not mean) preserves the number of incoming messages,
    which is essential for detecting divisions: a dividing cell has 2 strong
    incoming/outgoing connections while a normal cell has 1.  The log-count
    makes this explicit to the update MLP without dominating the signal.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        # +1 for log-count feature
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, edge_index, edge_feat):
        src, dst = edge_index[0], edge_index[1]
        N = h.shape[0]

        msgs = self.msg(torch.cat([h[src], h[dst], edge_feat], dim=-1))  # (E, D)

        # Sum-aggregate (preserves count signal)
        agg   = torch.zeros_like(h)
        count = torch.zeros(N, 1, device=h.device)
        idx   = dst.unsqueeze(1).expand_as(msgs)
        agg.scatter_add_(0, idx, msgs)
        count.scatter_add_(0, dst.unsqueeze(1),
                           torch.ones(len(dst), 1, device=h.device))

        # log(count+1): encodes "how many neighbours contributed"
        # 0 neighbours→0, 1→0.69, 2→1.10, 3→1.39 — clearly separable
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


class MitosisHead(nn.Module):
    """
    Dividing-parent prediction.
    Input: GNN features + top-2 cross-frame outgoing edge scores (detached).
    The top-2 scores directly encode the 1→2 topology signature of division.
    """
    def __init__(self, node_dim):
        super().__init__()
        in_dim = node_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, node_dim // 2), nn.GELU(),
            nn.Linear(node_dim // 2, node_dim // 4), nn.GELU(),
            nn.Linear(node_dim // 4, 1),
        )

    def forward(self, h, top2_cross):
        return self.mlp(torch.cat([h, top2_cross], dim=-1)).squeeze(-1)


class DaughterHead(nn.Module):
    """
    Division-daughter prediction.
    Input: GNN features + max cross-frame incoming edge score (detached).
    The incoming score tells each node how strongly a frame-t mother claims it.
    """
    def __init__(self, node_dim):
        super().__init__()
        in_dim = node_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, node_dim // 2), nn.GELU(),
            nn.Linear(node_dim // 2, node_dim // 4), nn.GELU(),
            nn.Linear(node_dim // 4, 1),
        )

    def forward(self, h, max_in_cross):
        return self.mlp(torch.cat([h, max_in_cross.unsqueeze(-1)], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SimpleTrackingNet(nn.Module):
    """
    Simplified tracking network: single edge classifier, no auxiliary heads.

    All connections — intra-frame same-cell, cross-frame same-cell, and
    cross-frame parent→daughter — share a single binary label (connected=1).

    The GNN is critical for parent-daughter detection: pairwise appearance
    between parent and daughter can be low, but GNN message passing gives
    the mother node a distinctive representation when it receives messages
    from two daughter-like neighbors simultaneously.  The edge classifier
    then scores mother→daughter edges high based on this enriched context.

    No FG filtering is applied to GNN edges; the model must learn from
    context that background detections have uniformly low edge scores.
    """

    def __init__(self, feat_dim=128, gnn_layers=2, edge_dim=4, pos_dim=4):
        super().__init__()
        self.encoder     = Encoder3D(feat_dim)
        self.pos_enc     = nn.Linear(pos_dim, feat_dim)
        self.fg_head     = ForegroundHead(feat_dim)   # applied to pre-GNN CNN features
        self.gnn         = nn.ModuleList(
            [MPNNLayer(feat_dim, edge_dim) for _ in range(gnn_layers)]
        )
        self.classifier  = EdgeClassifier(feat_dim, edge_dim)
        # Sister head: predicts whether two frame-(t+1) cells are daughters
        # of the same division.  Sisters look alike and are close — a strong,
        # specific signal that replaces the mother head for division detection.
        self.sister_head = EdgeClassifier(feat_dim, edge_dim)

    def forward(self, patches, positions, edge_index, edge_feat):
        """
        Returns (conn_logits, sister_logits, fg_logits).

        fg_logits  : (N,) pre-GNN foreground score per detection.
                     Used at inference to filter background nodes before clustering.
        Score-gated message passing: after the first GNN layer, only edges that
        look positive (score >= 0.3) carry messages in later layers.
        """
        h_cnn = self.encoder(patches) + self.pos_enc(positions)
        fg_logits = self.fg_head(h_cnn)   # (N,) — pure appearance, no graph context

        h = h_cnn
        for i, layer in enumerate(self.gnn):
            if i == 0 or edge_index.shape[1] == 0:
                h = layer(h, edge_index, edge_feat)
            else:
                # Gate: only aggregate along edges that scored > 0.3 so far
                with torch.no_grad():
                    prelim = torch.sigmoid(
                        self.classifier(h, edge_index, edge_feat)
                    )
                mask = prelim > 0.3
                h = layer(h, edge_index[:, mask], edge_feat[mask])

        connected_logits = self.classifier(h, edge_index, edge_feat)
        sister_logits    = self.sister_head(h, edge_index, edge_feat)
        return connected_logits, sister_logits, fg_logits


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

    def _top2_cross_scores(self, edge_logits, edge_index, edge_feat, N, scores=None):
        """
        (N, 2) top-2 cross-frame outgoing edge scores per node.
        scores: pre-computed (E,) float tensor; if None, sigmoid(edge_logits) is used.
        Encodes the 1→2 topology signature: a dividing parent has two high scores,
        a normal cell has one.
        """
        result = torch.zeros(N, 2, device=edge_logits.device)
        if edge_index.shape[1] == 0:
            return result
        cross_mask = edge_feat[:, 3] < 0.5  # same_frame == 0
        if cross_mask.sum() == 0:
            return result
        src = edge_index[0][cross_mask]
        s   = (scores[cross_mask] if scores is not None
               else torch.sigmoid(edge_logits[cross_mask])).detach()
        order  = src.argsort()
        src_s, scores_s = src[order], s[order]
        unique_ns, counts = src_s.unique_consecutive(return_counts=True)
        offset = 0
        for n, cnt in zip(unique_ns.tolist(), counts.tolist()):
            grp = scores_s[offset:offset + cnt]
            k   = min(2, cnt)
            result[n, :k] = grp.topk(k).values
            offset += cnt
        return result

    def _max_incoming_cross_scores(self, edge_logits, edge_index, edge_feat, N, scores=None):
        """
        (N,) max cross-frame incoming edge score per node.
        scores: pre-computed (E,) float tensor; if None, sigmoid(edge_logits) is used.
        Tells each frame-t+1 node how strongly the best frame-t candidate claims it.
        """
        result = torch.zeros(N, device=edge_logits.device)
        if edge_index.shape[1] == 0:
            return result
        cross_mask = edge_feat[:, 3] < 0.5
        if cross_mask.sum() == 0:
            return result
        dst = edge_index[1][cross_mask]
        s   = (scores[cross_mask] if scores is not None
               else torch.sigmoid(edge_logits[cross_mask])).detach()
        order  = dst.argsort()
        dst_s, scores_s = dst[order], s[order]
        unique_ns, counts = dst_s.unique_consecutive(return_counts=True)
        offset = 0
        for n, cnt in zip(unique_ns.tolist(), counts.tolist()):
            result[n] = scores_s[offset:offset + cnt].max()
            offset += cnt
        return result

    def forward(self, patches, positions, edge_index, edge_feat,
                fg_gt=None, gt_labels=None, gt_valid=None):
        """
        Full forward pass (training + inference).

        Args:
            fg_gt     : (N,) float32 GT FG labels. When provided (training), GNN sees
                        only FG-FG edges. When None (inference), predicted FG is used.
            gt_labels : (E,) float32 GT edge labels. When provided, topology features
                        (top2_cross, max_in_cross) are computed from GT instead of
                        predicted scores, breaking the circular dependency that causes
                        one daughter to always score near zero.
            gt_valid  : (E,) bool mask — only GT-valid edges contribute to topology.

        Returns:
            edge_logits      : (E,)  same-cell probability logits
            mitosis_logits   : (N,)  per-node dividing-parent logits
            fg_logits        : (N,)  per-node foreground logits (pre-GNN)
            daughter_logits  : (N,)  per-node division-daughter logits
        """
        h_cnn = self._encode(patches, positions)
        fg_logits = self.foreground_head(h_cnn)

        if fg_gt is not None:
            fg_mask = fg_gt.bool()
        else:
            fg_mask = torch.sigmoid(fg_logits) >= 0.5

        h = h_cnn
        N = h.shape[0]
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            fg_e  = fg_mask[src] & fg_mask[dst]
            fg_ei = edge_index[:, fg_e]
            fg_ef = edge_feat[fg_e]
            for layer in self.gnn:
                h = layer(h, fg_ei, fg_ef)

        edge_logits = self.classifier(h, edge_index, edge_feat)

        # Topology features: use GT labels during training to break the circular
        # dependency where the edge classifier's preference for one daughter biases
        # the topology input and causes the other daughter to score near zero.
        if gt_labels is not None:
            gt_scores = torch.zeros(edge_logits.shape[0], device=edge_logits.device)
            if gt_valid is not None:
                gt_scores[gt_valid.bool()] = gt_labels[gt_valid.bool()].float()
            else:
                gt_scores = gt_labels.float()
            top2_cross   = self._top2_cross_scores(edge_logits, edge_index, edge_feat, N, scores=gt_scores)
            max_in_cross = self._max_incoming_cross_scores(edge_logits, edge_index, edge_feat, N, scores=gt_scores)
        else:
            top2_cross   = self._top2_cross_scores(edge_logits, edge_index, edge_feat, N)
            max_in_cross = self._max_incoming_cross_scores(edge_logits, edge_index, edge_feat, N)

        mitosis_logits  = self.mitosis_head(h, top2_cross)
        daughter_logits = self.daughter_head(h, max_in_cross)
        return edge_logits, mitosis_logits, fg_logits, daughter_logits
