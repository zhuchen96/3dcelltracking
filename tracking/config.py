from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # Paths
    data_root: str = '/srv/home/chen/3dtracking'
    cache_dir: str = '/srv/home/chen/3dtracking/cache'

    # Patch extraction
    patch_size: Tuple[int, int, int] = (16, 24, 24)  # ZYX voxels around detection center

    # Graph construction radii (in XY pixel units; Z scaled by z_anisotropy)
    r_intra: float = 20.0   # intra-frame neighbor radius
    r_cross: float = 30.0   # cross-frame neighbor radius
    z_anisotropy: float = 0.5  # multiply Z coords by this before distance calc

    # Model
    feat_dim: int = 128
    gnn_layers: int = 2

    # Training
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    pos_weight: float = 5.0   # BCE weight on positive (same-cell) edges
    grad_clip: float = 1.0

    # Loss weights
    mitosis_loss_weight: float = 2.0   # weight of mitosis head loss vs edge loss
    mitosis_pos_weight: float = 20.0   # heavy upweight for rare dividing-parent class

    # Inference thresholds
    intra_threshold: float = 0.3    # merge same-frame detections of the same cell
    cross_threshold: float = 0.5    # cross-frame link threshold
    mitosis_threshold: float = 0.4  # both daughters must score >= this to call division
    mitosis_head_threshold: float = 0.3  # sigmoid threshold for mitosis head prediction

    # Data splits
    train_exps: Tuple[str, ...] = ('0501', '0507')
    val_exps: Tuple[str, ...] = ('0515',)
