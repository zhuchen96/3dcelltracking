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
    r_intra: float = 35.0   # intra-frame neighbor radius
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
    mitosis_loss_weight: float = 5.0    # weight of mitosis head loss vs edge loss
    mitosis_pos_weight: float = 20.0    # heavy upweight for rare dividing-parent class
    daughter_loss_weight: float = 5.0   # weight of daughter head loss
    daughter_pos_weight: float = 20.0   # same rarity as mother — ~2 per frame pair
    fg_loss_weight: float = 1.0         # weight of foreground head loss
    fg_pos_weight: float = 3.0          # FG:BG ratio ~1:2.3
    sister_loss_weight: float = 5.0     # weight of sister-edge loss (SimpleTrackingNet)
    sister_pos_weight: float = 100.0    # ~2 sister pairs per ~500 intra-t+1 edges → 1:250

    # Inference thresholds
    intra_threshold: float = 0.5          # merge same-frame detections of the same cell
    cross_threshold: float = 0.5          # cross-frame link threshold
    mitosis_threshold: float = 0.6        # affinity threshold for second daughter candidate
    mitosis_head_threshold: float = 0.9   # sigmoid threshold for mitosis head (mother)
    daughter_head_threshold: float = 0.3  # sigmoid threshold for daughter head (hard gate)
    fg_threshold: float = 0.5             # sigmoid threshold for foreground head at inference
    rescue_margin: int = 30               # YX pixels from border; interior unmatched t+1 clusters are rescued as daughters
    min_track_length: int = 5             # drop tracks shorter than this (frames); keeps daughters regardless
    phantom_max_frames: int = 2           # max frames a phantom node persists before the track ends

    # Training augmentation
    aug_drop_prob: float = 0.0            # probability of dropping an FG node during training (0 = off)

    # Data splits
    train_exps: Tuple[str, ...] = ('0501', '0507')
    val_exps: Tuple[str, ...] = ('0515',)
