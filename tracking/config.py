from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # Paths
    data_root: str = '/srv/home/chen/3dtracking'
    cache_dir: str = '/srv/home/chen/3dtracking/cache'

    # Patch extraction
    patch_size: Tuple[int, int, int] = (16, 40, 40)  # ZYX voxels around detection center

    # Graph construction radii (in XY pixel units; Z scaled by z_anisotropy)
    r_intra: float = 35.0   # intra-frame neighbor radius
    r_cross: float = 30.0   # cross-frame neighbor radius
    z_anisotropy: float = 1.292  # Z voxel / XY voxel = 0.42 µm / 0.325 µm

    # Model
    feat_dim: int = 128
    gnn_layers: int = 2
    in_channels: int = 1   # raw intensity only (16×40×40 patch, z-score normalised)

    # Training
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    pos_weight: float = 5.0   # BCE weight on positive (same-cell) edges
    grad_clip: float = 1.0

    # Loss weights
    fg_loss_weight: float = 1.0         # weight of foreground head loss
    fg_pos_weight: float = 3.0          # FG:BG ratio ~1:2.3
    sister_loss_weight: float = 10.0    # weight of sister-edge loss (SimpleTrackingNet)
    sister_pos_weight: float = 100.0    # ~2 sister pairs per ~500 intra-t+1 edges → 1:250

    # Inference thresholds
    intra_threshold: float = 0.5        # merge same-frame detections of the same cell
    cross_threshold: float = 0.4        # cross-frame link threshold
    mitosis_threshold: float = 0.4      # min conn affinity to each daughter candidate
    fg_threshold: float = 0.5           # foreground head threshold at inference
    min_track_length: int = 5           # drop tracks shorter than this; daughters always kept
    phantom_max_frames: int = 2         # max frames a phantom node persists before track ends

    # Metric learning
    metric_loss_weight: float = 0.0       # weight of cosine embedding loss on h_cnn cross-frame pairs

    # Training augmentation / schedule
    warmup_epochs: int   = 5             # linear LR warmup epochs (0 = no warmup, pure cosine)
    aug_drop_prob: float = 0.05          # probability of dropping an FG node during training

    # Data splits  (0004=test, 0003=val, all others=train)
    train_exps: Tuple[str, ...] = ('0001', '0002', '0501', '0507', '0515', '0517', '0522', '0528', '0605')
    val_exps: Tuple[str, ...] = ('0003',)
