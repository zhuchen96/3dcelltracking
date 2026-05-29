"""
Visualize inference tracks as an annotated MIP video.

Per-frame rendering:
  background  — Z-MIP of raw image (percentile-normalized, grayscale→RGB)
  cyan dot + tail    — ongoing tracked cells
  yellow dot + tail  — new cells (appeared, no parent)
  magenta dot + tail — division daughters
  orange dashed line — division event: parent centroid → daughter centroid

Usage:
    python -m tracking.viz_tracks \\
        --exp 0515 \\
        --res cache/ctc_0515 \\
        --out cache/viz_0515.mp4 \\
        --tail 10 --fps 10 --ds 3
"""

import os
import argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

from tracking.config import Config
from tracking.eval_tra import centroids_from_mask, read_track_txt
from tracking.preprocess import EXPERIMENT_DIRS, sorted_tifs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_norm(img2d, lo=1.0, hi=99.5):
    """uint16 2D → uint8, percentile stretch."""
    v = img2d.astype(np.float32)
    vlo, vhi = np.percentile(v, lo), np.percentile(v, hi)
    v = np.clip((v - vlo) / max(vhi - vlo, 1), 0.0, 1.0)
    return (v * 255).astype(np.uint8)


def render_frame(mip8, centroids_t, track_meta, cent_by_tid,
                 t, tail_len, ds, dot_r=3):
    """
    Returns an (H, W, 3) uint8 RGB numpy array for frame t.

    centroids_t : dict tid → [z, y, x]  (active detections at frame t)
    cent_by_tid : dict tid → {frame → [z, y, x]}  (full trajectory)
    """
    H, W = mip8.shape
    Hd = (H // ds) & ~1   # round down to even (libx264 requirement)
    Wd = (W // ds) & ~1

    dpi = 100
    fig, ax = plt.subplots(figsize=(Wd / dpi, Hd / dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()

    # Background: grayscale MIP
    bg = np.stack([mip8[::ds, ::ds]] * 3, axis=-1)
    ax.imshow(bg, aspect='equal', interpolation='nearest', origin='upper')
    ax.set_xlim(0, Wd - 1)
    ax.set_ylim(Hd - 1, 0)

    for tid, meta in track_meta.items():
        if meta['first'] > t or meta['last'] < t:
            continue
        c = centroids_t.get(tid)
        if c is None:
            continue

        y_px = c[1] / ds
        x_px = c[2] / ds

        is_div_daughter = (meta['parent'] != 0 and meta['first'] == t)
        is_new          = (meta['parent'] == 0  and meta['first'] == t)

        if is_div_daughter:
            color = 'magenta'
        elif is_new:
            color = 'yellow'
        else:
            color = 'cyan'

        # Tail: draw segments with increasing alpha toward current position
        traj = cent_by_tid.get(tid, {})
        tail_frames = [f for f in range(max(meta['first'], t - tail_len), t + 1)
                       if f in traj]
        if len(tail_frames) > 1:
            pts_y = [traj[f][1] / ds for f in tail_frames]
            pts_x = [traj[f][2] / ds for f in tail_frames]
            n = len(pts_y)
            for i in range(n - 1):
                alpha = 0.15 + 0.65 * (i / (n - 1))
                lw    = 0.5 + 0.8 * (i / (n - 1))
                ax.plot([pts_x[i], pts_x[i+1]], [pts_y[i], pts_y[i+1]],
                        '-', color=color, alpha=alpha, linewidth=lw,
                        solid_capstyle='round')

        # Current dot
        ax.plot(x_px, y_px, 'o', color=color, markersize=dot_r,
                markeredgewidth=0, zorder=5)

        # Division line: parent last centroid → daughter current centroid
        if is_div_daughter:
            pid = meta['parent']
            pm  = track_meta.get(pid)
            if pm is not None:
                pc = cent_by_tid.get(pid, {}).get(pm['last'])
                if pc is not None:
                    ax.plot([pc[2] / ds, x_px], [pc[1] / ds, y_px],
                            '--', color='orange', linewidth=1.2, alpha=0.95,
                            zorder=4)

    ax.text(4 / ds, 4 / ds, f't={t:03d}',
            color='white', fontsize=max(5, 8 // ds), va='top', zorder=10)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame_rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return frame_rgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_video(cfg, exp_id, res_dir, out_path,
               tail_len=10, fps=10, ds=3, dot_r=3):
    dirs     = EXPERIMENT_DIRS[exp_id]
    raw_dir  = os.path.join(cfg.data_root, dirs['raw'])
    raw_tifs = sorted_tifs(raw_dir)
    n_frames = len(raw_tifs)

    track_meta = read_track_txt(os.path.join(res_dir, 'res_track.txt'))
    res_masks  = sorted(f for f in os.listdir(res_dir)
                        if f.startswith('mask') and f.endswith('.tif'))
    n_frames = min(n_frames, len(res_masks))

    print(f'Frames: {n_frames}   Tracks: {len(track_meta)}')

    # ---- Precompute all centroids ----
    print('Loading centroids from masks ...')
    cent_by_tid = {}   # tid → {frame → [z, y, x]}
    frame_cents = []   # per-frame: tid → [z, y, x]
    for t in range(n_frames):
        mask = tifffile.imread(os.path.join(res_dir, res_masks[t]))
        c    = centroids_from_mask(mask)   # {label: array([z,y,x])}
        fc   = {}
        for tid, pos in c.items():
            fc[tid] = pos
            cent_by_tid.setdefault(tid, {})[t] = pos
        frame_cents.append(fc)
        if t % 50 == 0:
            print(f'  frame {t}/{n_frames}')

    # ---- Render frames ----
    print('Rendering frames ...')
    writer = imageio.get_writer(out_path, fps=fps,
                                output_params=['-vcodec', 'libx264',
                                               '-pix_fmt', 'yuv420p',
                                               '-crf', '18'])
    for t in range(n_frames):
        raw = tifffile.imread(raw_tifs[t])       # (Z, Y, X) uint16
        mip = raw.max(axis=0)                    # (Y, X)
        mip8 = percentile_norm(mip)

        frame_rgb = render_frame(
            mip8, frame_cents[t], track_meta, cent_by_tid,
            t, tail_len, ds, dot_r,
        )
        writer.append_data(frame_rgb)

        if t % 20 == 0:
            print(f'  rendered {t}/{n_frames}')

    writer.close()
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',  default='0515')
    parser.add_argument('--res',  default=None,
                        help='res dir (default: cache/ctc_<exp>)')
    parser.add_argument('--out',  default=None,
                        help='output .mp4 path')
    parser.add_argument('--tail', type=int,   default=10,
                        help='tail length in frames')
    parser.add_argument('--fps',  type=int,   default=10)
    parser.add_argument('--ds',   type=int,   default=3,
                        help='spatial downsample factor (3 → ~860×120 px)')
    parser.add_argument('--dot',  type=int,   default=3,
                        help='dot radius in pixels (after downsampling)')
    args = parser.parse_args()

    cfg = Config()
    res = args.res or os.path.join(cfg.cache_dir, f'ctc_{args.exp}')
    out = args.out or os.path.join(cfg.cache_dir, f'viz_{args.exp}.mp4')
    make_video(cfg, args.exp, res, out,
               tail_len=args.tail, fps=args.fps, ds=args.ds, dot_r=args.dot)
