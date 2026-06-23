"""
Detailed analysis of tracking results vs GT.

Usage:
    python -m tracking.analyze_results \
        --gt   cache/ctc_gt/0004/01_GT/TRA \
        --res  results/triplet_largepatch \
        --res2 results/real_baseline \
        --out  results/analysis
"""

import os, argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from tracking.eval_tra import (
    centroids_from_mask, read_track_txt, hungarian_match, iso_dist, extract_divisions
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def track_lengths(tracks):
    return np.array([m['last'] - m['first'] + 1 for m in tracks.values()])


def per_frame_stats(gt_dir, res_dir, threshold=15.0, z_anisotropy=0.5):
    gt_masks  = sorted(f for f in os.listdir(gt_dir)  if f.startswith('man_track') and f.endswith('.tif'))
    res_masks = sorted(f for f in os.listdir(res_dir) if f.startswith('mask')      and f.endswith('.tif'))
    n = min(len(gt_masks), len(res_masks))

    frames, tp_arr, fp_arr, fn_arr, dist_arr, idsw_arr = [], [], [], [], [], []
    prev_pairs = {}

    for t in range(n):
        gt_c  = centroids_from_mask(tifffile.imread(os.path.join(gt_dir,  gt_masks[t])))
        res_c = centroids_from_mask(tifffile.imread(os.path.join(res_dir, res_masks[t])))
        pairs = hungarian_match(gt_c, res_c, threshold, z_anisotropy)
        mg = {gl for gl, _ in pairs}
        mr = {rl for _, rl in pairs}

        idsw = sum(1 for gl, rl in pairs if gl in prev_pairs and prev_pairs[gl] != rl)
        dists = [iso_dist(gt_c[gl].copy(), res_c[rl].copy(), z_anisotropy) for gl, rl in pairs]

        frames.append(t)
        tp_arr.append(len(pairs))
        fp_arr.append(len(res_c) - len(mr))
        fn_arr.append(len(gt_c) - len(mg))
        dist_arr.append(np.mean(dists) if dists else 0.0)
        idsw_arr.append(idsw)
        prev_pairs = {gl: rl for gl, rl in pairs}

    return dict(frames=np.array(frames),
                tp=np.array(tp_arr), fp=np.array(fp_arr),
                fn=np.array(fn_arr), dist=np.array(dist_arr),
                idsw=np.array(idsw_arr))


def track_coverage(gt_tracks, res_tracks, gt_dir, res_dir,
                   threshold=15.0, z_anisotropy=0.5):
    """For each GT track, compute fraction of frames where it is correctly matched."""
    gt_masks  = sorted(f for f in os.listdir(gt_dir)  if f.startswith('man_track') and f.endswith('.tif'))
    res_masks = sorted(f for f in os.listdir(res_dir) if f.startswith('mask')      and f.endswith('.tif'))
    n = min(len(gt_masks), len(res_masks))

    # Build per-frame matches
    frame_pairs = {}  # t → [(gt_lbl, res_lbl)]
    for t in range(n):
        gt_c  = centroids_from_mask(tifffile.imread(os.path.join(gt_dir,  gt_masks[t])))
        res_c = centroids_from_mask(tifffile.imread(os.path.join(res_dir, res_masks[t])))
        frame_pairs[t] = hungarian_match(gt_c, res_c, threshold, z_anisotropy)

    # Per GT track: fraction of frames matched, number of ID switches
    stats = {}
    for gt_tid, gt_meta in gt_tracks.items():
        matched = 0
        total   = 0
        prev_res = None
        sw = 0
        for t in range(gt_meta['first'], min(gt_meta['last'] + 1, n)):
            total += 1
            res_lbl = next((rl for gl, rl in frame_pairs.get(t, []) if gl == gt_tid), None)
            if res_lbl is not None:
                matched += 1
                if prev_res is not None and res_lbl != prev_res:
                    sw += 1
            prev_res = res_lbl
        stats[gt_tid] = dict(
            coverage=matched / max(total, 1),
            length=gt_meta['last'] - gt_meta['first'] + 1,
            id_switches=sw,
            is_daughter=gt_meta['parent'] != 0,
        )
    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(gt_dir, res_dirs, labels, out_dir, threshold=15.0, z_anisotropy=0.5):
    os.makedirs(out_dir, exist_ok=True)
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    gt_tracks = read_track_txt(os.path.join(gt_dir, 'man_track.txt'))

    # ---- 1. Per-frame Det-Precision, Det-Recall, ID-switches ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    all_stats = []
    for i, (res_dir, label) in enumerate(zip(res_dirs, labels)):
        s = per_frame_stats(gt_dir, res_dir, threshold, z_anisotropy)
        all_stats.append(s)
        prec = s['tp'] / np.maximum(s['tp'] + s['fp'], 1)
        rec  = s['tp'] / np.maximum(s['tp'] + s['fn'], 1)
        axes[0].plot(s['frames'], prec,  color=colors[i], lw=0.8, label=label, alpha=0.85)
        axes[1].plot(s['frames'], rec,   color=colors[i], lw=0.8, label=label, alpha=0.85)
        axes[2].plot(s['frames'], s['idsw'], color=colors[i], lw=0.8, label=label, alpha=0.85)

    axes[0].set_ylabel('Det-Precision'); axes[0].set_ylim(0, 1.05); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_ylabel('Det-Recall');    axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    axes[2].set_ylabel('ID switches/frame'); axes[2].grid(alpha=0.3)
    axes[2].set_xlabel('Frame')
    fig.suptitle('Per-frame Detection Quality & ID Switches', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_per_frame_detection.png'), dpi=150)
    plt.close()
    print('Saved: 1_per_frame_detection.png')

    # ---- 2. Track length distributions ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    gt_len = track_lengths(gt_tracks)
    bins = np.linspace(0, 300, 31)

    for i, (res_dir, label) in enumerate(zip(res_dirs, labels)):
        res_tracks = read_track_txt(os.path.join(res_dir, 'res_track.txt'))
        res_len = track_lengths(res_tracks)
        axes[i].hist(gt_len, bins=bins, alpha=0.5, color='gray', label='GT', density=True)
        axes[i].hist(res_len, bins=bins, alpha=0.7, color=colors[i], label=label, density=True)
        axes[i].set_xlabel('Track length (frames)')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{label}\n'
                          f'n={len(res_tracks)}  mean={res_len.mean():.0f}  '
                          f'median={np.median(res_len):.0f}')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    axes[-1].set_visible(False) if len(res_dirs) < 3 else None
    fig.suptitle('Track Length Distribution  (GT mean=203, median=249)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_track_lengths.png'), dpi=150)
    plt.close()
    print('Saved: 2_track_lengths.png')

    # ---- 3. GT track coverage heatmap ----
    fig, axes = plt.subplots(1, len(res_dirs), figsize=(7 * len(res_dirs), 5))
    if len(res_dirs) == 1:
        axes = [axes]

    for i, (res_dir, label) in enumerate(zip(res_dirs, labels)):
        res_tracks = read_track_txt(os.path.join(res_dir, 'res_track.txt'))
        cov = track_coverage(gt_tracks, res_tracks, gt_dir, res_dir, threshold, z_anisotropy)
        covs = [v['coverage'] for v in cov.values()]
        sw   = [v['id_switches'] for v in cov.values()]
        lens = [v['length']     for v in cov.values()]
        sc = axes[i].scatter(lens, covs, c=sw, cmap='hot_r', vmin=0, vmax=5,
                             s=30, alpha=0.7, edgecolors='none')
        plt.colorbar(sc, ax=axes[i], label='ID switches')
        axes[i].set_xlabel('GT track length (frames)')
        axes[i].set_ylabel('Coverage (fraction of frames matched)')
        pct_full = np.mean(np.array(covs) >= 0.95) * 100
        axes[i].set_title(f'{label}\n{pct_full:.0f}% of GT tracks ≥95% covered')
        axes[i].set_xlim(0, 300); axes[i].set_ylim(0, 1.05)
        axes[i].axhline(0.95, color='green', lw=0.8, ls='--', alpha=0.6)
        axes[i].grid(alpha=0.2)

    fig.suptitle('GT Track Coverage  (colour = # ID switches, green line = 95%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_track_coverage.png'), dpi=150)
    plt.close()
    print('Saved: 3_track_coverage.png')

    # ---- 4. Cumulative ID switches over time ----
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (s, label) in enumerate(zip(all_stats, labels)):
        ax.plot(s['frames'], np.cumsum(s['idsw']), color=colors[i], lw=1.5, label=label)
    ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative ID switches')
    ax.set_title('Cumulative ID Switches Over Time')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_cumulative_idsw.png'), dpi=150)
    plt.close()
    print('Saved: 4_cumulative_idsw.png')

    # ---- 5. Detection error over time (FP and FN stacked) ----
    fig, axes = plt.subplots(len(res_dirs), 1, figsize=(14, 4 * len(res_dirs)), sharex=True)
    if len(res_dirs) == 1:
        axes = [axes]
    for i, (s, label) in enumerate(zip(all_stats, labels)):
        axes[i].fill_between(s['frames'],  s['fp'], label='FP (extra detections)', color='#FF5722', alpha=0.6)
        axes[i].fill_between(s['frames'], -s['fn'], label='FN (missed detections)', color='#2196F3', alpha=0.6)
        axes[i].axhline(0, color='black', lw=0.5)
        axes[i].set_ylabel('Count'); axes[i].set_title(label)
        axes[i].legend(loc='upper right'); axes[i].grid(alpha=0.2)
    axes[-1].set_xlabel('Frame')
    fig.suptitle('False Positives (+) and False Negatives (−) Per Frame', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '5_fp_fn_over_time.png'), dpi=150)
    plt.close()
    print('Saved: 5_fp_fn_over_time.png')

    # ---- 6. Summary bar chart ----
    metrics = {
        'Det-F1':  [],
        'MOTA':    [],
        'IDF1':    [],
        'Div-F1':  [],
    }
    from tracking.eval_tra import evaluate
    for res_dir in res_dirs:
        m = evaluate(gt_dir, res_dir, threshold=threshold, z_anisotropy=z_anisotropy)
        metrics['Det-F1'].append(m['det_f1'])
        metrics['MOTA'].append(m['mota'])
        metrics['IDF1'].append(m['idf1'])
        metrics['Div-F1'].append(m['div_f1'])

    x = np.arange(len(metrics))
    w = 0.8 / len(res_dirs)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(labels):
        vals = [metrics[k][i] for k in metrics]
        bars = ax.bar(x + i * w - (len(res_dirs) - 1) * w / 2, vals,
                      w * 0.9, label=label, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(list(metrics.keys()), fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title('Summary Metrics Comparison')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '6_summary_comparison.png'), dpi=150)
    plt.close()
    print('Saved: 6_summary_comparison.png')

    print(f'\nAll plots saved to: {out_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt',    required=True)
    parser.add_argument('--res',   required=True, help='Primary result dir (triplet model)')
    parser.add_argument('--res2',  default=None,  help='Optional second result dir (baseline)')
    parser.add_argument('--out',   default='results/analysis')
    parser.add_argument('--threshold',    type=float, default=15.0)
    parser.add_argument('--z-anisotropy', type=float, default=0.5)
    args = parser.parse_args()

    res_dirs = [args.res]
    labels   = [os.path.basename(args.res)]
    if args.res2:
        res_dirs.append(args.res2)
        labels.append(os.path.basename(args.res2))

    plot_all(args.gt, res_dirs, labels, args.out,
             threshold=args.threshold, z_anisotropy=args.z_anisotropy)
