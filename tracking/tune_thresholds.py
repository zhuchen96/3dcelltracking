"""
Grid search over inference thresholds on a fixed test sequence.

Usage:
    python -m tracking.tune_thresholds --exp 0004 \
        --ckpt cache/checkpoints_simple/baseline_v2/best.pt \
        --gt   cache/ctc_gt/0004/01_GT/TRA
"""

import argparse
import io
import itertools
import os
import sys
from contextlib import redirect_stdout

from tracking.config import Config
from tracking.eval_tra import evaluate
from tracking.infer_ctc_simple import run_ctc_inference_simple


def sweep(exp_id, ckpt_path, gt_dir, cfg, out_base):
    grid = {
        'sister_threshold':   [0.1, 0.2, 0.3, 0.4, 0.5],
        'mitosis_threshold':  [0.3, 0.4, 0.5, 0.6],
        'fg_threshold':       [0.3, 0.5],
        'cross_threshold':    [0.4, 0.5],
    }

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f'Running {len(combos)} combinations …\n')

    header = (f"{'sister':>7} {'mitosis':>8} {'fg':>5} {'cross':>6} | "
              f"{'MOTA':>6} {'IDF1':>6} {'DetF1':>6} | "
              f"{'DivP':>5} {'DivR':>5} {'DivF1':>6} {'D_TP':>5} {'D_FP':>4} {'D_FN':>4}")
    print(header)
    print('-' * len(header))

    results = []

    for combo in combos:
        params = dict(zip(keys, combo))

        # Temporarily override config fields
        cfg.fg_threshold    = params['fg_threshold']
        cfg.cross_threshold = params['cross_threshold']

        tag = (f"st{params['sister_threshold']:.2f}"
               f"_mt{params['mitosis_threshold']:.2f}"
               f"_fg{params['fg_threshold']:.2f}"
               f"_cr{params['cross_threshold']:.2f}")
        out_dir = os.path.join(out_base, tag)

        # Run inference silently
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_ctc_inference_simple(
                cfg, exp_id, ckpt_path, out_dir,
                sister_threshold=params['sister_threshold'],
                mitosis_threshold=params['mitosis_threshold'],
            )

        # Evaluate silently
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            m = evaluate(gt_dir, out_dir, threshold=15.0)

        row = dict(**params, **m)
        results.append(row)

        print(f"{params['sister_threshold']:>7.2f} "
              f"{params['mitosis_threshold']:>8.2f} "
              f"{params['fg_threshold']:>5.2f} "
              f"{params['cross_threshold']:>6.2f} | "
              f"{m['mota']:>6.4f} {m['idf1']:>6.4f} {m['det_f1']:>6.4f} | "
              f"{m['div_p']:>5.3f} {m['div_r']:>5.3f} {m['div_f1']:>6.4f} "
              f"{m['div_tp']:>5d} {m['div_fp']:>4d} {m['div_fn']:>4d}")
        sys.stdout.flush()

    # Ranking
    print('\n--- Top 5 by MOTA + Div-F1 ---')
    results.sort(key=lambda r: r['mota'] + r['div_f1'], reverse=True)
    for r in results[:5]:
        print(f"  sister={r['sister_threshold']:.2f} mitosis={r['mitosis_threshold']:.2f} "
              f"fg={r['fg_threshold']:.2f} cross={r['cross_threshold']:.2f}  →  "
              f"MOTA={r['mota']:.4f}  IDF1={r['idf1']:.4f}  "
              f"DivF1={r['div_f1']:.4f}  DivR={r['div_r']:.3f}")

    print('\n--- Top 5 by Div-F1 alone ---')
    results.sort(key=lambda r: r['div_f1'], reverse=True)
    for r in results[:5]:
        print(f"  sister={r['sister_threshold']:.2f} mitosis={r['mitosis_threshold']:.2f} "
              f"fg={r['fg_threshold']:.2f} cross={r['cross_threshold']:.2f}  →  "
              f"MOTA={r['mota']:.4f}  IDF1={r['idf1']:.4f}  "
              f"DivF1={r['div_f1']:.4f}  DivR={r['div_r']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',  default='0004')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gt',   required=True)
    parser.add_argument('--out',  default=None)
    args = parser.parse_args()

    cfg = Config()
    out_base = args.out or os.path.join(cfg.cache_dir, 'threshold_sweep', args.exp)
    os.makedirs(out_base, exist_ok=True)

    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(cfg.data_root, args.ckpt)

    sweep(args.exp, args.ckpt, args.gt, cfg, out_base)
