"""
Grid-sweep inference thresholds on a single experiment with GT available.

Usage:
    python -m tracking.sweep_thresholds \
        --exp 0004 \
        --ckpt cache/checkpoints_simple/triplet_largepatch/best.pt \
        --gt   cache/ctc_gt/0004/01_GT/TRA

Sweeps cross_threshold and sister_threshold; reports a ranked table.
"""

import argparse
import os
import shutil
import tempfile
import itertools
import json

from tracking.config import Config
from tracking.infer_ctc_simple import run_ctc_inference_simple
from tracking.eval_tra import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',   required=True)
    parser.add_argument('--ckpt',  required=True)
    parser.add_argument('--gt',    required=True)
    parser.add_argument('--legacy', action='store_true')
    # Grids
    parser.add_argument('--cross',  nargs='+', type=float,
                        default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument('--sister', nargs='+', type=float,
                        default=[0.30, 0.40, 0.50, 0.60])
    parser.add_argument('--mito',   nargs='+', type=float,
                        default=[0.35, 0.40, 0.45])
    parser.add_argument('--fg',     nargs='+', type=float,
                        default=[0.5])
    parser.add_argument('--sort-by', default='idf1',
                        choices=['idf1', 'mota', 'div_f1', 'det_f1'])
    parser.add_argument('--eval-threshold', type=float, default=15.0)
    args = parser.parse_args()

    cfg = Config()
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(cfg.data_root, args.ckpt)

    combos = list(itertools.product(args.cross, args.sister, args.mito, args.fg))
    print(f'Running {len(combos)} configurations …\n')

    results = []
    tmpdir = tempfile.mkdtemp(prefix='sweep_')

    try:
        for i, (ct, st, mt, fg) in enumerate(combos):
            out = os.path.join(tmpdir, f'r{i:04d}')
            cfg2 = Config()
            cfg2.cross_threshold   = ct
            cfg2.fg_threshold      = fg

            run_ctc_inference_simple(
                cfg2, args.exp, args.ckpt, out,
                sister_threshold=st,
                mitosis_threshold=mt,
                legacy=args.legacy,
            )

            m = evaluate(args.gt, out, threshold=args.eval_threshold,
                         z_anisotropy=cfg2.z_anisotropy)

            results.append(dict(
                cross=ct, sister=st, mito=mt, fg=fg, **m
            ))

            shutil.rmtree(out)   # free disk space immediately

            print(f'[{i+1:3d}/{len(combos)}] '
                  f'cross={ct:.2f} sister={st:.2f} mito={mt:.2f} fg={fg:.2f}  '
                  f'IDF1={m["idf1"]:.4f}  MOTA={m["mota"]:.4f}  '
                  f'Div-F1={m["div_f1"]:.4f}  Det-F1={m["det_f1"]:.4f}')

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    results.sort(key=lambda r: r[args.sort_by], reverse=True)

    print(f'\n{"="*90}')
    print(f'Top 10 by {args.sort_by}')
    print(f'{"cross":>6}{"sister":>7}{"mito":>6}{"fg":>5}  '
          f'{"IDF1":>7}{"MOTA":>7}{"Div-F1":>8}{"Det-F1":>8}{"ID-SW":>7}')
    print('-' * 90)
    for r in results[:10]:
        print(f'{r["cross"]:6.2f}{r["sister"]:7.2f}{r["mito"]:6.2f}{r["fg"]:5.2f}  '
              f'{r["idf1"]:7.4f}{r["mota"]:7.4f}{r["div_f1"]:8.4f}'
              f'{r["det_f1"]:8.4f}{r["idsw"]:7d}')
    print('=' * 90)

    # Save full results
    out_json = f'sweep_{args.exp}_{os.path.basename(args.ckpt).replace(".pt","")}.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nFull results saved to {out_json}')


if __name__ == '__main__':
    main()
