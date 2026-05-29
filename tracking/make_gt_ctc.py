"""
Generate CTC-format ground truth data from one experiment.

Output structure:
  <out_root>/
    <seq>/                        raw images: t000.tif, t001.tif, ...
    <seq>_GT/
      SEG/                        man_seg000.tif, ... (GT instance masks)
      TRA/                        man_track000.tif, ... + man_track.txt

Place inference results in <out_root>/<seq>_RES/ (mask000.tif + res_track.txt)
then run the CTC evaluation tool from that directory.

man_track.txt format (one line per track):
  L  B  E  P
  label  first_frame  last_frame  parent_label (0 = no parent)

Usage:
    python -m tracking.make_gt_ctc --exp 0515 --out ctc_eval/0515 --seq 01
"""

import os
import argparse
import json
import shutil

import numpy as np
import tifffile

from tracking.config import Config
from tracking.preprocess import EXPERIMENT_DIRS, sorted_tifs


def make_gt_ctc(exp_id, data_root, out_root, seq='01'):
    dirs = EXPERIMENT_DIRS[exp_id]

    raw_files  = sorted_tifs(os.path.join(data_root, dirs['raw']))
    mask_files = sorted_tifs(os.path.join(data_root, dirs['mask']))
    mit_path   = os.path.join(data_root, dirs['mitosis'])
    n_frames   = len(raw_files)

    assert len(mask_files) == n_frames, (
        f'Raw/mask count mismatch: {len(raw_files)} vs {len(mask_files)}'
    )

    seq_dir = os.path.join(out_root, seq)
    gt_dir  = os.path.join(out_root, f'{seq}_GT')
    seg_dir = os.path.join(gt_dir, 'SEG')
    tra_dir = os.path.join(gt_dir, 'TRA')
    res_dir = os.path.join(out_root, f'{seq}_RES')   # placeholder for results

    for d in [seq_dir, seg_dir, tra_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    # -----------------------------------------------------------------------
    # Raw images → 01/t000.tif ...
    # -----------------------------------------------------------------------
    print(f'[{exp_id}] Copying {n_frames} raw images ...')
    for t, src in enumerate(raw_files):
        shutil.copy2(src, os.path.join(seq_dir, f't{t:03d}.tif'))
    print(f'  Done → {seq_dir}')

    # -----------------------------------------------------------------------
    # GT masks → SEG/man_seg%03d.tif and TRA/man_track%03d.tif
    # Also scan labels to build per-cell first/last frame table.
    # -----------------------------------------------------------------------
    print(f'[{exp_id}] Processing {n_frames} mask frames ...')
    cell_span = {}   # cell_id (int) -> [first_frame, last_frame]

    for t, src in enumerate(mask_files):
        mask = tifffile.imread(src)

        tifffile.imwrite(os.path.join(seg_dir, f'man_seg{t:03d}.tif'),
                         mask, compression='lzw')
        tifffile.imwrite(os.path.join(tra_dir, f'man_track{t:03d}.tif'),
                         mask, compression='lzw')

        for lbl in np.unique(mask):
            lbl = int(lbl)
            if lbl == 0:
                continue
            if lbl not in cell_span:
                cell_span[lbl] = [t, t]
            else:
                cell_span[lbl][1] = t   # update last frame

        if t % 20 == 0:
            print(f'  frame {t}/{n_frames}  unique cells so far: {len(cell_span)}')

    # -----------------------------------------------------------------------
    # Mitosis JSON → parent lookup
    # -----------------------------------------------------------------------
    with open(mit_path) as f:
        mit = json.load(f)

    child_to_parent = {}   # daughter_cell_id -> parent_cell_id
    for cid_str, info in mit['Children'].items():
        child_to_parent[int(cid_str)] = int(info['ParentID'])

    # -----------------------------------------------------------------------
    # Write TRA/man_track.txt
    # -----------------------------------------------------------------------
    txt_path = os.path.join(tra_dir, 'man_track.txt')
    with open(txt_path, 'w') as f:
        for lbl in sorted(cell_span):
            first, last = cell_span[lbl]
            parent = child_to_parent.get(lbl, 0)
            f.write(f'{lbl} {first} {last} {parent}\n')

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_mit = len(child_to_parent)
    print(f'\n[{exp_id}] Done.')
    print(f'  Tracks   : {len(cell_span)}  ({n_mit} mitosis daughters)')
    print(f'  Raw      : {seq_dir}/')
    print(f'  SEG      : {seg_dir}/')
    print(f'  TRA      : {tra_dir}/')
    print(f'  RES (put inference results here): {res_dir}/')
    print()
    print('Next steps:')
    print(f'  1. Run: python -m tracking.infer_ctc --exp {exp_id} --out {res_dir}')
    print(f'  2. Run CTC evaluation tool in: {out_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='0515')
    parser.add_argument('--out', default=None,
                        help='Output root dir (default: cache/ctc_gt/<exp>)')
    parser.add_argument('--seq', default='01',
                        help='Sequence index string, e.g. 01 or 02')
    args = parser.parse_args()

    cfg = Config()
    out = args.out or os.path.join(cfg.cache_dir, 'ctc_gt', args.exp)
    make_gt_ctc(args.exp, cfg.data_root, out, seq=args.seq)
