"""
Detailed failure analysis for tracking results.

Produces:
  A. Per-frame FP/FN/IDSW time series
  B. FP breakdown: background noise vs near-GT (merge artefact)
  C. FN breakdown: always-missing vs occasionally-missing cells
  D. ID-switch breakdown: which GT cells switch and when
  E. Division failure analysis: why each GT division was missed or wrongly fired

Usage:
    python -m tracking.analyze_errors \
        --gt  cache/ctc_gt/0004/01_GT/TRA \
        --res cache/threshold_sweep/real_baseline/st0.10_mt0.40_fg0.50_cr0.50 \
        --threshold 15
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Helpers (mirrors eval_tra, kept local to avoid coupling)
# ---------------------------------------------------------------------------

def read_track_txt(path):
    tracks = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            L, B, E, P = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            tracks[L] = {'first': B, 'last': E, 'parent': P}
    return tracks


def centroids_from_mask(mask):
    pos = np.argwhere(mask > 0)
    if len(pos) == 0:
        return {}
    lbls = mask[pos[:, 0], pos[:, 1], pos[:, 2]]
    order = np.argsort(lbls)
    pos, lbls = pos[order], lbls[order]
    split = np.where(np.diff(lbls))[0] + 1
    groups = np.split(pos, split)
    unique = lbls[np.concatenate([[0], split])]
    return {int(u): g.mean(0) for u, g in zip(unique, groups)}


def iso_dist(c0, c1, z_aniso=0.5):
    d = c0.copy() - c1.copy()
    d[0] *= z_aniso
    return float(np.linalg.norm(d))


def match_frame(gt_cents, res_cents, threshold, z_aniso=0.5):
    gt_l, res_l = list(gt_cents.keys()), list(res_cents.keys())
    if not gt_l or not res_l:
        return [], list(gt_l), list(res_l)
    C = np.full((len(gt_l), len(res_l)), 1e9)
    for i, gl in enumerate(gt_l):
        for j, rl in enumerate(res_l):
            d = iso_dist(gt_cents[gl], res_cents[rl], z_aniso)
            if d <= threshold:
                C[i, j] = d
    ri, ci = linear_sum_assignment(C)
    pairs, matched_gt, matched_res = [], set(), set()
    for r, c in zip(ri, ci):
        if C[r, c] <= threshold:
            pairs.append((gt_l[r], res_l[c], C[r, c]))
            matched_gt.add(gt_l[r])
            matched_res.add(res_l[c])
    fn_ids = [g for g in gt_l  if g not in matched_gt]
    fp_ids = [r for r in res_l if r not in matched_res]
    return pairs, fn_ids, fp_ids


def nearest_gt_dist(res_cent, gt_cents, z_aniso=0.5):
    if not gt_cents:
        return float('inf')
    return min(iso_dist(res_cent, c, z_aniso) for c in gt_cents.values())


# ---------------------------------------------------------------------------
# Load all frames
# ---------------------------------------------------------------------------

def load_all_frames(gt_dir, res_dir):
    gt_masks  = sorted(f for f in os.listdir(gt_dir)  if f.startswith('man_track') and f.endswith('.tif'))
    res_masks = sorted(f for f in os.listdir(res_dir) if f.startswith('mask')      and f.endswith('.tif'))
    n = min(len(gt_masks), len(res_masks))
    return gt_masks, res_masks, n


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(gt_dir, res_dir, threshold=15.0, z_aniso=0.5, neighbor_radius=20.0):
    gt_tracks  = read_track_txt(os.path.join(gt_dir,  'man_track.txt'))
    res_tracks = read_track_txt(os.path.join(res_dir, 'res_track.txt'))
    gt_masks, res_masks, n_frames = load_all_frames(gt_dir, res_dir)

    print(f'Frames: {n_frames}   GT tracks: {len(gt_tracks)}   RES tracks: {len(res_tracks)}')

    # -----------------------------------------------------------------------
    # A. Per-frame matching pass
    # -----------------------------------------------------------------------
    frame_fp   = []          # count per frame
    frame_fn   = []
    frame_idsw = []

    gt_cent_cache  = {}      # t -> {lbl: zyx}
    res_cent_cache = {}

    prev_gt_to_res = {}      # gt_lbl -> res_lbl (last frame)

    # Accumulators for later analysis
    gt_miss_count  = defaultdict(int)   # gt_lbl -> frames it was missed
    gt_total_count = defaultdict(int)   # gt_lbl -> frames it exists

    fp_near_gt  = []   # FP dets close to a GT cell (merge / double det)
    fp_far_gt   = []   # FP dets far from any GT cell (background)

    idsw_events = []   # (t, gt_lbl, old_res, new_res)

    for t in range(n_frames):
        gt_mask  = tifffile.imread(os.path.join(gt_dir,  gt_masks[t]))
        res_mask = tifffile.imread(os.path.join(res_dir, res_masks[t]))

        gt_c  = centroids_from_mask(gt_mask)
        res_c = centroids_from_mask(res_mask)
        gt_cent_cache[t]  = gt_c
        res_cent_cache[t] = res_c

        pairs, fn_ids, fp_ids = match_frame(gt_c, res_c, threshold, z_aniso)

        # GT miss tracking
        for gl in gt_c:
            gt_total_count[gl] += 1
        for gl in fn_ids:
            gt_miss_count[gl] += 1

        # FP characterisation
        for rl in fp_ids:
            d = nearest_gt_dist(res_c[rl], gt_c, z_aniso)
            if d <= neighbor_radius:
                fp_near_gt.append((t, rl, d))
            else:
                fp_far_gt.append((t, rl, d))

        # ID switches
        cur_gt_to_res = {gl: rl for gl, rl, _ in pairs}
        idsw = 0
        for gl, rl in cur_gt_to_res.items():
            if gl in prev_gt_to_res and prev_gt_to_res[gl] != rl:
                idsw += 1
                idsw_events.append((t, gl, prev_gt_to_res[gl], rl))
        prev_gt_to_res = cur_gt_to_res

        frame_fp.append(len(fp_ids))
        frame_fn.append(len(fn_ids))
        frame_idsw.append(idsw)

        if t % 50 == 0:
            print(f'  frame {t:3d}: GT={len(gt_c):3d}  RES={len(res_c):3d}  '
                  f'FP={len(fp_ids):3d}  FN={len(fn_ids):3d}  IDSW={idsw}')

    total_fp   = sum(frame_fp)
    total_fn   = sum(frame_fn)
    total_idsw = len(idsw_events)
    total_gt   = sum(gt_total_count.values())

    # -----------------------------------------------------------------------
    # B. FP breakdown
    # -----------------------------------------------------------------------
    print('\n' + '='*60)
    print('B. FALSE POSITIVE BREAKDOWN')
    print(f'   Total FP detections : {total_fp}')
    print(f'   Near a GT cell (<{neighbor_radius}px):  {len(fp_near_gt)}  '
          f'({100*len(fp_near_gt)/max(total_fp,1):.1f}%)  '
          f'← likely double/merge detections')
    print(f'   Far from GT (>{neighbor_radius}px): {len(fp_far_gt)}  '
          f'({100*len(fp_far_gt)/max(total_fp,1):.1f}%)  '
          f'← background noise')
    if fp_near_gt:
        dists = [d for _, _, d in fp_near_gt]
        print(f'   Near-GT dist  mean={np.mean(dists):.1f}  '
              f'median={np.median(dists):.1f}  max={np.max(dists):.1f} px')

    # -----------------------------------------------------------------------
    # C. FN breakdown
    # -----------------------------------------------------------------------
    print('\n' + '='*60)
    print('C. FALSE NEGATIVE (MISSED DETECTION) BREAKDOWN')
    print(f'   Total FN detections : {total_fn}')

    always_missed = {g for g, c in gt_miss_count.items() if c == gt_total_count[g]}
    sometimes_missed = {g for g, c in gt_miss_count.items() if 0 < c < gt_total_count[g]}

    print(f'   GT cells always missed  (0% detected)  : {len(always_missed)}')
    print(f'   GT cells sometimes missed               : {len(sometimes_missed)}')
    print(f'   (missed ≥50% of frames)                 : '
          f'{sum(1 for g in sometimes_missed if gt_miss_count[g]/gt_total_count[g]>=0.5)}')

    # FN temporal distribution — quartiles
    q = n_frames // 4
    fn_by_quarter = [sum(frame_fn[q*i:q*(i+1)]) for i in range(4)]
    print(f'   FN by quarter: {fn_by_quarter}  '
          f'(frames 0-{q}/{q}-{2*q}/{2*q}-{3*q}/{3*q}-{n_frames})')

    # -----------------------------------------------------------------------
    # D. ID switch breakdown
    # -----------------------------------------------------------------------
    print('\n' + '='*60)
    print('D. ID SWITCH BREAKDOWN')
    print(f'   Total ID switches : {total_idsw}')

    cells_with_idsw = defaultdict(list)
    for t, gl, old_r, new_r in idsw_events:
        cells_with_idsw[gl].append(t)

    print(f'   GT cells with ≥1 switch : {len(cells_with_idsw)}')
    repeat_switchers = {g: ts for g, ts in cells_with_idsw.items() if len(ts) > 1}
    print(f'   GT cells with ≥2 switches: {len(repeat_switchers)}')
    if repeat_switchers:
        for g, ts in sorted(repeat_switchers.items(), key=lambda x: -len(x[1]))[:5]:
            print(f'     GT cell {g}: {len(ts)} switches at frames {ts[:8]}')

    idsw_by_quarter = [
        sum(1 for t, *_ in idsw_events if q*i <= t < q*(i+1))
        for i in range(4)
    ]
    print(f'   IDSW by quarter: {idsw_by_quarter}')

    # Check overlap between ID-switch cells and dividing cells
    gt_div_parents = {m['parent'] for m in gt_tracks.values() if m['parent'] != 0}
    gt_daughters   = {tid for tid, m in gt_tracks.items() if m['parent'] != 0}
    idsw_in_div_parents  = {g for g in cells_with_idsw if g in gt_div_parents}
    idsw_in_daughters    = {g for g in cells_with_idsw if g in gt_daughters}
    print(f'   IDSW on dividing parents : {len(idsw_in_div_parents)} / {len(gt_div_parents)} parents')
    print(f'   IDSW on daughter cells   : {len(idsw_in_daughters)} / {len(gt_daughters)} daughters')

    # -----------------------------------------------------------------------
    # E. Division failure analysis
    # -----------------------------------------------------------------------
    print('\n' + '='*60)
    print('E. DIVISION FAILURE ANALYSIS')

    # GT division events: (parent_id, div_frame) = last frame parent exists
    gt_divs = {}
    for tid, m in gt_tracks.items():
        if m['parent'] != 0:
            div_frame = m['first'] - 1
            parent_id = m['parent']
            gt_divs.setdefault((parent_id, div_frame), []).append(tid)

    # RES division events
    res_divs = {}
    for tid, m in res_tracks.items():
        if m['parent'] != 0:
            div_frame = m['first'] - 1
            parent_id = m['parent']
            res_divs.setdefault((parent_id, div_frame), []).append(tid)

    print(f'\n   GT divisions  : {len(gt_divs)}')
    print(f'   RES divisions : {len(res_divs)}')

    # For each GT division, trace what happened
    tp_divs = fn_divs = 0
    fn_reasons = defaultdict(int)

    for (gt_parent, div_frame), gt_daughters_ids in sorted(gt_divs.items()):
        gt_parent_cent_at_div = gt_cent_cache.get(div_frame, {}).get(gt_parent)

        # Was the GT parent detected at div_frame?
        if gt_parent_cent_at_div is None:
            fn_reasons['gt_parent_not_in_mask'] += 1
            continue

        # Find the matching RES detection at div_frame
        res_c_at_div = res_cent_cache.get(div_frame, {})
        pairs_at_div, _, _ = match_frame(
            {gt_parent: gt_parent_cent_at_div}, res_c_at_div, threshold, z_aniso
        )
        res_parent_label = pairs_at_div[0][1] if pairs_at_div else None

        # Was the GT division caught by result?
        found_match = False
        for (res_parent, res_div_frame), _ in res_divs.items():
            if abs(res_div_frame - div_frame) <= 1:
                res_pc = res_cent_cache.get(res_div_frame, {}).get(res_parent)
                if res_pc is not None:
                    d = iso_dist(gt_parent_cent_at_div, res_pc, z_aniso)
                    if d <= threshold:
                        found_match = True
                        break

        if found_match:
            tp_divs += 1
            continue

        fn_divs += 1

        # Trace why the division was missed
        if res_parent_label is None:
            fn_reasons['parent_lost_at_div_frame'] += 1
            # Find when the parent was last tracked
            last_tracked = None
            for t2 in range(div_frame - 1, max(div_frame - 20, -1), -1):
                gt_c_t = gt_cent_cache.get(t2, {}).get(gt_parent)
                if gt_c_t is None:
                    break
                res_c_t = res_cent_cache.get(t2, {})
                p, _, _ = match_frame({gt_parent: gt_c_t}, res_c_t, threshold, z_aniso)
                if p:
                    last_tracked = t2
                    break
            gap = (div_frame - last_tracked) if last_tracked is not None else -1
            print(f'   MISS div parent={gt_parent:4d} frame={div_frame:3d}: '
                  f'parent LOST at div frame  '
                  f'(last tracked t={last_tracked}, gap={gap}fr)')
        else:
            # Parent is in result — check daughters
            next_frame = div_frame + 1
            gt_daug_cents = {d: gt_cent_cache.get(next_frame, {}).get(d)
                             for d in gt_daughters_ids}
            gt_daug_cents = {d: c for d, c in gt_daug_cents.items() if c is not None}
            res_c_next = res_cent_cache.get(next_frame, {})

            detected_daughters = []
            for d_id, d_cent in gt_daug_cents.items():
                p, _, _ = match_frame({d_id: d_cent}, res_c_next, threshold, z_aniso)
                if p:
                    detected_daughters.append((d_id, p[0][1]))

            if len(detected_daughters) < 2:
                fn_reasons['daughters_not_detected'] += 1
                print(f'   MISS div parent={gt_parent:4d} frame={div_frame:3d}: '
                      f'only {len(detected_daughters)}/{len(gt_daug_cents)} daughters detected')
            else:
                fn_reasons['daughters_detected_but_not_linked_as_division'] += 1
                # Both daughters detected — check if they were linked to parent
                d_res_labels = [rl for _, rl in detected_daughters]
                print(f'   MISS div parent={gt_parent:4d} frame={div_frame:3d}: '
                      f'both daughters detected (res_labels={d_res_labels}) '
                      f'but not assigned as division  '
                      f'← sister/mitosis threshold may be too high, or assigned as normal link')

    print(f'\n   Division TP: {tp_divs}   FN: {fn_divs}')
    print(f'   FN breakdown:')
    for reason, count in sorted(fn_reasons.items(), key=lambda x: -x[1]):
        print(f'     {reason}: {count}')

    # False positive divisions
    print(f'\n   RES division FP analysis:')
    fp_div_count = 0
    for (res_parent, res_div_frame), res_daug_ids in res_divs.items():
        res_pc = res_cent_cache.get(res_div_frame, {}).get(res_parent)
        if res_pc is None:
            continue
        matched_gt_div = False
        for (gt_parent, div_frame), _ in gt_divs.items():
            if abs(res_div_frame - div_frame) <= 1:
                gt_pc = gt_cent_cache.get(div_frame, {}).get(gt_parent)
                if gt_pc is not None and iso_dist(res_pc, gt_pc, z_aniso) <= threshold:
                    matched_gt_div = True
                    break
        if not matched_gt_div:
            fp_div_count += 1
            # What is this cell in the GT?
            gt_match = None
            gt_c_at_frame = gt_cent_cache.get(res_div_frame, {})
            pairs, _, _ = match_frame({res_parent: res_pc}, gt_c_at_frame, threshold, z_aniso)
            gt_match = pairs[0][0] if pairs else None
            is_gt_divider = gt_match in gt_div_parents if gt_match else False
            print(f'   FP div  res_parent={res_parent:4d} frame={res_div_frame:3d}: '
                  f'GT cell={gt_match}  is_actual_divider={is_gt_divider}')
    print(f'   Total FP divisions: {fp_div_count}')

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    mota = 1.0 - (total_fp + total_fn + total_idsw) / max(total_gt, 1)
    print('\n' + '='*60)
    print('SUMMARY')
    print(f'  Total GT dets  : {total_gt}')
    print(f'  FP             : {total_fp}  ({100*total_fp/max(total_gt,1):.1f}% of GT)')
    print(f'    background   : {len(fp_far_gt)}  ({100*len(fp_far_gt)/max(total_fp,1):.1f}% of FP)')
    print(f'    near-GT      : {len(fp_near_gt)}  ({100*len(fp_near_gt)/max(total_fp,1):.1f}% of FP)')
    print(f'  FN             : {total_fn}  ({100*total_fn/max(total_gt,1):.1f}% of GT)')
    print(f'    always missed: {len(always_missed)} cells')
    print(f'    intermittent : {len(sometimes_missed)} cells')
    print(f'  ID switches    : {total_idsw}')
    print(f'  MOTA           : {mota:.4f}')
    print(f'  Div TP/FP/FN   : {tp_divs} / {fp_div_count} / {fn_divs}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt',  required=True)
    parser.add_argument('--res', required=True)
    parser.add_argument('--threshold',       type=float, default=15.0)
    parser.add_argument('--neighbor-radius', type=float, default=20.0,
                        help='FP within this distance of a GT cell = near-GT (default 20px)')
    args = parser.parse_args()
    analyze(args.gt, args.res, args.threshold, neighbor_radius=args.neighbor_radius)
