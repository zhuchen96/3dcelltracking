"""
Comprehensive track quality analysis: why does pred have so many more tracks than GT?

Categories for each predicted track:
  A. Pure FP background : pred track never overlaps any GT cell in any frame
  B. Duplicate same-frame: pred track overlaps a GT cell that another pred track
                           also overlaps in the SAME frame (split detection)
  C. GT-track fragment  : pred track covers a real GT cell, but that GT cell is
                           split across 2+ pred tracks over time (discontinuous)
  D. ID-switch track    : pred track covers GT cell X in some frames, GT cell Y
                           in others (wrong cross-frame link)
  E. False-div daughter : pred track has parent > 0, but GT has no division there
  F. Normal             : pred track covers one GT cell consistently throughout

For GT tracks:
  - How many GT tracks are split into multiple pred tracks (fragmentation count)
  - How many GT cells are completely missed

Usage:
    python -m tracking.diagnose_tracks
"""

import os
import glob
from collections import defaultdict

import numpy as np
import tifffile

GT_DIR    = '/srv/home/chen/3dtracking/cache/ctc_eval/2nd_with_mito/01_GT/TRA'
PRED_DIR  = '/srv/home/chen/3dtracking/cache/ctc_0515'
GT_TXT    = os.path.join(GT_DIR,   'man_track.txt')
PRED_TXT  = os.path.join(PRED_DIR, 'res_track.txt')


# ---------------------------------------------------------------------------
# Load track metadata
# ---------------------------------------------------------------------------

def load_tracks(path):
    """Returns dict {id: (first, last, parent)}"""
    tracks = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            tid, b, e, p = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            tracks[tid] = (b, e, p)
    return tracks


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    gt_tracks   = load_tracks(GT_TXT)
    pred_tracks = load_tracks(PRED_TXT)

    gt_files   = sorted(glob.glob(os.path.join(GT_DIR,   'man_track*.tif')))
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, 'mask*.tif')))
    n_frames   = min(len(gt_files), len(pred_files))

    print(f'GT tracks  : {len(gt_tracks)}')
    print(f'Pred tracks: {len(pred_tracks)}')
    print(f'Excess     : {len(pred_tracks) - len(gt_tracks)}')
    print(f'Frames     : {n_frames}')
    print()

    # -- per-frame mappings --
    # pred_primary_gt[pred_id] = {t: gt_id_with_most_overlap}
    # gt_covered_by[gt_id]     = {t: set(pred_ids)}
    # pred_fp_frames[pred_id]  = # frames where pred_id had 0 GT overlap

    pred_primary_gt  = defaultdict(dict)   # pred_id -> {t: gt_id}
    gt_covered_by    = defaultdict(lambda: defaultdict(set))  # gt_id -> {t: set(pred_ids)}
    pred_fp_frame_ct = defaultdict(int)    # pred_id -> # FP frames (no GT overlap)
    pred_total_frames = defaultdict(int)   # pred_id -> # frames it appears

    # For duplicate detection: per (t, gt_id) -> set of pred_ids
    frame_gt_to_preds = defaultdict(lambda: defaultdict(set))  # t -> gt_id -> set(pred_ids)

    print('Scanning frames...')
    for fi, (gf, pf) in enumerate(zip(gt_files, pred_files)):
        gt   = tifffile.imread(gf).ravel().astype(np.int32)
        pred = tifffile.imread(pf).ravel().astype(np.int32)
        t = fi

        # --- which pred_ids appear this frame ---
        pred_ids_this = np.unique(pred[pred > 0])
        for pid in pred_ids_this:
            pred_total_frames[pid] += 1

        # --- overlap: voxels where gt > 0 ---
        gt_pos  = gt > 0
        if gt_pos.any():
            g_vals = gt[gt_pos]
            p_vals = pred[gt_pos]

            # encode (gt_id, pred_id) pair
            MAX_P = int(pred.max()) + 1 if pred.max() > 0 else 1
            pairs = g_vals.astype(np.int64) * 200000 + p_vals.astype(np.int64)
            unique_pairs, counts = np.unique(pairs, return_counts=True)
            gt_part   = (unique_pairs // 200000).astype(np.int32)
            pred_part = (unique_pairs % 200000).astype(np.int32)

            # For each gt_id: which pred_ids overlap it this frame?
            for gid, pid, cnt in zip(gt_part, pred_part, counts):
                if pid == 0:
                    continue
                frame_gt_to_preds[t][int(gid)].add(int(pid))
                gt_covered_by[int(gid)][t].add(int(pid))

            # For each pred_id: what is its primary GT cell this frame?
            # (GT cell with most overlap)
            pred_to_gt_counts = defaultdict(lambda: defaultdict(int))
            for gid, pid, cnt in zip(gt_part, pred_part, counts):
                if pid == 0 or gid == 0:
                    continue
                pred_to_gt_counts[int(pid)][int(gid)] += int(cnt)

            for pid_this in pred_ids_this:
                pid_this = int(pid_this)
                if pid_this in pred_to_gt_counts:
                    best_gt = max(pred_to_gt_counts[pid_this],
                                  key=pred_to_gt_counts[pid_this].get)
                    pred_primary_gt[pid_this][t] = best_gt
                else:
                    pred_fp_frame_ct[pid_this] += 1  # no GT overlap this frame
        else:
            # No GT cells → all pred this frame are FP
            for pid in pred_ids_this:
                pred_fp_frame_ct[int(pid)] += 1

        if fi % 50 == 0:
            print(f'  frame {fi:03d}/{n_frames}')

    # -----------------------------------------------------------------------
    # Categorize each predicted track
    # -----------------------------------------------------------------------

    cat_bg_fp        = []   # A: never overlaps any GT cell
    cat_duplicate    = []   # B: same GT cell, same frame, as another pred track
    cat_fragment     = []   # C: covers real GT cell but GT cell also in other pred tracks
    cat_id_switch    = []   # D: primary GT cell changes mid-track
    cat_false_div    = []   # E: daughter track (parent>0) but GT has no division
    cat_normal       = []   # F: clean 1-to-1

    # GT division info: which GT tracks are daughters? (parent > 0)
    gt_daughter_ids = {tid for tid, (b, e, p) in gt_tracks.items() if p > 0}
    # GT parent ids at what frame?
    gt_divisions = {}  # parent_gt_id -> (frame_of_division, [daughter_ids])
    for tid, (b, e, p) in gt_tracks.items():
        if p > 0:
            # parent p ends at b-1; daughters start at b
            gt_divisions.setdefault(p, {'frame': b, 'daughters': []})
            gt_divisions[p]['daughters'].append(tid)

    for pid, (pb, pe, pp) in pred_tracks.items():
        pid = int(pid)
        total_frames_appearing = pred_total_frames.get(pid, 0)
        fp_frames = pred_fp_frame_ct.get(pid, 0)

        # A: Pure FP background — never overlaps any GT cell
        if pid not in pred_primary_gt or fp_frames == total_frames_appearing:
            cat_bg_fp.append(pid)
            continue

        # Get primary GT timeline
        gt_timeline = pred_primary_gt[pid]  # {t: gt_id}
        gt_ids_seen = set(gt_timeline.values())

        # D: ID switch — primary GT cell changes over time
        if len(gt_ids_seen) > 1:
            cat_id_switch.append(pid)
            continue

        # Single primary GT cell
        primary_gt = next(iter(gt_ids_seen))

        # E: False division daughter — pred has parent but GT has no division
        if pp > 0:
            # Find what pred_parent is covering at the frame before this track starts
            pred_parent_primary = pred_primary_gt.get(int(pp), {})
            parent_gt_at_split = pred_parent_primary.get(pb - 1, None)
            # Check if GT has a real division of that GT cell at that frame
            has_gt_division = (parent_gt_at_split is not None and
                               parent_gt_at_split in gt_divisions and
                               abs(gt_divisions[parent_gt_at_split]['frame'] - pb) <= 2)
            if not has_gt_division:
                cat_false_div.append(pid)
                continue

        # B: Duplicate — another pred track covers the same GT cell in same frame
        is_duplicate = False
        for t, gid in gt_timeline.items():
            preds_covering = frame_gt_to_preds[t].get(gid, set())
            if len(preds_covering) > 1 and any(
                    other_pid != pid and other_pid in pred_tracks
                    for other_pid in preds_covering):
                is_duplicate = True
                break
        if is_duplicate:
            cat_duplicate.append(pid)
            continue

        # C: Fragment — this pred track covers a GT cell, but that GT cell
        #    has OTHER pred tracks covering it at different times
        all_pred_for_this_gt = set()
        for t, pids in gt_covered_by[primary_gt].items():
            all_pred_for_this_gt.update(pids)
        all_pred_for_this_gt.discard(pid)
        # Filter to pred tracks that exist in pred_tracks (not transient noise)
        covering_others = {p for p in all_pred_for_this_gt if p in pred_tracks}
        if covering_others:
            cat_fragment.append(pid)
            continue

        cat_normal.append(pid)

    # -----------------------------------------------------------------------
    # GT track fragmentation summary
    # -----------------------------------------------------------------------

    gt_frag_counts = {}  # gt_id -> number of distinct pred tracks covering it
    gt_never_detected = []

    for gid in gt_tracks:
        preds_ever = set()
        for t, pids in gt_covered_by.get(gid, {}).items():
            preds_ever.update(pids)
        preds_ever = {p for p in preds_ever if p in pred_tracks}
        gt_frag_counts[gid] = len(preds_ever)
        if len(preds_ever) == 0:
            gt_never_detected.append(gid)

    frag_hist = defaultdict(int)
    for gid, n in gt_frag_counts.items():
        frag_hist[n] += 1

    gt_frag_excess = sum(max(0, n - 1) for n in gt_frag_counts.values())

    # -----------------------------------------------------------------------
    # False division analysis
    # -----------------------------------------------------------------------
    pred_div_tids = [tid for tid, (b, e, p) in pred_tracks.items() if p > 0]
    pred_div_count = len(pred_div_tids) // 2  # each division creates 2 daughters

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print()
    print('=' * 60)
    print('  TRACK COUNT BREAKDOWN')
    print('=' * 60)
    print(f'  GT  tracks total     : {len(gt_tracks)}')
    print(f'  Pred tracks total    : {len(pred_tracks)}')
    print(f'  Excess (pred - GT)   : {len(pred_tracks) - len(gt_tracks)}')
    print()
    print('  Predicted track categories:')
    print(f'    A. Pure FP (background)   : {len(cat_bg_fp):5d}')
    print(f'    B. Duplicate same-frame   : {len(cat_duplicate):5d}')
    print(f'    C. Fragment (track break) : {len(cat_fragment):5d}')
    print(f'    D. ID-switch track        : {len(cat_id_switch):5d}')
    print(f'    E. False division daughter: {len(cat_false_div):5d}')
    print(f'    F. Normal (correct)       : {len(cat_normal):5d}')
    total_check = (len(cat_bg_fp) + len(cat_duplicate) + len(cat_fragment) +
                   len(cat_id_switch) + len(cat_false_div) + len(cat_normal))
    print(f'    Total accounted for       : {total_check:5d}')
    print()

    print('  GT track fragmentation:')
    print(f'    GT tracks never detected  : {len(gt_never_detected)}')
    for n_frags in sorted(frag_hist.keys()):
        label = ('(missed)' if n_frags == 0
                 else '(clean)' if n_frags == 1
                 else f'(split into {n_frags})')
        print(f'    GT tracks covered by {n_frags} pred : {frag_hist[n_frags]:4d}  {label}')
    print(f'    Excess tracks from splits  : {gt_frag_excess}  '
          f'(each split into k → k-1 extra)')
    print()

    print('  Division summary:')
    gt_div_count = len([tid for tid, (b, e, p) in gt_tracks.items() if p > 0]) // 2
    print(f'    GT  divisions             : {gt_div_count}')
    print(f'    Pred divisions            : {pred_div_count}')
    print(f'    False division daughters  : {len(cat_false_div)}  '
          f'(→ {len(cat_false_div)//2} FP division events)')
    print()

    print('  Excess track accounting:')
    acc_bg    = len(cat_bg_fp)
    acc_dup   = len(cat_duplicate)
    acc_frag  = len(cat_fragment) - len(cat_fragment)  # fragments are also in gt_frag
    acc_false = len(cat_false_div)
    acc_id    = len(cat_id_switch)
    total_acc = acc_bg + acc_dup + gt_frag_excess + acc_false + acc_id
    print(f'    FP background tracks      : +{acc_bg}')
    print(f'    Duplicate detections      : +{acc_dup}')
    print(f'    GT frag extras (split-1)  : +{gt_frag_excess}')
    print(f'    False division daughters  : +{acc_false}')
    print(f'    ID-switch tracks          : +{acc_id}')
    print(f'    Sum                       : +{total_acc}  '
          f'(vs actual excess {len(pred_tracks) - len(gt_tracks)})')
    print('=' * 60)


if __name__ == '__main__':
    main()
