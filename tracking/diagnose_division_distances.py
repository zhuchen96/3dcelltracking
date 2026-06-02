"""
Measure the actual spatial distances between GT dividing mothers and their
daughters, to see whether r_cross=30 is covering them.

Usage:
    python -m tracking.diagnose_division_distances
"""

import os, json
import numpy as np
from tracking.config import Config
from tracking.preprocess import EXPERIMENT_DIRS
from tracking.dataset import _scale

def main():
    cfg = Config()
    exp_id = cfg.val_exps[0]
    dirs = EXPERIMENT_DIRS[exp_id]

    mit_path = os.path.join(cfg.data_root, dirs['mitosis'])
    with open(mit_path) as f:
        mit = json.load(f)
    parents_gt = {int(k): v for k, v in mit['Parents'].items()}
    p2c_gt = {}
    for cid, info in mit['Children'].items():
        p2c_gt.setdefault(info['ParentID'], []).append(int(cid))

    cache_dir = os.path.join(cfg.cache_dir, exp_id)

    dists_to_dau1 = []
    dists_to_dau2 = []
    n_outside_rcross = 0
    n_total = 0

    for pid, pinfo in parents_gt.items():
        t = pinfo['LastFrame']
        daughters = p2c_gt.get(pid, [])
        if len(daughters) < 2:
            continue

        # Load frame t (mother) and frame t+1 (daughters)
        d0 = np.load(os.path.join(cache_dir, f'frame_{t:04d}.npz'))
        d1 = np.load(os.path.join(cache_dir, f'frame_{t+1:04d}.npz'))

        ids0, c0 = d0['cell_ids'], d0['centers']
        ids1, c1 = d1['cell_ids'], d1['centers']

        # Find mother position
        m_idx = np.where(ids0 == pid)[0]
        if len(m_idx) == 0:
            continue
        mother_pos = c0[m_idx[0]]

        # Find daughter positions
        dau_dists = []
        for did in daughters:
            d_idx = np.where(ids1 == did)[0]
            if len(d_idx) == 0:
                continue
            dau_pos = c1[d_idx[0]]
            # Isotropic distance (same as used in match_cross_frame)
            dp = (mother_pos - dau_pos).astype(float)
            dp[0] *= cfg.z_anisotropy
            dist = np.linalg.norm(dp)
            dau_dists.append(dist)

        if len(dau_dists) < 2:
            continue

        dau_dists.sort()
        d1_dist, d2_dist = dau_dists[0], dau_dists[1]
        dists_to_dau1.append(d1_dist)
        dists_to_dau2.append(d2_dist)
        n_total += 1

        max_dist = max(d1_dist, d2_dist)
        if max_dist > cfg.r_cross:
            n_outside_rcross += 1
            print(f'  OUTSIDE r_cross: pid={pid} t={t}  '
                  f'd1={d1_dist:.1f}  d2={d2_dist:.1f}  (r_cross={cfg.r_cross})')

    print()
    print(f'GT divisions analysed : {n_total}')
    print(f'r_cross               : {cfg.r_cross}')
    print()
    print(f'Distance mother → nearer daughter:')
    print(f'  min={min(dists_to_dau1):.1f}  '
          f'median={np.median(dists_to_dau1):.1f}  '
          f'max={max(dists_to_dau1):.1f}  '
          f'p90={np.percentile(dists_to_dau1, 90):.1f}')
    print(f'Distance mother → farther daughter:')
    print(f'  min={min(dists_to_dau2):.1f}  '
          f'median={np.median(dists_to_dau2):.1f}  '
          f'max={max(dists_to_dau2):.1f}  '
          f'p90={np.percentile(dists_to_dau2, 90):.1f}')
    print()
    print(f'Divisions where farther daughter > r_cross: '
          f'{n_outside_rcross}/{n_total} '
          f'({100*n_outside_rcross/max(n_total,1):.0f}%)')

    # Suggest r_cross to cover 95% of divisions
    max_dists = [max(a, b) for a, b in zip(dists_to_dau1, dists_to_dau2)]
    for pct in [90, 95, 99, 100]:
        r = np.percentile(max_dists, pct)
        print(f'  r_cross to cover {pct}% of divisions: {r:.1f} px')

if __name__ == '__main__':
    main()
