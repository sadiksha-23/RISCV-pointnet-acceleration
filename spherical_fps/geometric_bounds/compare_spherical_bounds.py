#!/usr/bin/env python3
"""Exact KITTI FPS: compare original XYZ enclosing-sphere region bounds
against cosine-law bounds calculated over spherical azimuth/elevation/radius ranges.
This is offline analysis, not a faster FPS implementation.
"""
import argparse
import csv
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spherical_kitti_tests import build_regions, exact_fps, load_kitti


def angle_cos_min(lo, hi, psi):
    """Exact minimum of cos(az-psi) over each azimuth interval, broadcast [S,R]."""
    left, right = np.cos(lo - psi), np.cos(hi - psi)
    ans = np.minimum(left, right)
    # Include the angle psi + pi + 2*pi*z if contained in [lo,hi].
    z = np.ceil((lo - psi - np.pi) / (2 * np.pi))
    trough = psi + np.pi + 2 * np.pi * z
    return np.where(trough <= hi + 1e-13, -1., ans)


def spherical_upper_bound_all_samples(points, origin, regions, chosen, member):
    """Return [samples, regions] rigorous region distance upper bounds.

    Write p=origin+r*u(az,el), v=s-origin.
    ||p-s||^2 = r^2 + ||v||^2 - 2*r*(v dot u).
    Min v dot u on az/el intervals is computed analytically; maximum
    quadratic distance over radial interval occurs at one of its endpoints.
    Actual per-region point extrema (with conservatism) define intervals.
    """
    nr = len(regions['counts'])
    r, az, el = regions['r'], regions['az'], regions['el']
    lo_r = np.full(nr, np.inf)
    hi_r = np.full(nr, -np.inf)
    lo_a = np.full(nr, np.inf)
    hi_a = np.full(nr, -np.inf)
    lo_e = np.full(nr, np.inf)
    hi_e = np.full(nr, -np.inf)
    for dst, op, src in [(lo_r, np.minimum.at, r), (hi_r, np.maximum.at, r),
                         (lo_a, np.minimum.at, az), (hi_a, np.maximum.at, az),
                         (lo_e, np.minimum.at, el), (hi_e, np.maximum.at, el)]:
        op(dst, member, src)
    lo_r = np.maximum(0., lo_r - 1e-9)[None, :]
    hi_r = (hi_r + 1e-9)[None, :]
    lo_a = (lo_a - 1e-12)[None, :]
    hi_a = (hi_a + 1e-12)[None, :]
    lo_e = (lo_e - 1e-12)[None, :]
    hi_e = (hi_e + 1e-12)[None, :]

    v = chosen - origin
    xy = np.hypot(v[:, 0], v[:, 1])[:, None]
    vz = v[:, 2:3]
    psi = np.arctan2(v[:, 1], v[:, 0])[:, None]
    v_sq = np.sum(v * v, axis=1)[:, None]

    min_cosaz = angle_cos_min(lo_a, hi_a, psi)
    A = xy * min_cosaz
    B = vz
    # Minimize A cos(el) + B sin(el) on [lo_e,hi_e].
    val_lo = A * np.cos(lo_e) + B * np.sin(lo_e)
    val_hi = A * np.cos(hi_e) + B * np.sin(hi_e)
    dotmin = np.minimum(val_lo, val_hi)
    stationary_min = (np.arctan2(B, A) + np.pi + np.pi) % (2 * np.pi) - np.pi
    within = (stationary_min >= lo_e) & (stationary_min <= hi_e)
    dotmin = np.minimum(dotmin, np.where(within, -np.hypot(A, B), np.inf))

    d2_lo = lo_r**2 + v_sq - 2 * lo_r * dotmin
    d2_hi = hi_r**2 + v_sq - 2 * hi_r * dotmin
    ans = np.sqrt(np.maximum(np.maximum(d2_lo, d2_hi), 0.))
    # Explicit floating point cushion; crucial for conservative pruning.
    return ans + 1e-7 * (1 + ans)


def all_queries(points, selected, before, reg, origin, name, grid_name, verify_states):
    chosen = points[selected]
    centers, radius, reps = reg['center'], reg['radius'], reg['representative']
    dist_centers = np.linalg.norm(chosen[:, None, :] - centers[None, :, :], axis=2)
    xyz_each = dist_centers + radius[None, :]
    sph_each = spherical_upper_bound_all_samples(points, origin, reg, chosen, reg['member'])
    rep_each = np.linalg.norm(chosen[:, None, :] - reps[None, :, :], axis=2)
    xyz_cumulative = np.minimum.accumulate(xyz_each, axis=0)
    sph_cumulative = np.minimum.accumulate(sph_each, axis=0)
    rep_cumulative = np.minimum.accumulate(rep_each, axis=0)
    records = []
    valid_checks = 0
    for k in range(1, len(selected)):
        lb = float(rep_cumulative[k-1].max())
        bounds = {
            'xyz': xyz_cumulative[k-1],
            'spherical_cos': sph_cumulative[k-1],
            'combined': np.minimum(xyz_cumulative[k-1], sph_cumulative[k-1]),
        }
        eps = 1e-8 * max(1., lb, float(np.max(xyz_cumulative[k-1])))
        w = int(reg['member'][selected[k]])
        for method, ub in bounds.items():
            excluded = ub < lb - eps
            if excluded[w] or ub[w] + max(1e-6, 1e-7 * before[k]) < before[k]:
                raise AssertionError(f'Exact FPS winner excluded: {name}/{grid_name}, k={k}, method={method}')
            # Additional validation: at k in a small set, bound >= actual max
            # nearest-selected distance for *all* points in every region.
            if k in verify_states:
                by_region = np.zeros(len(reg['counts']), dtype=np.float64)
                np.maximum.at(by_region, reg['member'], verify_states[k])
                if np.any(ub + 1e-6 < by_region):
                    bad = int(np.flatnonzero(ub + 1e-6 < by_region)[0])
                    raise AssertionError(f'Bound violated by actual point: {name}/{grid_name} k={k} method={method} region={bad}')
                valid_checks += 1
            records.append({
                'origin': name, 'grid': grid_name, 'k_selected': k,
                'method': method,
                'pruned_points': int(reg['counts'][excluded].sum()),
                'pruned_point_pct': round(100. * reg['counts'][excluded].sum() / len(points), 4),
                'pruned_regions': int(excluded.sum()),
                'occupied_regions': len(reg['counts']),
                'lower_bound_m': round(lb, 6),
                'winner_distance_m': round(float(before[k]), 6),
                'exact_winner_retained': True,
            })
    return records, valid_checks


def save_csv(file, rows):
    with open(file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(records, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, org in zip(axes.flat[:3], ['first', 'centroid', 'sensor']):
        for method in ['xyz', 'spherical_cos', 'combined']:
            series = [r for r in records if r['origin'] == org and r['grid'] == 'shells_24x8x6' and r['method'] == method]
            ax.plot([x['k_selected'] for x in series], [x['pruned_point_pct'] for x in series], label=method)
        ax.set(title=f'{org} origin, angular sectors + radial shells', xlabel='Previously selected FPS points', ylabel='Safely pruned points (%)', ylim=(0, 100))
        ax.grid(alpha=.2)
        ax.legend()
    ax = axes.flat[3]
    names = ['first','centroid','sensor']
    xs = np.arange(len(names))
    for i, method in enumerate(['xyz','spherical_cos','combined']):
        avgs = [np.mean([r['pruned_point_pct'] for r in records if r['origin'] == org and r['grid'] == 'shells_24x8x6' and r['method'] == method]) for org in names]
        ax.bar(xs+(i-1)*.25, avgs, width=.25, label=method)
    ax.set(title='Mean pruning across all 127 iterations (shells)', ylabel='Safely pruned points (%)', xticks=xs, xticklabels=names)
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    fig.suptitle('Spherical cosine bounds versus XYZ enclosing-sphere bounds (not runtimes)')
    fig.savefig(output/'pruning_comparison.png', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[2]
    ap.add_argument('--kitti', type=Path, default=project/'datasets/kitti/0000000000.bin')
    ap.add_argument('--out', type=Path, default=project/'spherical_fps/geometric_bounds/results/cosine_bounds')
    ap.add_argument('--n', type=int, default=100000)
    ap.add_argument('--m', type=int, default=128)
    a = ap.parse_args()
    if not a.kitti.is_file():
        raise FileNotFoundError(f'KITTI file not found: {a.kitti}')
    a.out.mkdir(parents=True, exist_ok=True)
    print(f'KITTI input: {a.kitti}\nOutputs: {a.out}', flush=True)
    t0 = time.perf_counter()
    points = load_kitti(a.kitti, a.n)
    selected, before = exact_fps(points, a.m, 0)
    print(f'Exact FPS first-10: {selected[:10].tolist()}\nFPS reference time on host only: {time.perf_counter()-t0:.3f}s', flush=True)
    if a.n == 100000 and a.m >= 10 and a.kitti.name == '0000000000.bin':
        assert selected[:10].tolist() == [0,3052,20530,853,29561,30686,7897,14320,30792,6898], 'KITTI reference first-10 differ'

    # Prepare point-wise exact closest-selected distance at verification checkpoints.
    verify_k = sorted(set(x for x in (1, 16, 64, a.m-1) if 1 <= x < a.m))
    exact_d2 = np.full(len(points), np.inf)
    exact_states = {}
    for k in range(1, a.m):
        delta = points - points[selected[k-1]]
        np.minimum(exact_d2, np.einsum('ij,ij->i',delta,delta), out=exact_d2)
        if k in verify_k:
            exact_states[k] = np.sqrt(exact_d2.copy())

    origins = {'first': points[selected[0]], 'centroid': points.mean(axis=0), 'sensor': np.zeros(3)}
    configs = [('angular_12x6',12,6,1),('angular_24x8',24,8,1),('shells_24x8x6',24,8,6)]
    records=[]
    checks=0
    for org, origin in origins.items():
        for grid, az, el, rad in configs:
            start=time.perf_counter()
            reg=build_regions(points,origin,az,el,rad)
            rr, ok=all_queries(points,selected,before,reg,origin,org,grid,exact_states)
            checks+=ok
            records.extend(rr)
            print(f'  {org:8s} {grid:16s} occupied={len(reg["counts"]):3d} analysis={time.perf_counter()-start:.2f}s',flush=True)
    save_csv(a.out/'per_iteration_bounds.csv',records)
    summaries=[]
    print(f'\nAVERAGE PERCENT OF POINTS SAFELY PRUNED ({a.m-1} iterations)')
    print('origin    grid                  XYZ        SPHERICAL     COMBINED   uplift (combined - XYZ)')
    for org in origins:
        for grid, *_ in configs:
            vals={meth:np.mean([r['pruned_point_pct'] for r in records if r['origin']==org and r['grid']==grid and r['method']==meth]) for meth in ['xyz','spherical_cos','combined']}
            print(f'{org:9s} {grid:18s} {vals["xyz"]:8.2f}%   {vals["spherical_cos"]:8.2f}%   {vals["combined"]:8.2f}%   {vals["combined"]-vals["xyz"]:+8.2f} pp')
            for stage, (lo,hi) in {'early':(1,16),'middle':(17,64),'late':(65,a.m-1)}.items():
                if lo <= hi and lo < a.m:
                    stage_vals={meth:np.mean([r['pruned_point_pct'] for r in records if r['origin']==org and r['grid']==grid and r['method']==meth and lo<=r['k_selected']<=hi]) for meth in vals}
                    summaries.append({'origin':org,'grid':grid,'stage':stage,'start_k':lo,'end_k':min(hi,a.m-1),**{f'{m}_avg_pct':round(v,4) for m,v in stage_vals.items()}})
    save_csv(a.out/'pruning_by_stage.csv',summaries)
    plot(records,a.out)
    print(f'\nPASS: all {a.m-1} exact winners retained for all configurations; {checks} bound-vs-all-points validation checks passed.')
    print('Note: geometric pruning fraction, NOT distance-count or runtime speedup.')
    print(f'Files: {a.out / "per_iteration_bounds.csv"}, {a.out / "pruning_by_stage.csv"}, {a.out / "pruning_comparison.png"}',flush=True)


if __name__=='__main__':
    main()
