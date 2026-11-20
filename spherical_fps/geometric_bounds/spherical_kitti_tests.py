#!/usr/bin/env python3
"""Exploratory spherical FPS experiments (not an accelerated FPS implementation).

1. Radial selections by origin.
2. Spherical angular sector predictions, origin comparison.
3. Exact safe spherical-region pruning using a bounding-sphere certificate.

Uses Euclidean distances in original XYZ throughout; coordinate transforms never
change the FPS reference. All reports are descriptive, not runtime speedups.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PI = math.pi


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_kitti(path, n):
    raw = np.fromfile(path, dtype='<f4')
    if raw.size % 4:
        raise ValueError(f'KITTI binary should contain XYZI float32 values; got {raw.size} floats')
    if raw.size // 4 < n:
        raise ValueError(f'KITTI file contains only {raw.size // 4} points; requested {n}')
    return raw.reshape(-1, 4)[:n, :3].astype(np.float64, copy=True)


def exact_fps(points, m, seed):
    """Greedy farthest-first in float64 XYZ with deterministic lowest-index argmax."""
    n = len(points)
    if not 0 <= seed < n or not 1 <= m <= n:
        raise ValueError('Invalid seed or m')
    selected = np.empty(m, dtype=np.int64)
    before = np.full(m, np.nan, dtype=np.float64)
    nearest_sq = np.full(n, np.inf, dtype=np.float64)
    current = seed
    for k in range(m):
        selected[k] = current
        d = points - points[current]
        d2 = np.einsum('ij,ij->i', d, d)
        np.minimum(nearest_sq, d2, out=nearest_sq)
        if k + 1 < m:
            current = int(np.argmax(nearest_sq))
            before[k + 1] = float(np.sqrt(nearest_sq[current]))
    return selected, before


def spherical(points, origin):
    relative = points - origin
    radii = np.linalg.norm(relative, axis=1)
    azimuth = np.arctan2(relative[:, 1], relative[:, 0])
    elevation = np.arctan2(relative[:, 2], np.hypot(relative[:, 0], relative[:, 1]))
    return radii, azimuth, elevation


def build_regions(points, origin, az_bins, el_bins, radial_bins):
    """Spherical bins; each occupied bin enclosed by a real XYZ bounding sphere.

    R = max distance from the *actual point cloud* in the cell to the cell
    centroid, therefore ||p-s|| <= ||center-s||+R for every member p.
    """
    start = time.perf_counter()
    r, az, el = spherical(points, origin)
    ai = np.minimum((az + PI) / (2 * PI) * az_bins, az_bins - 1).astype(np.int64)
    ei = np.minimum((el + PI / 2) / PI * el_bins, el_bins - 1).astype(np.int64)
    if radial_bins == 1:
        ri = np.zeros(len(points), dtype=np.int64)
    else:
        maximum = float(max(np.max(r), 1e-12))
        ri = np.minimum(r / maximum * radial_bins, radial_bins - 1).astype(np.int64)
    labels = (ri * el_bins + ei) * az_bins + ai
    unique, first, member = np.unique(labels, return_index=True, return_inverse=True)
    number = len(unique)
    counts = np.bincount(member, minlength=number)
    center = np.stack([np.bincount(member, weights=points[:, axis], minlength=number) / counts
                       for axis in range(3)], axis=1)
    radii = np.zeros(number, dtype=np.float64)
    distances = np.linalg.norm(points - center[member], axis=1)
    np.maximum.at(radii, member, distances)
    # Inflate slightly to guard float rounding around tight inequalities.
    radii = np.nextafter(radii + 1e-10, np.inf)
    return {'member': member, 'counts': counts, 'center': center, 'radius': radii,
            'representative': points[first], 'label': unique,
            'setup_seconds': time.perf_counter() - start,
            'r': r, 'az': az, 'el': el}


def evaluate_at_checkpoint(points, selected, before, regions, k, origin_name, grid_name):
    """Ask where p[k] will be after selections p[0:k]. No full-N search in predictor."""
    t0 = time.perf_counter()
    chosen = points[selected[:k]]
    center = regions['center']
    reps = regions['representative']
    # For each region, upper bound the nearest-selected distance of ALL members.
    # min_s ||p-s|| <= min_s (||center-s|| + R).
    dist_center = np.linalg.norm(center[:, None, :] - chosen[None, :, :], axis=2)
    ub = dist_center.min(axis=1) + regions['radius']
    # The nearest distance of one existing point per occupied region is a
    # rigorous global lower bound on the best candidate distance.
    dist_rep = np.linalg.norm(reps[:, None, :] - chosen[None, :, :], axis=2)
    lb = float(dist_rep.min(axis=1).max())
    # Strict conservative pruning: only cells certainly below an attainable LB.
    safe_epsilon = 1e-8 * max(1., float(ub.max()), lb)
    pruned = ub < lb - safe_epsilon
    winner_group = int(regions['member'][selected[k]])
    winner_radius = float(before[k])
    if pruned[winner_group] or ub[winner_group] + safe_epsilon < winner_radius:
        raise AssertionError('INVALID BOUND: exact FPS winner would have been excluded')
    order = np.argsort(-ub, kind='stable')
    rank = int(np.where(order == winner_group)[0][0]) + 1
    total = len(points)
    counts = regions['counts']
    return {
        'origin': origin_name, 'grid': grid_name, 'selected_so_far': k,
        'next_selection_order': k + 1, 'next_original_index': int(selected[k]),
        'occupied_regions': len(counts),
        'winner_region_rank_by_UB': rank,
        'top1_region_contains_winner': int(rank == 1),
        'top5_regions_contain_winner': int(rank <= 5),
        'certified_pruned_regions': int(pruned.sum()),
        'certified_pruned_region_pct': round(float(100 * pruned.mean()), 3),
        'certified_pruned_points': int(counts[pruned].sum()),
        'certified_pruned_point_pct': round(float(100 * counts[pruned].sum() / total), 3),
        'representative_lower_bound_m': round(lb, 6),
        'exact_winner_distance_m': round(winner_radius, 6),
        'winner_region_upper_bound_m': round(float(ub[winner_group]), 6),
        'query_seconds': round(time.perf_counter() - t0, 6),
        'winner_safely_retained': 1,
    }


def plot_radial(points, selected, origins, out):
    fig, axes = plt.subplots(1, len(origins), figsize=(5.3 * len(origins), 4.5), constrained_layout=True)
    for ax, (name, origin) in zip(np.atleast_1d(axes), origins.items()):
        rad, _, _ = spherical(points, origin)
        order = np.arange(1, len(selected) + 1)
        ax.plot(order, rad[selected], '.', alpha=.7, markersize=5)
        ax.plot(order, np.maximum.accumulate(rad[selected]), '-', alpha=.55,
                label='Largest radius selected so far')
        ax.set_title(f'{name} as spherical origin')
        ax.set_xlabel('FPS selection order')
        ax.set_ylabel('Radial distance from origin (m)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Test 1: Radial patterns of exact FPS selections')
    fig.savefig(out / '01_radial_selection_vs_order.png', dpi=160)
    plt.close(fig)


def plot_angles(points, selected, origins, out):
    fig, axes = plt.subplots(1, len(origins), figsize=(5.4 * len(origins), 4.8), constrained_layout=True)
    for ax, (name, origin) in zip(np.atleast_1d(axes), origins.items()):
        r, az, el = spherical(points, origin)
        background = np.arange(0, len(points), max(1, len(points) // 8000))
        ax.scatter(np.degrees(az[background]), np.degrees(el[background]), s=1, alpha=.07, color='grey')
        sc = ax.scatter(np.degrees(az[selected]), np.degrees(el[selected]), s=25,
                        c=np.arange(1, len(selected)+1), cmap='viridis', edgecolors='none')
        ax.set_title(f'Origin: {name}')
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(alpha=.15)
        fig.colorbar(sc, ax=ax, label='FPS selection order', shrink=.75)
    fig.suptitle('Test 2: Same exact FPS indices; spherical coordinates change with origin')
    fig.savefig(out / '02_angular_selected_by_origin.png', dpi=160)
    plt.close(fig)


def plot_pruning(rows, out):
    if not rows:
        return
    names = list(dict.fromkeys((r['origin'], r['grid']) for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for o, g in names:
        data = sorted((r for r in rows if r['origin'] == o and r['grid'] == g), key=lambda r: r['selected_so_far'])
        x = [r['selected_so_far'] for r in data]
        axes[0].plot(x, [r['certified_pruned_point_pct'] for r in data], marker='o', label=f'{o}: {g}')
        axes[1].plot(x, [r['winner_region_rank_by_UB'] for r in data], marker='o', label=f'{o}: {g}')
    axes[0].set_title('Test 3: Points in provably excluded spherical bins')
    axes[0].set_ylabel('Percentage of original points (%)')
    axes[1].set_title('Test 2: Rank of true next-point region by bound score')
    axes[1].set_ylabel('True next region rank (1 = highest score)')
    for ax in axes:
        ax.set_xlabel('Number of FPS points already selected')
        ax.grid(alpha=.2)
        ax.set_xticks(sorted(set(r['selected_so_far'] for r in rows)))
    axes[0].legend(fontsize=7, ncol=2, loc='upper left')
    axes[1].legend(fontsize=7, ncol=2, loc='upper right')
    fig.savefig(out / '03_region_bounds_and_prediction.png', dpi=160)
    plt.close(fig)


def parse_args():
    project = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description='Exact FPS + spherical region research tests on KITTI')
    p.add_argument('--kitti', type=Path, default=project / 'datasets/kitti/0000000000.bin')
    p.add_argument('--out', type=Path, default=project / 'spherical_fps/geometric_bounds/results/region_prediction_127_checkpoints')
    p.add_argument('--n', type=int, default=100000)
    p.add_argument('--m', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--origins', nargs='+', choices=['first', 'centroid', 'sensor', 'bbox'],
                   default=['first', 'centroid', 'sensor'])
    p.add_argument('--checkpoints', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64, 127])
    p.add_argument('--no-plots', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    if not a.kitti.is_file():
        raise FileNotFoundError(f'KITTI data not found at {a.kitti}. Use --kitti /path/to/your/file.bin')
    a.out.mkdir(parents=True, exist_ok=True)
    print(f'KITTI: {a.kitti}\nOutput: {a.out}', flush=True)
    points = load_kitti(a.kitti, a.n)
    start = time.perf_counter()
    selected, before = exact_fps(points, a.m, a.seed)
    baseline = time.perf_counter() - start
    print(f'Exact FPS float64 baseline: N={len(points)} M={a.m}, {baseline:.3f}s (host only)')
    print('First 10 indices:', selected[:10].tolist())
    known = [0, 3052, 20530, 853, 29561, 30686, 7897, 14320, 30792, 6898]
    if a.n == 100000 and a.seed == 0 and a.m >= 10 and a.kitti.name == '0000000000.bin':
        print('Previously recorded KITTI first-10:', 'MATCH' if selected[:10].tolist() == known else 'DIFFER (investigate float precision / tie policy / dataset)')
    origins_all = {'first': points[selected[0]], 'centroid': points.mean(axis=0),
                   'sensor': np.zeros(3), 'bbox': (points.min(axis=0) + points.max(axis=0)) / 2}
    origins = {o: origins_all[o] for o in a.origins}
    checkpoints = sorted(set(k for k in a.checkpoints if 1 <= k < a.m))
    if not checkpoints:
        raise ValueError('Need a checkpoint between 1 and M-1')
    print('Origins:', ', '.join(origins), '| checkpoints:', checkpoints)
    configs = [('angular_12x6', 12, 6, 1),
               ('angular_24x8', 24, 8, 1),
               ('shells_24x8x6', 24, 8, 6)]
    polar_rows = []
    for origin_name, origin in origins.items():
        rad, az, el = spherical(points, origin)
        for j, idx in enumerate(selected):
            polar_rows.append({'origin': origin_name, 'selection_order': j+1, 'original_index': int(idx),
                               'radius_m': float(rad[idx]), 'azimuth_deg': float(np.degrees(az[idx])),
                               'elevation_deg': float(np.degrees(el[idx])),
                               'nearest_before_selection_m': float(before[j])})
    write_csv(a.out / '01_selected_spherical_by_origin.csv', polar_rows)

    rows = []
    setup_rows = []
    for origin_name, origin in origins.items():
        for name, a_bins, e_bins, r_bins in configs:
            reg = build_regions(points, origin, a_bins, e_bins, r_bins)
            print(f'  {origin_name:8s} {name:16s} occupied={len(reg["counts"]):4d}, setup={reg["setup_seconds"]:.3f}s', flush=True)
            setup_rows.append({'origin': origin_name, 'grid': name, 'azimuth_bins': a_bins,
                              'elevation_bins': e_bins, 'radial_shells': r_bins,
                              'occupied_regions': len(reg['counts']),
                              'max_points_in_region': int(np.max(reg['counts'])),
                              'median_points_in_region': float(np.median(reg['counts'])),
                              'mean_bounding_radius_m': float(np.mean(reg['radius'])),
                              'setup_seconds_host': round(reg['setup_seconds'], 6)})
            for k in checkpoints:
                rows.append(evaluate_at_checkpoint(points, selected, before, reg, k, origin_name, name))
    write_csv(a.out / '02_region_preparation.csv', setup_rows)
    write_csv(a.out / '03_region_prediction_and_proven_pruning.csv', rows)
    if not a.no_plots:
        plot_radial(points, selected, origins, a.out)
        plot_angles(points, selected, origins, a.out)
        plot_pruning(rows, a.out)
    print('\nSUMMARY (averages across requested checkpoints; only within this experiment)')
    print(f'{"origin":9s} {"grid":17s} {"top1 %":>8s} {"top5 %":>8s} {"safe prune points %":>20s}')
    for origin_name in origins:
        for name, *_ in configs:
            rr = [r for r in rows if r['origin'] == origin_name and r['grid'] == name]
            print(f'{origin_name:9s} {name:17s} {100*np.mean([r["top1_region_contains_winner"] for r in rr]):8.1f} '
                  f'{100*np.mean([r["top5_regions_contain_winner"] for r in rr]):8.1f} '
                  f'{np.mean([r["certified_pruned_point_pct"] for r in rr]):20.2f}')
    print('\nALL EXACT WINNERS RETAINED BY CERTIFICATES: PASS')
    print('NOTE: top-K uses optimistic region UPPER BOUNDS; it is a heuristic ranking, NOT a proof.')
    print('NOTE: pruning uses a provable bound with one sampled point per occupied region; setup and query are NOT optimized.')
    print('Outputs:', a.out)


if __name__ == '__main__':
    main()
