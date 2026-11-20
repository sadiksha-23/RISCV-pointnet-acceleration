#!/usr/bin/env python3
"""KITTI experiment 1: exact FPS geometry, spacing and selection history.

Requires numpy, matplotlib. Runs on the first N KITTI XYZI records; computes
true 3D Euclidean FPS distances without using 2D projections for decisions.
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128)
KNOWN_FIRST10 = [0, 3052, 20530, 853, 29561, 30686, 7897, 14320, 30792, 6898]


def exact_fps(points: np.ndarray, m: int):
    """Start from point zero, lowest index wins ties; nearest-distance tracked exactly."""
    if not 1 <= m <= len(points):
        raise ValueError('Require 1 <= M <= N')
    picked = np.empty(m, dtype=np.int64)
    picked[0] = 0
    selected_dist = np.full(m, np.nan, dtype=np.float64)
    cover = np.empty(m, dtype=np.float64)
    closest_sq = np.empty(len(points), dtype=np.float64)
    diff = points - points[0]
    np.einsum('ij,ij->i', diff, diff, out=closest_sq)
    closest_sq[0] = 0.
    cover[0] = np.sqrt(closest_sq.max())
    for k in range(1, m):
        idx = int(np.argmax(closest_sq))
        picked[k] = idx
        selected_dist[k] = np.sqrt(closest_sq[idx])
        diff = points - points[idx]
        new_sq = np.einsum('ij,ij->i', diff, diff)
        np.minimum(closest_sq, new_sq, out=closest_sq)
        closest_sq[idx] = 0.
        cover[k] = np.sqrt(closest_sq.max())
    return picked, selected_dist, cover


def selected_spacing(points: np.ndarray):
    """Nearest neighbor IN SELECTED SET using real 3D, not 2D projected spacing."""
    if len(points) < 2:
        return np.full(len(points), np.nan)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def stats(vals):
    if len(vals) < 2:
        return dict(mean=float('nan'), std=float('nan'), cv=float('nan'), min=float('nan'), max=float('nan'))
    avg = float(vals.mean())
    sd = float(vals.std())
    return dict(mean=avg, std=sd, cv=sd/avg if avg > 0 else float('nan'), min=float(vals.min()), max=float(vals.max()))


def save_table(path: Path, names, rows):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(rows)


def draw_projections(points, picked, background, output, title):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (a, b), label in zip(axs, [(0, 1), (0, 2), (1, 2)], ['XY', 'XZ', 'YZ']):
        ax.scatter(background[:, a], background[:, b], s=1, alpha=.17, color='#6f9cba', rasterized=True)
        h = ax.scatter(points[picked, a], points[picked, b], c=np.arange(1, len(picked)+1),
                       cmap='viridis', s=38, edgecolors='black', linewidths=.35, zorder=4)
        for order, idx in enumerate(picked[:10], 1):
            ax.annotate(str(order), (points[idx, a], points[idx, b]), xytext=(4, 3),
                        textcoords='offset points', fontsize=8)
        ax.set_title(label + ' projection (for display only)')
        ax.set_xlabel('XYZ'[a] + ' (meters)')
        ax.set_ylabel('XYZ'[b] + ' (meters)')
        ax.set_aspect('equal', adjustable='box')
    fig.colorbar(h, ax=axs, shrink=.74, label='FPS selection order')
    fig.suptitle(title + '\nFPS computations use full 3D distances', y=1.04)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bin', dest='bin_path', type=Path,
                   help='KITTI velodyne .bin; default: root/datasets/kitti/0000000000.bin')
    p.add_argument('--n', type=int, default=100000)
    p.add_argument('--m', type=int, default=128)
    p.add_argument('--out', type=Path, help='Results folder (defaults to spherical_fps/selection_patterns/results/kitti_N..._M...)')
    p.add_argument('--seed', type=int, default=42, help='Only used for plot background sampling')
    p.add_argument('--max-background', type=int, default=14000, help='Background dots per plot')
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_file = args.bin_path if args.bin_path else root / 'datasets/kitti/0000000000.bin'
    output = args.out if args.out else root / 'spherical_fps/selection_patterns/results' / f'kitti_{args.n}_{args.m}'
    data_file = data_file.resolve()
    output = output.resolve()
    if not data_file.is_file():
        raise SystemExit(f'KITTI frame not found: {data_file}\nSupply --bin /path/to/0000000000.bin')
    if data_file.stat().st_size % 16:
        raise SystemExit('Invalid KITTI .bin size (must contain float32 XYZI records, 16 bytes each)')
    frame = np.fromfile(data_file, dtype='<f4').reshape(-1, 4)
    if args.n > len(frame) or args.n < 2 or not 2 <= args.m <= args.n:
        raise SystemExit(f'Requested N={args.n} M={args.m}, but file contains {len(frame)} records; require 2 <= M <= N <= records')
    points = np.array(frame[:args.n, :3], dtype=np.float64, copy=True)
    if not np.isfinite(points).all():
        raise SystemExit('Point cloud has non-finite XYZ coordinates')
    output.mkdir(parents=True, exist_ok=True)
    print(f'KITTI_FILE={data_file}')
    print(f'FRAME_TOTAL_POINTS={len(frame)}')
    print(f'USED_POINTS={args.n} M={args.m} FIRST_POINT_INDEX=0 TIE_BREAK=LOWEST_INDEX')
    print(f'OUTPUT_DIR={output}')
    print('FPS_DISTANCE=EXACT_3D_EUCLIDEAN (float64, no approximations)')

    t0 = time.perf_counter()
    picked, arrival_gap, cover = exact_fps(points, args.m)
    print(f'FPS_PYTHON_RUNTIME_MS={(time.perf_counter()-t0)*1000:.3f} (NOT gem5 and not an optimized algorithm benchmark)')
    assert len(set(map(int, picked))) == args.m
    assert np.all(np.diff(cover) <= 1e-10)
    assert np.allclose(arrival_gap[1:], cover[:-1], rtol=1e-10, atol=1e-8)
    print('PASS_EXACT_FPS_INVARIANTS=True')
    if args.n == 100000 and args.m >= 10:
        if np.array_equal(picked[:10], KNOWN_FIRST10):
            print('REFERENCE_FIRST10_MATCH=True (matches our earlier KITTI run)')
        else:
            print('REFERENCE_FIRST10_MATCH=False (check frame/order/float precision against earlier run)')
            print(f'EXPECTED_FIRST10={KNOWN_FIRST10}')
    print('FIRST10=', picked[:10].tolist())
    print('LAST5=', picked[-5:].tolist())

    chosen = points[picked]
    last_nn = selected_spacing(chosen)
    centroid = points.mean(axis=0)
    radius_centroid = np.linalg.norm(points-centroid, axis=1)
    sorted_radius = np.sort(radius_centroid)
    selected_radius_centroid = radius_centroid[picked]
    centroid_percentile = np.searchsorted(sorted_radius, selected_radius_centroid, side='right') * 100.0 / len(points)

    checkpoint_rows = []
    print('\n=== REAL 3D SPACING AT SELECTION CHECKPOINTS ===')
    print('  K    Cover(m)    NN_mean(m)   NN_std(m)    CV(%)    NN_min(m)   NN_max(m)')
    for k in sorted(set([v for v in CHECKPOINTS if v <= args.m] + [args.m])):
        vals = selected_spacing(chosen[:k])
        s = stats(vals)
        checkpoint_rows.append([k, cover[k-1], s['mean'], s['std'], s['cv'], s['min'], s['max']])
        print(f"{k:3d}  {cover[k-1]:10.4f}   {s['mean']:10.4f}   {s['std']:9.4f}   {100*s['cv']:7.2f}   {s['min']:10.4f}   {s['max']:10.4f}")
    save_table(output / 'checkpoints_3d_spacing.csv',
               ['num_selected', 'cover_radius_m', 'nearest_neighbor_mean_m', 'nearest_neighbor_std_m',
                'nearest_neighbor_cv', 'nearest_neighbor_min_m', 'nearest_neighbor_max_m'], checkpoint_rows)
    relative = chosen - chosen[0]
    r = np.linalg.norm(relative, axis=1)
    azimuth = np.degrees(np.arctan2(relative[:, 1], relative[:, 0]))
    elevation = np.degrees(np.arctan2(relative[:, 2], np.hypot(relative[:, 0], relative[:, 1])))
    save_table(output / 'selected_points.csv',
               ['selection_order', 'original_index', 'x_m', 'y_m', 'z_m',
                'nearest_before_selection_m', 'nearest_after_all_selections_m',
                'radius_from_first_m', 'azimuth_from_first_deg', 'elevation_from_first_deg',
                'radius_from_centroid_m', 'radial_percentile_from_centroid'],
               [[i+1, int(idx), *chosen[i], arrival_gap[i], last_nn[i], r[i], azimuth[i], elevation[i],
                 selected_radius_centroid[i], centroid_percentile[i]] for i, idx in enumerate(picked)])
    save_table(output / 'coverage_curve.csv', ['num_selected', 'coverage_radius_m'],
               [(i+1, cover[i]) for i in range(args.m)])

    rng = np.random.default_rng(args.seed)
    subset = np.sort(rng.choice(len(points), size=min(args.max_background, len(points)), replace=False))
    background = points[subset]
    draw_projections(points, picked, background, output/'01_selected_projections.png',
                     f'KITTI exact FPS: N={args.n:,} M={args.m}')

    # Progressive spatial occupancy, same camera/projection scales for each subplot.
    ks = sorted(set([v for v in (1, 2, 4, 8, 16, 32, 64, 128) if v <= args.m] + [args.m]))
    fig, axes = plt.subplots((len(ks)+3)//4, 4, figsize=(18, 3.9*((len(ks)+3)//4)), squeeze=False)
    xlim = np.quantile(points[:, 0], [.001, .999])
    ylim = np.quantile(points[:, 1], [.001, .999])
    for ax, k in zip(axes.flat, ks):
        ax.scatter(background[:, 0], background[:, 1], s=.7, alpha=.1, rasterized=True)
        ax.scatter(chosen[:k, 0], chosen[:k, 1], s=18, c=np.arange(k), cmap='viridis',
                   edgecolors='black', linewidths=.25)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'M={k}, cover={cover[k-1]:.2f}m', fontsize=10)
    for ax in list(axes.flat)[len(ks):]:
        ax.axis('off')
    fig.suptitle('KITTI exact FPS selection progression (XY views; 3D FPS distances)', fontsize=15)
    fig.tight_layout()
    fig.savefig(output/'02_selection_progression_xy.png', dpi=140)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].plot(np.arange(1, args.m+1), cover, linewidth=1.8)
    axs[0].set(title='Maximum distance to selected set (coverage radius)', xlabel='Number selected', ylabel='Coverage radius (m)')
    axs[1].plot(np.arange(2, args.m+1), arrival_gap[1:], linewidth=1.8)
    axs[1].set(title='Distance when each point was selected', xlabel='Selection order', ylabel='Nearest previous selected distance (m)')
    for ax in axs: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'03_coverage_and_selection_distance.png', dpi=155)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axs[0].hist(last_nn, bins=min(18, max(5, args.m//5)), edgecolor='black', linewidth=.5)
    axs[0].set(title='Final 3D nearest-selected-neighbor spacing', xlabel='Nearest neighbor distance (m)', ylabel='Selected point count')
    axs[1].plot([r[0] for r in checkpoint_rows if r[0] >= 2],
                [100*r[4] for r in checkpoint_rows if r[0] >= 2], 'o-')
    axs[1].set(title='How final selected-set spacing varies with M', xlabel='Number selected', ylabel='Nearest neighbor CV (%)')
    for ax in axs: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'04_spacing_histogram_and_cv.png', dpi=155)
    plt.close(fig)

    # These are spherical coordinates about FIRST FPS point, not sensor-origin coordinates.
    rel_all = background-chosen[0]
    all_az = np.degrees(np.arctan2(rel_all[:, 1], rel_all[:, 0]))
    all_rad = np.linalg.norm(rel_all, axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].scatter(all_az, all_rad, s=1, alpha=.08, color='steelblue', rasterized=True)
    h = axs[0].scatter(azimuth, r, c=np.arange(1, args.m+1), cmap='viridis', s=27, edgecolors='black', linewidths=.2)
    axs[0].set(title='Spherical: azimuth vs radius', xlabel='Azimuth (degrees)', ylabel='Radius from first selected point (m)')
    axs[1].scatter(azimuth, elevation, c=np.arange(1, args.m+1), cmap='viridis', s=27, edgecolors='black', linewidths=.2)
    axs[1].set(title='Selected-point directions', xlabel='Azimuth (degrees)', ylabel='Elevation (degrees)')
    fig.colorbar(h, ax=axs, shrink=.8, label='FPS selection order')
    fig.suptitle('KITTI representation centered on FIRST selected point (not original LiDAR sensor)')
    fig.savefig(output/'05_spherical_from_first.png', dpi=155, bbox_inches='tight')
    plt.close(fig)

    print('\n=== EARLY POINT RADIAL POSITION (FROM CENTROID, NOT CONVEX HULL) ===')
    print('Order  Index   Radial_percentile')
    for i in range(min(10, args.m)):
        print(f'{i+1:5d} {int(picked[i]):6d}  {centroid_percentile[i]:7.2f}%')
    print('\n=== INTERPRETATION ===')
    print('CV measures how similar the FINAL nearest-neighbor distances are in 3D.')
    print('Lower CV means more similar spacing; there is no universal threshold proving uniformity.')
    print('FPS guarantees nonincreasing coverage; this alone is not evidence of a new pattern.')
    print('Centroid radial percentile is only an outerness proxy, NOT a convex-hull/boundary test.')
    print('All 5 PNG plots and 3 CSV tables written to:', output)

if __name__ == '__main__':
    main()
