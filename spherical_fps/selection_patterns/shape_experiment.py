#!/usr/bin/env python3
"""Experiment 1: geometric structure of exact greedy farthest-point sampling.

Synthetic point clouds only. No gem5 timings, approximation, pruning, or RVV.
Usage: python3 polar_experiment1.py --n 1000 --m 32 --seed 42
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots when run from WSL, without needing a GUI.
import matplotlib.pyplot as plt
import numpy as np


def make_shapes(n, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    shapes = {}

    # Ordered circle and line deliberately demonstrate the geometric pattern clearly.
    shapes["01_circle_outline"] = np.column_stack((np.cos(t), np.sin(t)))
    shapes["02_line"] = np.column_stack((np.linspace(-1, 1, n), np.zeros(n)))

    radius = np.sqrt(rng.uniform(0, 1, n))  # Uniform point density over disk area.
    angle = rng.uniform(-np.pi, np.pi, n)
    shapes["03_filled_disk"] = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    shapes["04_filled_square"] = rng.uniform(-1, 1, size=(n, 2))

    # Three rings, equal numbers of points per ring (up to rounding).
    counts = [n // 3, n // 3, n - 2 * (n // 3)]
    rings = []
    for ring_radius, count in zip((0.35, 0.65, 1.0), counts):
        ang = np.linspace(0, 2 * np.pi, count, endpoint=False)
        rings.append(np.column_stack((ring_radius * np.cos(ang), ring_radius * np.sin(ang))))
    shapes["05_concentric_rings"] = np.concatenate(rings, axis=0)

    n_left = n // 2
    n_right = n - n_left
    left = rng.normal(loc=(-0.80, 0.0), scale=(0.18, 0.20), size=(n_left, 2))
    right = rng.normal(loc=(0.80, 0.0), scale=(0.18, 0.20), size=(n_right, 2))
    shapes["06_two_clusters"] = np.concatenate((left, right), axis=0)

    # Uniform directions on unit sphere surface, and uniform points in unit ball.
    z = rng.uniform(-1, 1, n)
    a = rng.uniform(-np.pi, np.pi, n)
    xy = np.sqrt(1 - z * z)
    directions = np.column_stack((xy * np.cos(a), xy * np.sin(a), z))
    shapes["07_sphere_surface_3d"] = directions
    shapes["08_filled_ball_3d"] = directions * np.cbrt(rng.uniform(0, 1, n))[:, None]

    left_3d = rng.normal(loc=(-0.9, 0, 0), scale=(0.17, 0.19, 0.21), size=(n_left, 3))
    right_3d = rng.normal(loc=(0.9, 0, 0), scale=(0.17, 0.19, 0.21), size=(n_right, 3))
    shapes["09_two_clusters_3d"] = np.concatenate((left_3d, right_3d), axis=0)
    return shapes


def exact_fps(points, m):
    """Global greedy FPS; initial index 0; np.argmax chooses lowest index on ties.

    Returns indices, coverage radius after each addition, and the nearest distance
    of each point *at the moment it was selected*.
    """
    if m < 1 or m > len(points):
        raise ValueError("m must satisfy 1 <= m <= n")
    samples = [0]
    delta = points - points[0]
    nearest_sq = np.einsum("ij,ij->i", delta, delta)
    cover_radius = [float(np.sqrt(np.max(nearest_sq)))]
    selection_gap = [float("nan")]
    for _ in range(1, m):
        idx = int(np.argmax(nearest_sq))
        selection_gap.append(float(np.sqrt(nearest_sq[idx])))
        samples.append(idx)
        delta = points - points[idx]
        d_sq = np.einsum("ij,ij->i", delta, delta)
        np.minimum(nearest_sq, d_sq, out=nearest_sq)
        cover_radius.append(float(np.sqrt(np.max(nearest_sq))))
    return np.asarray(samples, dtype=int), np.asarray(cover_radius), np.asarray(selection_gap)


def plot_shape(ax, points, chosen, k, label_count=8):
    dim = points.shape[1]
    chosen = chosen[:k]
    seq = np.arange(k)
    if dim == 2:
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.13, rasterized=True)
        ax.scatter(points[chosen, 0], points[chosen, 1], c=seq, s=75, edgecolors="black", linewidths=0.55, zorder=4)
        for rank, idx in enumerate(chosen[:label_count]):
            ax.annotate(str(rank + 1), (points[idx, 0], points[idx, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, alpha=0.09, rasterized=True)
        ax.scatter(points[chosen, 0], points[chosen, 1], points[chosen, 2], c=seq, s=75, edgecolors="black", linewidths=0.4, depthshade=False)
        for rank, idx in enumerate(chosen[:min(label_count, 5)]):
            p = points[idx]
            ax.text(p[0], p[1], p[2], str(rank + 1), fontsize=8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # Equal scales so sphere isn't drawn as an ellipsoid.
        span = np.ptp(points, axis=0)
        center = np.mean([points.min(axis=0), points.max(axis=0)], axis=0)
        half = max(span.max() * 0.57, 0.01)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=21, azim=-59)


def save_figures(points, chosen, cover, key, out, m):
    dim = points.shape[1]
    if dim == 2:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        plot_shape(ax, points, chosen, m, label_count=10)
        ax.set_title(f"{key.replace('_', ' ')} | exact FPS | selected {m} of {len(points)}\nNumbers = first 10 selections; initial sample = original index 0")
    else:
        # Three pairwise projections are more readable and much faster to render than
        # depth-sorted 3D scatter. All FPS calculations still used full 3D distances.
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
        for ax, dims, title in zip(axs, [(0, 1), (0, 2), (1, 2)], ["XY", "XZ", "YZ"]):
            plot_shape(ax, points[:, dims], chosen, m, label_count=7)
            ax.set_xlabel(title[0])
            ax.set_ylabel(title[1])
            ax.set_title(f"{title} projection")
        fig.suptitle(f"{key.replace('_', ' ')} | 3D exact FPS | selected {m} of {len(points)}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / f"{key}_selected.png", dpi=145)
    plt.close(fig)

    ks = sorted(set(min(m, k) for k in (1, 2, 4, 8, 16, 32, m)))
    cols = 3
    rows = (len(ks) + cols - 1) // cols
    fig = plt.figure(figsize=(14, 4.5 * rows))
    for panel, k in enumerate(ks, start=1):
        # For 3D progression use a quick XY projection; exact FPS itself is 3D.
        ax = fig.add_subplot(rows, cols, panel)
        if dim == 3:
            plot_shape(ax, points[:, :2], chosen, k, label_count=4)
            ax.set_xlabel("X (XY projection)")
            ax.set_ylabel("Y (XY projection)")
        else:
            plot_shape(ax, points, chosen, k, label_count=4)
        ax.set_title(f"After {k} selections | coverage radius = {cover[k-1]:.3f}")
    fig.suptitle(f"Selection progression: {key.replace('_', ' ')}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"{key}_progression.png", dpi=115)
    plt.close(fig)

    # A fixed-origin angle/radius plot is meaningful for 2D shapes only.
    if dim == 2:
        origin = points[0]
        relative = points - origin
        angles = np.degrees(np.arctan2(relative[:, 1], relative[:, 0]))
        radial = np.linalg.norm(relative, axis=1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(angles, radial, s=7, alpha=0.16, rasterized=True)
        ax.scatter(angles[chosen], radial[chosen], c=np.arange(m), s=65, edgecolors="black", linewidths=0.45)
        for rank, idx in enumerate(chosen[:min(10, m)]):
            ax.annotate(str(rank + 1), (angles[idx], radial[idx]), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set_xlim(-185, 185)
        ax.set_xlabel("Angle relative to FIRST selected point (degrees)")
        ax.set_ylabel("Radius relative to FIRST selected point (shape units)")
        ax.set_title(f"Polar representation of {key.replace('_', ' ')}\nSame points, different coordinates — not a sine wave")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"{key}_polar_from_first.png", dpi=180)
        plt.close(fig)


def save_selected_csv(points, indices, gaps, key, out):
    dim = points.shape[1]
    with (out / f"{key}_selected_points.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["selection_order", "original_index", "x", "y"] + (["z"] if dim == 3 else []) + ["nearest_distance_before_selection"])
        for rank, idx in enumerate(indices):
            w.writerow([rank + 1, int(idx)] + list(map(float, points[idx])) + [float(gaps[rank])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="Points per shape (default 1000)")
    parser.add_argument("--m", type=int, default=32, help="FPS samples per shape (default 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--out", default="benchmark/analysis/fps_polar_experiment1", help="Output folder relative to current working directory")
    args = parser.parse_args()
    if args.n < 32 or not 1 <= args.m <= args.n:
        parser.error("Require n >= 32 and 1 <= m <= n")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = make_shapes(args.n, args.seed)
    results = []
    coverage_by_dim = {2: [], 3: []}

    print(f"EXPERIMENT=EXACT_GREEDY_FPS_SHAPES N={args.n} M={args.m} SEED={args.seed}")
    print("START_POINT=ORIGINAL_INDEX_0")
    print("TIES=LOWEST_ORIGINAL_INDEX")
    print(f"OUTPUT_DIR={out}")

    for key, points in cases.items():
        t0 = time.perf_counter()
        indices, cover, gap = exact_fps(points, args.m)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert len(set(indices)) == args.m, f"Duplicate selected indices in {key}"
        assert np.all(cover[1:] <= cover[:-1] + 1e-10), f"Coverage unexpectedly increased in {key}"

        save_figures(points, indices, cover, key, out, args.m)
        save_selected_csv(points, indices, gap, key, out)

        # Pairwise minimum among chosen points (measure of spacing).
        selected = points[indices]
        delta = selected[:, None, :] - selected[None, :, :]
        pd2 = np.einsum("ijk,ijk->ij", delta, delta)
        np.fill_diagonal(pd2, np.inf)
        smallest_separation = float(np.sqrt(pd2.min())) if args.m > 1 else float("nan")

        results.append([key, points.shape[1], args.n, args.m, args.seed,
                        round(cover[0], 6), round(cover[-1], 6), round(smallest_separation, 6),
                        round(elapsed_ms, 4), ";".join(map(str, indices[:10]))])
        coverage_by_dim[points.shape[1]].append((key, cover))
        print(f"{key:25} dim={points.shape[1]} cover_initial={cover[0]:.4f} "
              f"cover_final={cover[-1]:.4f} fps_python_ms={elapsed_ms:.2f} "
              f"indices_first10={indices[:10].tolist()}")

    with (out / "experiment_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "dimensions", "points", "selected", "seed", "initial_cover_radius",
                    "final_cover_radius", "minimum_selected_pair_distance", "fps_python_ms_not_gem5", "first_10_original_indices"])
        w.writerows(results)

    for dim, series in coverage_by_dim.items():
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for key, cover in series:
            ax.plot(np.arange(1, args.m + 1), cover, label=key.replace("_", " "), linewidth=1.6)
        ax.set_xlabel("Number of FPS points already selected")
        ax.set_ylabel("Maximum distance to the nearest selected point")
        ax.set_title(f"FPS coverage radius across {dim}D shapes (lower is better)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"coverage_radius_{dim}d.png", dpi=180)
        plt.close(fig)

    print("DONE: Open *_progression.png, *_selected.png and *_polar_from_first.png")
    print("SUMMARY_CSV=" + str(out / "experiment_summary.csv"))
    print("NOTE: These are Python exploration timings, NOT comparable with gem5 results.")


if __name__ == "__main__":
    main()
