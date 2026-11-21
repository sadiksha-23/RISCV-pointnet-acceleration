
"""
UPPER BOUND PRUNING

Can we safely eliminate entire spherical regions when searching for the next FPS point?

This experiment:
1. Loads the KITTI point cloud.
2. Builds spherical regions.
3. Calculates an upper bound for each region.
4. Calculates a global lower bound from representative points.
5. Excludes regions that cannot contain the next FPS winner.
6. Verifies the selected indices against exact FPS.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from spherical_kitti_tests import (
    load_kitti,
    exact_fps,
    build_regions,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_KITTI = ROOT / "datasets/kitti/0000000000.bin"

DEFAULT_OUTPUT = (
    ROOT
    / "spherical_fps/geometric_bounds/results/upper_bound"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--kitti", type=Path, default=DEFAULT_KITTI)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--m", type=int, default=128)

    parser.add_argument(
        "--origin",
        choices=["centroid", "sensor", "first"],
        default="centroid",
    )

    parser.add_argument("--az-bins", type=int, default=24)
    parser.add_argument("--el-bins", type=int, default=8)
    parser.add_argument("--radial-bins", type=int, default=6)

    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)

    return parser.parse_args()


# ============================================================
# 2. CHOOSE SPHERICAL ORIGIN
# ============================================================

def get_origin(points, name):

    if name == "centroid":
        return points.mean(axis=0)

    if name == "sensor":
        return np.zeros(3)

    return points[0]


# ============================================================
# 3. RUN THE UPPER BOUND EXPERIMENT
# ============================================================

def main():

    args = parse_args()

    if not args.kitti.is_file():
        raise FileNotFoundError(args.kitti)

    if not 2 <= args.m <= args.n:
        raise ValueError("Require 2 <= M <= N")

    if min(args.az_bins, args.el_bins, args.radial_bins) < 1:
        raise ValueError("All region bin counts must be positive")

    args.out.mkdir(parents=True, exist_ok=True)

    print("\n=== UPPER BOUND PRUNING EXPERIMENT ===")

    # Load the input points.
    points = load_kitti(args.kitti, args.n)

    # Generate exact FPS indices for comparison.
    reference, _ = exact_fps(points, args.m, 0)

    origin = get_origin(points, args.origin)

    # Divide the point cloud into spherical regions.
    regions = build_regions(
        points,
        origin,
        args.az_bins,
        args.el_bins,
        args.radial_bins,
    )

    centers = regions["center"]
    radii = regions["radius"]
    members = regions["member"]
    counts = regions["counts"]
    representatives = regions["representative"]

    region_count = len(counts)

    print(f"Input points: {args.n}")
    print(f"Selected points: {args.m}")
    print(f"Origin: {args.origin}")
    print(f"Occupied regions: {region_count}")


    # ========================================================
    # 4. INITIALIZE FPS DISTANCES AND BOUNDS
    # ========================================================

    # Each point's distance to its nearest selected point.
    nearest_sq = np.full(args.n, np.inf)

    # Representative-point nearest distances.
    rep_nearest_sq = np.full(region_count, np.inf)

    # Upper bound on nearest-selected distance in each region.
    region_upper = np.full(region_count, np.inf)

    selected = [0]

    results = []

    total_pruned = 0
    total_possible = args.n * (args.m - 1)


    # ========================================================
    # 5. PROCESS EVERY FPS ITERATION
    # ========================================================

    for step in range(1, args.m):

        new_point = points[selected[-1]]

        # Update nearest distances for ALL points.
        diff = points - new_point

        dist_sq = np.einsum(
            "ij,ij->i", diff, diff
        )

        np.minimum(nearest_sq, dist_sq, out=nearest_sq)


        # ----------------------------------------------------
        # A. UPDATE REGION UPPER BOUNDS
        # ----------------------------------------------------

        # Triangle inequality:
        #
        # distance(p, s) <= distance(center, s) + radius
        #
        # This bounds all points inside a region.

        center_dist = np.linalg.norm(
            centers - new_point,
            axis=1,
        )

        new_upper = center_dist + radii

        # Keep the smallest bound across selected points.
        region_upper = np.minimum(
            region_upper,
            new_upper,
        )


        # ----------------------------------------------------
        # B. CALCULATE A GLOBAL LOWER BOUND
        # ----------------------------------------------------

        # Each region has an actual representative point.
        # The largest representative nearest-distance is
        # an attainable lower bound on the FPS winner.

        rep_diff = representatives - new_point

        rep_dist_sq = np.einsum(
            "ij,ij->i", rep_diff, rep_diff
        )

        rep_nearest_sq = np.minimum(
            rep_nearest_sq,
            rep_dist_sq,
        )

        global_lower = float(
            np.sqrt(rep_nearest_sq.max())
        )


        # ----------------------------------------------------
        # C. IDENTIFY REGIONS THAT CAN BE PRUNED
        # ----------------------------------------------------

        # If a region's upper bound is strictly below
        # the global lower bound, that region cannot
        # contain the next farthest point.
        #
        # Use a conservative tolerance for rounding.

        epsilon = 1e-8 * max(
            1.0,
            global_lower,
            float(region_upper.max()),
        )

        pruned_regions = (
            region_upper < global_lower - epsilon
        )

        pruned_points = int(
            counts[pruned_regions].sum()
        )

        # Keep only points from non-pruned regions.
        eligible_mask = ~pruned_regions[members]

        eligible_indices = np.flatnonzero(eligible_mask)

        if len(eligible_indices) == 0:
            raise AssertionError("No eligible points remain")


        # ----------------------------------------------------
        # D. FIND THE NEXT FPS POINT
        # ----------------------------------------------------

        best_position = int(
            np.argmax(nearest_sq[eligible_indices])
        )

        next_index = int(
            eligible_indices[best_position]
        )


        # ----------------------------------------------------
        # E. VERIFY EXACT CORRECTNESS
        # ----------------------------------------------------

        expected_index = int(reference[step])

        full_search_index = int(np.argmax(nearest_sq))

        if next_index != expected_index:
            raise AssertionError(
                f"Iteration {step}: expected {expected_index}, "
                f"got {next_index}"
            )

        if next_index != full_search_index:
            raise AssertionError(
                f"Iteration {step}: pruned search differs "
                "from the full search"
            )

        if pruned_regions[members[expected_index]]:
            raise AssertionError(
                "The correct FPS winner was pruned!"
            )

        selected.append(next_index)

        total_pruned += pruned_points

        results.append({
            "selected_so_far": step,
            "next_selected_index": next_index,
            "global_lower_bound": global_lower,
            "pruned_regions": int(pruned_regions.sum()),
            "pruned_points": pruned_points,
            "eligible_points": len(eligible_indices),
            "pruned_percent": 100 * pruned_points / args.n,
            "correct": True,
        })


    # ========================================================
    # 6. SAVE RESULTS
    # ========================================================

    output_file = args.out / "upper_bound_results.csv"

    with output_file.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0]),
        )

        writer.writeheader()
        writer.writerows(results)


    # ========================================================
    # 7. PRINT SUMMARY
    # ========================================================

    average_pruning = 100 * total_pruned / total_possible

    print("\n=== RESULTS ===")

    print(f"Average pruned candidates: {average_pruning:.2f}%")
    print(f"All {args.m} FPS indices matched: PASS")

    print("\nIMPORTANT:")
    print("This measures reduced candidate searching.")
    print("All point-distance updates were still performed.")
    print("No execution-time speedup has been established.")

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()