#!/usr/bin/env python3

"""
LOWER BOUNDS FOR DISTANCE UPDATES

Can we avoid recalculating point distances when a newly selected FPS point cannot improve them?

This experiment:
1. Loads KITTI and builds spherical regions.
2. Maintains exact nearest-selected distances.
3. Calculates a lower bound on the distance from the new
   selected point to every point in a region.
4. Skips updating a region when the new point is proven
   unable to improve any point's existing distance.
5. Checks all distances and selected indices against FPS.
6. Counts how many distance calculations could be avoided.

We test:
- XYZ bounding-sphere lower bound
- Spherical radial lower bound
- Combined lower bound

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
    / "spherical_fps/geometric_bounds/results/lower_bound"
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


def get_origin(points, name):

    if name == "centroid":
        return points.mean(axis=0)

    if name == "sensor":
        return np.zeros(3)

    return points[0]


# ============================================================
# 2. LOAD DATA AND BUILD REGIONS
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

    print("\n=== LOWER BOUND UPDATE EXPERIMENT ===")

    points = load_kitti(args.kitti, args.n)

    reference, _ = exact_fps(points, args.m, 0)

    origin = get_origin(points, args.origin)

    regions = build_regions(
        points,
        origin,
        args.az_bins,
        args.el_bins,
        args.radial_bins,
    )

    members = regions["member"]
    counts = regions["counts"]
    centers = regions["center"]
    radii = regions["radius"]

    region_count = len(counts)

    print(f"Input points: {args.n}")
    print(f"Selected points: {args.m}")
    print(f"Origin: {args.origin}")
    print(f"Occupied regions: {region_count}")


    # ========================================================
    # 3. PRECOMPUTE RADIAL RANGES
    # ========================================================

    # Every point has a radius from the chosen origin.
    # For each region, store its minimum and maximum radius.

    point_radii = regions["r"]

    radial_min = np.full(region_count, np.inf)
    radial_max = np.full(region_count, -np.inf)

    np.minimum.at(
        radial_min,
        members,
        point_radii,
    )

    np.maximum.at(
        radial_max,
        members,
        point_radii,
    )


    # ========================================================
    # 4. INITIALIZE FPS
    # ========================================================

    nearest_sq = np.full(args.n, np.inf)

    # Separate full-update reference for validation.
    reference_nearest_sq = np.full(args.n, np.inf)

    selected = [0]

    results = []

    total_updates = 0
    total_skipped = 0

    total_possible = args.n * (args.m - 1)


    # ========================================================
    # 5. PROCESS FPS ITERATIONS
    # ========================================================

    for step in range(args.m - 1):

        new_point = points[selected[-1]]


        # ----------------------------------------------------
        # A. FIND LARGEST EXISTING DISTANCE PER REGION
        # ----------------------------------------------------

        # For each region, find the largest current
        # nearest-selected distance among its points.

        max_current_sq = np.full(
            region_count,
            -np.inf,
        )

        np.maximum.at(
            max_current_sq,
            members,
            nearest_sq,
        )

        max_current = np.sqrt(max_current_sq)


        # ----------------------------------------------------
        # B. XYZ BOUNDING-SPHERE LOWER BOUND
        # ----------------------------------------------------

        # Every point is within its region's bounding sphere.
        #
        # distance(new_point, p)
        # >= distance(new_point, center) - region_radius

        center_distance = np.linalg.norm(
            centers - new_point,
            axis=1,
        )

        xyz_lower = np.maximum(
            center_distance - radii,
            0.0,
        )


        # ----------------------------------------------------
        # C. SPHERICAL RADIAL LOWER BOUND
        # ----------------------------------------------------

        # By the reverse triangle inequality:
        #
        # distance(new_point, p)
        # >= abs(radius(new_point) - radius(p))
        #
        # Use the nearest possible radius in each region.

        new_radius = np.linalg.norm(
            new_point - origin
        )

        radial_lower = np.maximum.reduce([
            radial_min - new_radius,
            new_radius - radial_max,
            np.zeros(region_count),
        ])


        # ----------------------------------------------------
        # D. COMBINE LOWER BOUNDS
        # ----------------------------------------------------

        # Both lower bounds are safe.
        # Their maximum gives a bound at least as strong
        # as either individual bound.

        combined_lower = np.maximum(
            xyz_lower,
            radial_lower,
        )


        # ----------------------------------------------------
        # E. DETERMINE WHICH REGIONS NEED UPDATES
        # ----------------------------------------------------

        # If the minimum possible NEW distance for a region
        # is larger than every existing nearest distance
        # in that region, no point there can improve.
        #
        # Those regions can safely skip the update.

        epsilon = 1e-9 * max(
            1.0,
            float(new_radius),
            float(radii.max()),
        )

        skip_xyz = xyz_lower > max_current + epsilon

        skip_radial = radial_lower > max_current + epsilon

        skip_combined = (
            combined_lower > max_current + epsilon
        )

        # Apply the combined lower bound.
        update_mask = ~skip_combined[members]


        # ----------------------------------------------------
        # F. UPDATE ONLY POINTS THAT NEED IT
        # ----------------------------------------------------

        update_points = points[update_mask]

        diff = update_points - new_point

        updated_dist_sq = np.einsum(
            "ij,ij->i",
            diff,
            diff,
        )

        nearest_sq[update_mask] = np.minimum(
            nearest_sq[update_mask],
            updated_dist_sq,
        )

        updates = int(update_mask.sum())
        skipped = args.n - updates

        total_updates += updates
        total_skipped += skipped


        # ----------------------------------------------------
        # G. VERIFY AGAINST FULL DISTANCE UPDATES
        # ----------------------------------------------------

        # For validation only, calculate every distance.
        # This extra work is NOT part of the proposed
        # optimized algorithm.

        full_diff = points - new_point

        full_dist_sq = np.einsum(
            "ij,ij->i",
            full_diff,
            full_diff,
        )

        reference_nearest_sq = np.minimum(
            reference_nearest_sq,
            full_dist_sq,
        )

        if not np.array_equal(
            nearest_sq,
            reference_nearest_sq,
        ):
            raise AssertionError(
                f"Distance mismatch at iteration {step + 1}"
            )


        # ----------------------------------------------------
        # H. VERIFY THE NEXT SELECTED POINT
        # ----------------------------------------------------

        next_index = int(np.argmax(nearest_sq))

        expected_index = int(reference[step + 1])

        if next_index != expected_index:
            raise AssertionError(
                f"Iteration {step + 1}: "
                f"expected {expected_index}, got {next_index}"
            )

        selected.append(next_index)


        # ----------------------------------------------------
        # I. RECORD EXPERIMENT RESULTS
        # ----------------------------------------------------

        skipped_xyz = int(counts[skip_xyz].sum())

        skipped_radial = int(counts[skip_radial].sum())

        results.append({
            "selected_so_far": step + 1,
            "next_selected_index": next_index,
            "skipped_xyz": skipped_xyz,
            "skipped_radial": skipped_radial,
            "skipped_combined": skipped,
            "distance_updates_performed": updates,
            "distance_updates_saved": skipped,
            "saved_percent": 100 * skipped / args.n,
            "correct": True,
        })


    # ========================================================
    # 6. SAVE RESULTS
    # ========================================================

    output_file = args.out / "lower_bound_results.csv"

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

    saved_percent = 100 * total_skipped / total_possible

    print("\n=== RESULTS ===")

    print(f"Baseline distance evaluations: {total_possible}")
    print(f"Evaluations performed by bounded updates: {total_updates}")
    print(f"Potential distance evaluations saved: {total_skipped}")

    print(f"Potential reduction: {saved_percent:.2f}%")

    print(f"All {args.m} FPS indices matched: PASS")
    print("All nearest distances matched: PASS")

    print("\nIMPORTANT:")
    print("This measures potential distance-update savings.")
    print("Validation still performs full-distance calculations.")
    print("Runtime improvement has not yet been established.")

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()