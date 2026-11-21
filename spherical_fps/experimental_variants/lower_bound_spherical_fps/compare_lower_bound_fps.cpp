// ============================================================
// LOWER-BOUND SPHERICAL FPS
//
// Compare:
//   1. BasicFPS
//   2. RadiusFPS (existing implementation)
//   3. Lower-Bound SphericalFPS (new implementation)
//
// Dataset: KITTI, N=100000, M=128
//
// Can spherical regions and geometric lower bounds reduce distance updates AND improve total execution time?
//
// All algorithms must select identical point indices.
// ============================================================

#define main radius_original_main
#include "../radiusfps_spherical_hybrid/radiusfps_spherical_test.cpp"
#undef main

#include <iostream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

using Clock = std::chrono::steady_clock;


// ============================================================
// 1. DATA STRUCTURES
// ============================================================

struct LBRegion {
    std::vector<int> members;

    double cx = 0, cy = 0, cz = 0;
    double radius = 0;

    double rmin = 1e300;
    double rmax = 0;

    // Largest nearest-selected squared distance in region.
    float max_nearest_sq = 1e38f;

    // Point with the largest nearest-selected distance.
    int best_index = -1;
};

struct LBResult {
    double prep_ms = 0;
    double sample_ms = 0;
    double total_ms = 0;

    uint64_t distance_evals = 0;
    uint64_t skipped_points = 0;
    uint64_t skipped_regions = 0;
    uint64_t region_tests = 0;

    int occupied_regions = 0;

    std::vector<int> indices;
};

static double elapsed_ms(
    Clock::time_point start,
    Clock::time_point end
) {
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}


// ============================================================
// 2. POINT DISTANCE
// ============================================================

// Use float arithmetic like the existing RadiusFPS.
// Squared distances avoid unnecessary square roots.

static inline float lb_sqdist(
    const std::vector<float>& xyz,
    int a,
    int b
) {
    size_t i = (size_t)a * 3;
    size_t j = (size_t)b * 3;

    float dx = xyz[i]     - xyz[j];
    float dy = xyz[i + 1] - xyz[j + 1];
    float dz = xyz[i + 2] - xyz[j + 2];

    return dx * dx + dy * dy + dz * dz;
}


// ============================================================
// 3. BASIC FPS REFERENCE
// ============================================================

static std::vector<int> run_basic_fps(
    const std::vector<float>& xyz,
    int n,
    int m
) {
    std::vector<float> nearest(n, 1e38f);
    std::vector<int> selected(m);

    selected[0] = 0;

    for (int step = 1; step < m; step++) {

        int newest = selected[step - 1];

        float best_distance = -1;
        int best_index = -1;

        for (int i = 0; i < n; i++) {

            float d = lb_sqdist(xyz, i, newest);

            nearest[i] = std::min(nearest[i], d);

            if (
                nearest[i] > best_distance ||
                (
                    nearest[i] == best_distance &&
                    (best_index < 0 || i < best_index)
                )
            ) {
                best_distance = nearest[i];
                best_index = i;
            }
        }

        selected[step] = best_index;
    }

    return selected;
}


// ============================================================
// 4. BUILD SPHERICAL REGIONS
// ============================================================

static std::vector<LBRegion> build_lb_regions(
    const std::vector<float>& xyz,
    int n,
    const double origin[3],
    int az_bins,
    int el_bins,
    int radial_bins
) {
    std::vector<double> radius(n);
    std::vector<double> azimuth(n);
    std::vector<double> elevation(n);

    double maximum_radius = 1e-12;

    for (int i = 0; i < n; i++) {

        size_t p = (size_t)i * 3;

        double x = (double)xyz[p]     - origin[0];
        double y = (double)xyz[p + 1] - origin[1];
        double z = (double)xyz[p + 2] - origin[2];

        radius[i] = std::sqrt(x*x + y*y + z*z);
        azimuth[i] = std::atan2(y, x);
        elevation[i] = std::atan2(z, std::hypot(x, y));

        maximum_radius = std::max(
            maximum_radius, radius[i]
        );
    }

    int total_bins = az_bins * el_bins * radial_bins;

    std::vector<LBRegion> groups(total_bins);

    for (int i = 0; i < n; i++) {

        int a = std::clamp(
            (int)((azimuth[i] + PI) / TWO_PI * az_bins),
            0, az_bins - 1
        );

        int e = std::clamp(
            (int)((elevation[i] + PI / 2) / PI * el_bins),
            0, el_bins - 1
        );

        int r = std::clamp(
            (int)(radius[i] / maximum_radius * radial_bins),
            0, radial_bins - 1
        );

        int id = (r * el_bins + e) * az_bins + a;

        LBRegion& g = groups[id];

        g.members.push_back(i);

        size_t p = (size_t)i * 3;

        g.cx += xyz[p];
        g.cy += xyz[p + 1];
        g.cz += xyz[p + 2];

        g.rmin = std::min(g.rmin, radius[i]);
        g.rmax = std::max(g.rmax, radius[i]);
    }

    // Keep only occupied regions.
    std::vector<LBRegion> occupied;

    for (LBRegion& g : groups) {

        if (g.members.empty()) continue;

        double count = (double)g.members.size();

        g.cx /= count;
        g.cy /= count;
        g.cz /= count;

        // Find an enclosing XYZ sphere around actual points.
        for (int i : g.members) {

            size_t p = (size_t)i * 3;

            double dx = (double)xyz[p]     - g.cx;
            double dy = (double)xyz[p + 1] - g.cy;
            double dz = (double)xyz[p + 2] - g.cz;

            double d = std::sqrt(dx*dx + dy*dy + dz*dz);

            g.radius = std::max(g.radius, d);
        }

        // Conservative padding for floating-point roundoff.
        g.radius += 1e-7 * (1.0 + g.radius);
        g.rmin = std::max(
            0.0, g.rmin - 1e-7 * (1.0 + g.rmin)
        );
        g.rmax += 1e-7 * (1.0 + g.rmax);

        g.best_index = g.members[0];

        occupied.push_back(std::move(g));
    }

    return occupied;
}


// ============================================================
// 5. CALCULATE REGION LOWER BOUND
// ============================================================

// For any point p in a region:
//
// XYZ bound:
//   distance(s,p) >= distance(s,center) - region_radius
//
// Radial bound:
//   distance(s,p) >= distance between radius(s)
//                    and the region's radial interval
//
// Take the maximum of both safe lower bounds.

static double region_lower_bound(
    const LBRegion& g,
    const std::vector<float>& xyz,
    int newest,
    const double origin[3]
) {
    size_t p = (size_t)newest * 3;

    double sx = xyz[p];
    double sy = xyz[p + 1];
    double sz = xyz[p + 2];

    double dx = sx - g.cx;
    double dy = sy - g.cy;
    double dz = sz - g.cz;

    double center_distance = std::sqrt(
        dx*dx + dy*dy + dz*dz
    );

    double xyz_lower = std::max(
        0.0, center_distance - g.radius
    );

    double ox = sx - origin[0];
    double oy = sy - origin[1];
    double oz = sz - origin[2];

    double selected_radius = std::sqrt(
        ox*ox + oy*oy + oz*oz
    );

    double radial_lower = std::max({
        0.0,
        g.rmin - selected_radius,
        selected_radius - g.rmax
    });

    double combined = std::max(
        xyz_lower, radial_lower
    );

    // Conservative reduction before testing the bound.
    return std::max(
        0.0,
        combined - 1e-6 * (1.0 + combined)
    );
}


// ============================================================
// 6. LOWER-BOUND SPHERICAL FPS
// ============================================================

static LBResult run_lower_bound_fps(
    const std::vector<float>& xyz,
    int n,
    int m,
    const double origin[3]
) {
    LBResult result;

    auto start = Clock::now();

    // Preparation: build spherical regions.
    std::vector<LBRegion> regions = build_lb_regions(
        xyz, n, origin, 24, 8, 6
    );

    result.occupied_regions = (int)regions.size();

    std::vector<float> nearest(n, 1e38f);

    result.indices.resize(m);
    result.indices[0] = 0;

    auto prepared = Clock::now();

    // Main FPS sampling loop.
    for (int step = 1; step < m; step++) {

        int newest = result.indices[step - 1];

        // ----------------------------------------------------
        // A. CHECK EACH SPHERICAL REGION
        // ----------------------------------------------------

        for (LBRegion& g : regions) {

            result.region_tests++;

            double lower = region_lower_bound(
                g, xyz, newest, origin
            );

            // The region can be skipped only if the new
            // selected point cannot improve ANY existing
            // nearest distance inside the region.

            double lower_sq = lower * lower;

            // Strict comparison and additional margin
            // protect against numerical equality.
            double margin = 1e-5 * (
                1.0 + std::abs((double)g.max_nearest_sq)
            );

            if (
                lower_sq >
                (double)g.max_nearest_sq + margin
            ) {
                result.skipped_regions++;
                result.skipped_points += g.members.size();

                // Cached nearest distances and region winner
                // remain valid because no value can improve.
                continue;
            }

            // ------------------------------------------------
            // B. UPDATE POINTS IN THIS REGION
            // ------------------------------------------------

            float region_best_distance = -1;
            int region_best_index = -1;

            for (int i : g.members) {

                float d = lb_sqdist(xyz, i, newest);

                result.distance_evals++;

                nearest[i] = std::min(nearest[i], d);

                if (
                    nearest[i] > region_best_distance ||
                    (
                        nearest[i] == region_best_distance &&
                        (
                            region_best_index < 0 ||
                            i < region_best_index
                        )
                    )
                ) {
                    region_best_distance = nearest[i];
                    region_best_index = i;
                }
            }

            // Cache the region's exact maximum.
            // Skipped regions keep their previous maximum.
            g.max_nearest_sq = region_best_distance;
            g.best_index = region_best_index;
        }

        // ----------------------------------------------------
        // C. SELECT THE NEXT FPS POINT
        // ----------------------------------------------------

        float global_best_distance = -1;
        int global_best_index = -1;

        for (const LBRegion& g : regions) {

            if (
                g.max_nearest_sq > global_best_distance ||
                (
                    g.max_nearest_sq == global_best_distance &&
                    (
                        global_best_index < 0 ||
                        g.best_index < global_best_index
                    )
                )
            ) {
                global_best_distance = g.max_nearest_sq;
                global_best_index = g.best_index;
            }
        }

        if (global_best_index < 0) {
            throw std::runtime_error(
                "No FPS winner found"
            );
        }

        result.indices[step] = global_best_index;
    }

    auto end = Clock::now();

    result.prep_ms = elapsed_ms(start, prepared);
    result.sample_ms = elapsed_ms(prepared, end);
    result.total_ms = elapsed_ms(start, end);

    return result;
}


// ============================================================
// 7. STATISTICS AND CORRECTNESS
// ============================================================

static double median_value(std::vector<double> values) {
    std::sort(values.begin(), values.end());

    size_t n = values.size();

    if (n % 2) return values[n / 2];

    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static void verify_indices(
    const std::vector<int>& reference,
    const std::vector<int>& actual,
    const std::string& name
) {
    if (reference.size() != actual.size()) {
        throw std::runtime_error(
            name + ": selection count mismatch"
        );
    }

    for (size_t i = 0; i < reference.size(); i++) {
        if (reference[i] != actual[i]) {
            throw std::runtime_error(
                name + " mismatch at selection " +
                std::to_string(i + 1) +
                ": expected " + std::to_string(reference[i]) +
                ", got " + std::to_string(actual[i])
            );
        }
    }
}


// ============================================================
// 8. MAIN BENCHMARK
// ============================================================

#ifndef LB_PROFILE_BUILD
int main(int argc, char** argv) {

    try {

        std::string dataset =
            "datasets/kitti/0000000000.bin";

        int n = 100000;
        int m = 128;
        int repeats = 5;

        if (argc > 1) dataset = argv[1];
        if (argc > 2) n = std::stoi(argv[2]);
        if (argc > 3) m = std::stoi(argv[3]);
        if (argc > 4) repeats = std::stoi(argv[4]);

        if (n < 1 || m < 1 || m > n || repeats < 1) {
            throw std::runtime_error(
                "Invalid N, M, or repeats"
            );
        }

        std::vector<float> xyz;

        if (!loadKittiFrame(dataset.c_str(), xyz)) {
            return 1;
        }

        if ((size_t)n * 3 > xyz.size()) {
            throw std::runtime_error(
                "Dataset contains fewer than N points"
            );
        }

        // Centroid origin, matching the Python experiment.
        double origin[3] = {0, 0, 0};

        for (int i = 0; i < n; i++) {
            for (int d = 0; d < 3; d++) {
                origin[d] += xyz[(size_t)i * 3 + d];
            }
        }

        for (int d = 0; d < 3; d++) {
            origin[d] /= n;
        }

        std::cout << std::fixed << std::setprecision(3);

        std::cout
            << "\n=== FPS ALGORITHM COMPARISON ===\n"
            << "N=" << n
            << " M=" << m
            << " repeats=" << repeats
            << "\nOrigin=centroid\n";

        // Reference correctness, outside timed runs.
        std::vector<int> reference = run_basic_fps(
            xyz, n, m
        );

        std::vector<double> basic_times;
        std::vector<double> radius_prep, radius_sample;
        std::vector<double> radius_times;
        std::vector<double> lower_prep, lower_sample;
        std::vector<double> lower_times;

        uint64_t last_evals = 0;
        uint64_t last_skips = 0;
        int last_regions = 0;

        for (int rep = 0; rep < repeats; rep++) {

            std::cout << "\nRun " << rep + 1
                      << "/" << repeats << "\n";

            // ----------------------------------------------
            // BASIC FPS
            // ----------------------------------------------

            auto b0 = Clock::now();
            std::vector<int> basic = run_basic_fps(
                xyz, n, m
            );
            auto b1 = Clock::now();

            double basic_ms = elapsed_ms(b0, b1);

            verify_indices(reference, basic, "BasicFPS");

            basic_times.push_back(basic_ms);

            // ----------------------------------------------
            // RADIUS FPS
            // ----------------------------------------------

            // Uses the repository's existing scalar RadiusFPS.
            // Voxel size is 2.0, matching earlier experiments.
            double radius_origin[3] = {0, 0, 0};

            Result radius = runOnce(
                xyz, n, m, 2.0f,
                false, radius_origin, false
            );

            verify_indices(
                reference, radius.indices, "RadiusFPS"
            );

            radius_prep.push_back(radius.prep_ms);
            radius_sample.push_back(radius.sample_ms);
            radius_times.push_back(radius.total_ms);

            // ----------------------------------------------
            // LOWER-BOUND SPHERICAL FPS
            // ----------------------------------------------

            LBResult lower = run_lower_bound_fps(
                xyz, n, m, origin
            );

            verify_indices(
                reference, lower.indices,
                "LowerBoundSphericalFPS"
            );

            lower_prep.push_back(lower.prep_ms);
            lower_sample.push_back(lower.sample_ms);
            lower_times.push_back(lower.total_ms);

            last_evals = lower.distance_evals;
            last_skips = lower.skipped_points;
            last_regions = lower.occupied_regions;

            std::cout
                << "BasicFPS: " << basic_ms << " ms\n"
                << "RadiusFPS: " << radius.total_ms << " ms\n"
                << "LowerBoundFPS: " << lower.total_ms << " ms\n"
                << "Exact indices: PASS (" << m << "/" << m << ")\n";
        }

        // ====================================================
        // 9. FINAL SUMMARY
        // ====================================================

        double basic_med = median_value(basic_times);
        double radius_med = median_value(radius_times);
        double lower_med = median_value(lower_times);

        uint64_t baseline_evals =
            (uint64_t)n * (uint64_t)(m - 1);

        double saved_pct = 100.0 * (
            1.0 - (double)last_evals / baseline_evals
        );

        std::cout << "\n=== MEDIAN RESULTS ===\n";

        std::cout
            << "BasicFPS total: " << basic_med << " ms\n"
            << "RadiusFPS preparation: "
            << median_value(radius_prep) << " ms\n"
            << "RadiusFPS sampling: "
            << median_value(radius_sample) << " ms\n"
            << "RadiusFPS total: " << radius_med << " ms\n"
            << "LowerBoundFPS preparation: "
            << median_value(lower_prep) << " ms\n"
            << "LowerBoundFPS sampling: "
            << median_value(lower_sample) << " ms\n"
            << "LowerBoundFPS total: " << lower_med << " ms\n";

        std::cout << "\n=== LOWER BOUND WORK ===\n";

        std::cout
            << "Occupied spherical regions: " << last_regions << "\n"
            << "Basic distance evaluations: " << baseline_evals << "\n"
            << "Lower-bound distance evaluations: " << last_evals << "\n"
            << "Region-skipped point updates: " << last_skips << "\n"
            << "Distance evaluation reduction: "
            << saved_pct << "%\n";

        std::cout << "\n=== SPEED RATIOS ===\n";

        std::cout
            << "Basic / LowerBound: "
            << basic_med / lower_med << "x\n"
            << "Radius / LowerBound: "
            << radius_med / lower_med << "x\n";

        std::cout
            << "\nCORRECTNESS: PASS (all algorithms, all runs)\n";

        std::cout
            << "\nNOTE: Native host timings, not gem5 timings.\n";

        return 0;

    } catch (const std::exception& e) {

        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}
#endif // LB_PROFILE_BUILD