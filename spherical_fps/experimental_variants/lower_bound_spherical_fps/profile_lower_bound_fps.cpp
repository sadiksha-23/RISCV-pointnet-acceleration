
// ============================================================
// LOWER-BOUND SPHERICAL FPS: PERFORMANCE PROFILER
//
// Profiles:
// 1. Spherical region preparation
// 2. Region lower-bound calculations
// 3. Point-distance updates
// 4. Global FPS winner selection
// 5. Other overhead
//
// Uses the SAME algorithm as our existing benchmark.
// Does not overwrite the original implementation.
// ============================================================

#define LB_PROFILE_BUILD
#include "compare_lower_bound_fps.cpp"

using ProfileClock = std::chrono::steady_clock;


// ============================================================
// 1. PROFILING RESULTS
// ============================================================

struct ProfileResult {
    LBResult fps;

    ProfileClock::duration bound_time{};
    ProfileClock::duration update_time{};
    ProfileClock::duration winner_time{};
};

static double duration_ms(ProfileClock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}


// ============================================================
// 2. INSTRUMENTED LOWER-BOUND FPS
// ============================================================

static ProfileResult run_profiled_lower_bound(
    const std::vector<float>& xyz,
    int n,
    int m,
    const double origin[3]
) {
    ProfileResult profile;
    LBResult& result = profile.fps;

    auto start = ProfileClock::now();

    // --------------------------------------------------------
    // PHASE A: PREPARATION
    // --------------------------------------------------------

    std::vector<LBRegion> regions = build_lb_regions(
        xyz, n, origin, 24, 8, 6
    );

    result.occupied_regions = (int)regions.size();

    std::vector<float> nearest(n, 1e38f);

    result.indices.resize(m);
    result.indices[0] = 0;

    auto prepared = ProfileClock::now();

    // --------------------------------------------------------
    // PHASE B: SAMPLING
    // --------------------------------------------------------

    for (int step = 1; step < m; step++) {

        int newest = result.indices[step - 1];

        for (LBRegion& g : regions) {

            // =================================================
            // B1. REGION-BOUND CALCULATION
            // =================================================

            auto bound_start = ProfileClock::now();

            double lower = region_lower_bound(
                g, xyz, newest, origin
            );

            double lower_sq = lower * lower;

            double margin = 1e-5 * (
                1.0 + std::abs((double)g.max_nearest_sq)
            );

            bool can_skip =
                lower_sq >
                (double)g.max_nearest_sq + margin;

            profile.bound_time +=
                ProfileClock::now() - bound_start;

            result.region_tests++;

            if (can_skip) {
                result.skipped_regions++;
                result.skipped_points += g.members.size();
                continue;
            }

            // =================================================
            // B2. DISTANCE UPDATES + REGION MAXIMUM
            // =================================================

            auto update_start = ProfileClock::now();

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

            g.max_nearest_sq = region_best_distance;
            g.best_index = region_best_index;

            profile.update_time +=
                ProfileClock::now() - update_start;
        }

        // =====================================================
        // B3. GLOBAL WINNER SELECTION
        // =====================================================

        auto winner_start = ProfileClock::now();

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
                "Profiler: no FPS winner found"
            );
        }

        result.indices[step] = global_best_index;

        profile.winner_time +=
            ProfileClock::now() - winner_start;
    }

    auto finished = ProfileClock::now();

    result.prep_ms = elapsed_ms(start, prepared);
    result.sample_ms = elapsed_ms(prepared, finished);
    result.total_ms = elapsed_ms(start, finished);

    return profile;
}


// ============================================================
// 3. STATISTICS
// ============================================================

static double mean_ms(const std::vector<double>& values) {
    if (values.empty()) return 0;

    return std::accumulate(
        values.begin(), values.end(), 0.0
    ) / values.size();
}

static double percentage(double part, double total) {
    return total > 0 ? 100.0 * part / total : 0;
}


// ============================================================
// 4. MAIN PROFILING EXPERIMENT
// ============================================================

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

        // ----------------------------------------------------
        // A. MEASURE CENTROID CALCULATION
        // ----------------------------------------------------

        auto origin_start = ProfileClock::now();

        double origin[3] = {0, 0, 0};

        for (int i = 0; i < n; i++) {
            for (int d = 0; d < 3; d++) {
                origin[d] += xyz[(size_t)i * 3 + d];
            }
        }

        for (int d = 0; d < 3; d++) {
            origin[d] /= n;
        }

        double origin_ms = elapsed_ms(
            origin_start, ProfileClock::now()
        );

        // ----------------------------------------------------
        // B. REFERENCE CORRECTNESS
        // ----------------------------------------------------

        std::vector<int> reference =
            run_basic_fps(xyz, n, m);

        std::vector<double> baseline_total;
        std::vector<double> profiled_total;
        std::vector<double> prep_times;
        std::vector<double> sample_times;

        std::vector<double> bound_times;
        std::vector<double> update_times;
        std::vector<double> winner_times;
        std::vector<double> other_times;

        uint64_t evaluations = 0;
        uint64_t skipped_points = 0;
        uint64_t skipped_regions = 0;
        uint64_t region_tests = 0;
        int occupied_regions = 0;

        std::cout << std::fixed << std::setprecision(3);

        std::cout
            << "\n=== LOWER-BOUND FPS PROFILER ===\n"
            << "N=" << n
            << " M=" << m
            << " repeats=" << repeats
            << "\nOrigin=centroid\n"
            << "Centroid calculation: "
            << origin_ms << " ms\n";

        // ----------------------------------------------------
        // C. REPEAT PROFILING
        // ----------------------------------------------------

        for (int rep = 0; rep < repeats; rep++) {

            LBResult baseline;
            ProfileResult profiled;

            // Alternate execution order to reduce cache bias.
            if (rep % 2 == 0) {
                baseline = run_lower_bound_fps(
                    xyz, n, m, origin
                );

                profiled = run_profiled_lower_bound(
                    xyz, n, m, origin
                );
            } else {
                profiled = run_profiled_lower_bound(
                    xyz, n, m, origin
                );

                baseline = run_lower_bound_fps(
                    xyz, n, m, origin
                );
            }

            verify_indices(
                reference,
                baseline.indices,
                "Original LowerBoundFPS"
            );

            verify_indices(
                reference,
                profiled.fps.indices,
                "Profiled LowerBoundFPS"
            );

            double bounds = duration_ms(
                profiled.bound_time
            );

            double updates = duration_ms(
                profiled.update_time
            );

            double winner = duration_ms(
                profiled.winner_time
            );

            double sampling = profiled.fps.sample_ms;

            // Includes outer-loop bookkeeping and timing
            // instrumentation that is not assigned to a phase.
            double other = sampling - bounds - updates - winner;

            baseline_total.push_back(baseline.total_ms);
            profiled_total.push_back(profiled.fps.total_ms);

            prep_times.push_back(profiled.fps.prep_ms);
            sample_times.push_back(sampling);

            bound_times.push_back(bounds);
            update_times.push_back(updates);
            winner_times.push_back(winner);
            other_times.push_back(other);

            evaluations = profiled.fps.distance_evals;
            skipped_points = profiled.fps.skipped_points;
            skipped_regions = profiled.fps.skipped_regions;
            region_tests = profiled.fps.region_tests;
            occupied_regions = profiled.fps.occupied_regions;

            std::cout
                << "\nRun " << rep + 1 << "/" << repeats
                << "\n  Original total: "
                << baseline.total_ms << " ms"
                << "\n  Profiled total: "
                << profiled.fps.total_ms << " ms"
                << "\n  Preparation: "
                << profiled.fps.prep_ms << " ms"
                << "\n  Region bounds: "
                << bounds << " ms"
                << "\n  Point updates: "
                << updates << " ms"
                << "\n  Winner selection: "
                << winner << " ms"
                << "\n  Other / profiling overhead: "
                << other << " ms"
                << "\n  Correctness: PASS (" << m << "/" << m << ")\n";
        }

        // ----------------------------------------------------
        // D. FINAL PROFILE SUMMARY
        // ----------------------------------------------------

        double prep = mean_ms(prep_times);
        double sample = mean_ms(sample_times);

        double bounds = mean_ms(bound_times);
        double updates = mean_ms(update_times);
        double winner = mean_ms(winner_times);
        double other = mean_ms(other_times);

        std::cout << "\n=== TIME BREAKDOWN ===\n";

        std::cout
            << "Centroid calculation (once): "
            << origin_ms << " ms\n"
            << "Preparation mean: "
            << prep << " ms\n"
            << "Sampling mean (profiled): "
            << sample << " ms\n";

        std::cout << "\nSampling components:\n";

        std::cout
            << "Region bounds: "
            << bounds << " ms ("
            << percentage(bounds, sample) << "%)\n"

            << "Point updates: "
            << updates << " ms ("
            << percentage(updates, sample) << "%)\n"

            << "Winner selection: "
            << winner << " ms ("
            << percentage(winner, sample) << "%)\n"

            << "Other / profiling overhead: "
            << other << " ms ("
            << percentage(other, sample) << "%)\n";

        std::cout << "\n=== REGION WORK ===\n";

        std::cout
            << "Occupied regions: " << occupied_regions << "\n"
            << "Region-bound tests: " << region_tests << "\n"
            << "Regions skipped: " << skipped_regions << "\n"
            << "Point updates skipped: " << skipped_points << "\n"
            << "Point-distance evaluations: " << evaluations << "\n";

        std::cout << "\n=== PROFILER OVERHEAD CHECK ===\n";

        std::cout
            << "Original median total: "
            << median_value(baseline_total) << " ms\n"
            << "Profiled median total: "
            << median_value(profiled_total) << " ms\n";

        std::cout
            << "\nAll indices matched BasicFPS: PASS\n"
            << "NOTE: Instrumentation adds overhead to sampling.\n"
            << "Use original benchmark for final speed claims.\n";

        return 0;

    } catch (const std::exception& e) {

        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
