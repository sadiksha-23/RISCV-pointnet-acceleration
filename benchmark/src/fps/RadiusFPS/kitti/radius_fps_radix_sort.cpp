#include <fstream>
#include <vector>
#include <stdint.h>
#include <inttypes.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

using namespace std;

extern "C" {
    #include <gem5/m5ops.h>
}

struct PointRecord {
    float x;
    float y;
    float z;

    int original_index;

    int vx;
    int vy;
    int vz;

    uint64_t voxel_key;
};

struct Voxel {
    int start;
    int count;

    float cx;
    float cy;
    float cz;

    float radius;

    float max_dist;

    int best_point;
    int best_original_index;
};

struct SamplingProfile {
    uint64_t setup_cycles;

    uint64_t voxel_bound_cycles;
    uint64_t voxel_bound_events;

    uint64_t voxel_metadata_cycles;
    uint64_t voxel_metadata_events;

    uint64_t point_prune_cycles;
    uint64_t point_prune_events;

    uint64_t full_distance_cycles;
    uint64_t full_distance_events;

    uint64_t local_best_cycles;
    uint64_t local_best_events;

    uint64_t global_select_cycles;
    uint64_t global_select_events;

    uint64_t voxel_checks;
    uint64_t voxel_skips;
    uint64_t active_voxels;

    uint64_t point_checks;
    uint64_t x_prunes;
    uint64_t y_prunes;
    uint64_t z_prunes;
    uint64_t full_checks;

    uint64_t total_profiled_cycles;
};

static inline uint64_t readCycles() {
    uint64_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles) :: "memory");
    return cycles;
}

static uint64_t measureTimerOverhead() {
    uint64_t total = 0;
    const int trials = 10000;

    for (int i = 0; i < trials; i++) {
        uint64_t t0 = readCycles();
        uint64_t t1 = readCycles();
        total += t1 - t0;
    }

    return total / trials;
}

static uint64_t correctedCycles(uint64_t raw, uint64_t events, uint64_t overhead) {
    uint64_t subtraction = events * overhead;

    if (raw > subtraction)
        return raw - subtraction;

    return 0;
}

static double percentOf(uint64_t value, uint64_t total) {
    if (total == 0)
        return 0.0;

    return 100.0 * (double)value / (double)total;
}

static void radixSortPoints(PointRecord *points, PointRecord *scratch, int n) {
    const int RADIX = 256;
    const int PASSES = 8;

    PointRecord *src = points;
    PointRecord *dst = scratch;

    for (int pass = 0; pass < PASSES; pass++) {
        int count[RADIX] = {0};
        int shift = pass * 8;

        for (int i = 0; i < n; i++) {
            int digit = (int)((src[i].voxel_key >> shift) & 0xFFULL);
            count[digit]++;
        }

        int position = 0;

        for (int i = 0; i < RADIX; i++) {
            int current = count[i];
            count[i] = position;
            position += current;
        }

        for (int i = 0; i < n; i++) {
            int digit = (int)((src[i].voxel_key >> shift) & 0xFFULL);
            dst[count[digit]] = src[i];
            count[digit]++;
        }

        PointRecord *swap = src;
        src = dst;
        dst = swap;
    }

    if (src != points) {
        for (int i = 0; i < n; i++)
            points[i] = src[i];
    }
}

static int buildVoxels(const float *dataset, int n, float voxel_size,
                       PointRecord *points, PointRecord *scratch,
                       Voxel *voxels,
                       float min_x, float min_y, float min_z) {

    int max_vy = 0;
    int max_vz = 0;

    for (int i = 0; i < n; i++) {
        float x = dataset[i * 3 + 0];
        float y = dataset[i * 3 + 1];
        float z = dataset[i * 3 + 2];

        points[i].x = x;
        points[i].y = y;
        points[i].z = z;

        points[i].original_index = i;

        points[i].vx = (int)floorf((x - min_x) / voxel_size);
        points[i].vy = (int)floorf((y - min_y) / voxel_size);
        points[i].vz = (int)floorf((z - min_z) / voxel_size);

        if (points[i].vy > max_vy)
            max_vy = points[i].vy;

        if (points[i].vz > max_vz)
            max_vz = points[i].vz;
    }

    uint64_t ny = (uint64_t)max_vy + 1ULL;
    uint64_t nz = (uint64_t)max_vz + 1ULL;

    for (int i = 0; i < n; i++) {
        uint64_t vx = (uint64_t)points[i].vx;
        uint64_t vy = (uint64_t)points[i].vy;
        uint64_t vz = (uint64_t)points[i].vz;

        points[i].voxel_key = (vx * ny + vy) * nz + vz;
    }

    radixSortPoints(points, scratch, n);

    int voxel_count = 0;
    int start = 0;

    while (start < n) {
        int vx = points[start].vx;
        int vy = points[start].vy;
        int vz = points[start].vz;

        int end = start + 1;

        while (end < n &&
               points[end].vx == vx &&
               points[end].vy == vy &&
               points[end].vz == vz) {
            end++;
        }

        Voxel *v = &voxels[voxel_count];

        v->start = start;
        v->count = end - start;

        v->cx = min_x + ((float)vx + 0.5f) * voxel_size;
        v->cy = min_y + ((float)vy + 0.5f) * voxel_size;
        v->cz = min_z + ((float)vz + 0.5f) * voxel_size;

        v->radius = 0.5f * sqrtf(3.0f) * voxel_size;

        v->max_dist = 1e38f;
        v->best_point = start;
        v->best_original_index = points[start].original_index;

        voxel_count++;
        start = end;
    }

    return voxel_count;
}

static void radiusFarthestPointSamplingProfile(
    int n,
    int m,
    PointRecord *points,
    Voxel *voxels,
    int voxel_count,
    float *temp,
    int *idxs,
    SamplingProfile *profile
) {
    if (m <= 0)
        return;

    uint64_t total_start = readCycles();

    uint64_t t0 = readCycles();

    for (int i = 0; i < n; i++)
        temp[i] = 1e38f;

    int old = 0;

    for (int i = 0; i < n; i++) {
        if (points[i].original_index == 0) {
            old = i;
            break;
        }
    }

    idxs[0] = 0;

    uint64_t t1 = readCycles();
    profile->setup_cycles += t1 - t0;

    for (int j = 1; j < m; j++) {
        float x1 = points[old].x;
        float y1 = points[old].y;
        float z1 = points[old].z;

        for (int v = 0; v < voxel_count; v++) {
            Voxel *voxel = &voxels[v];

            profile->voxel_checks++;

            t0 = readCycles();

            float cx = voxel->cx - x1;
            float cy = voxel->cy - y1;
            float cz = voxel->cz - z1;

            float center_dist = sqrtf(cx * cx + cy * cy + cz * cz);
            float lower_bound = center_dist - voxel->radius;

            if (lower_bound < 0.0f)
                lower_bound = 0.0f;

            float lower_bound_sq = lower_bound * lower_bound;
            bool skip_voxel = lower_bound_sq >= voxel->max_dist;

            t1 = readCycles();

            profile->voxel_bound_cycles += t1 - t0;
            profile->voxel_bound_events++;

            if (skip_voxel) {
                profile->voxel_skips++;
                continue;
            }

            profile->active_voxels++;

            t0 = readCycles();

            float new_voxel_max = -1.0f;
            int voxel_best_point = voxel->start;
            int voxel_best_original = points[voxel->start].original_index;
            int end = voxel->start + voxel->count;

            t1 = readCycles();

            profile->voxel_metadata_cycles += t1 - t0;
            profile->voxel_metadata_events++;

            for (int k = voxel->start; k < end; k++) {
                profile->point_checks++;

                float dx2 = 0.0f;
                float dy2 = 0.0f;
                float dz2 = 0.0f;

                bool pass_x = false;
                bool pass_y = false;
                bool pass_z = false;

                t0 = readCycles();

                float dx = points[k].x - x1;
                dx2 = dx * dx;

                if (dx2 < temp[k]) {
                    pass_x = true;

                    float dy = points[k].y - y1;
                    dy2 = dy * dy;

                    if (dy2 < temp[k]) {
                        pass_y = true;

                        float dz = points[k].z - z1;
                        dz2 = dz * dz;

                        if (dz2 < temp[k])
                            pass_z = true;
                    }
                }

                t1 = readCycles();

                profile->point_prune_cycles += t1 - t0;
                profile->point_prune_events++;

                if (!pass_x) {
                    profile->x_prunes++;
                }
                else if (!pass_y) {
                    profile->y_prunes++;
                }
                else if (!pass_z) {
                    profile->z_prunes++;
                }
                else {
                    profile->full_checks++;

                    t0 = readCycles();

                    float d = dx2 + dy2 + dz2;

                    if (d < temp[k])
                        temp[k] = d;

                    t1 = readCycles();

                    profile->full_distance_cycles += t1 - t0;
                    profile->full_distance_events++;
                }

                t0 = readCycles();

                if (temp[k] > new_voxel_max) {
                    new_voxel_max = temp[k];
                    voxel_best_point = k;
                    voxel_best_original = points[k].original_index;
                }
                else if (temp[k] == new_voxel_max) {
                    if (points[k].original_index < voxel_best_original) {
                        voxel_best_point = k;
                        voxel_best_original = points[k].original_index;
                    }
                }

                t1 = readCycles();

                profile->local_best_cycles += t1 - t0;
                profile->local_best_events++;
            }

            t0 = readCycles();

            voxel->max_dist = new_voxel_max;
            voxel->best_point = voxel_best_point;
            voxel->best_original_index = voxel_best_original;

            t1 = readCycles();

            profile->voxel_metadata_cycles += t1 - t0;
            profile->voxel_metadata_events++;
        }

        t0 = readCycles();

        float best = -1.0f;
        int best_point = -1;
        int best_original = -1;

        for (int v = 0; v < voxel_count; v++) {
            if (voxels[v].max_dist > best) {
                best = voxels[v].max_dist;
                best_point = voxels[v].best_point;
                best_original = voxels[v].best_original_index;
            }
            else if (voxels[v].max_dist == best) {
                if (best_original < 0 ||
                    voxels[v].best_original_index < best_original) {
                    best_point = voxels[v].best_point;
                    best_original = voxels[v].best_original_index;
                }
            }
        }

        old = best_point;
        idxs[j] = points[old].original_index;

        t1 = readCycles();

        profile->global_select_cycles += t1 - t0;
        profile->global_select_events++;
    }

    uint64_t total_end = readCycles();
    profile->total_profiled_cycles = total_end - total_start;
}

static bool loadKittiFrame(const char *filename, vector<float> &dataset) {
    ifstream file(filename, ios::binary | ios::ate);

    if (!file) {
        fprintf(stderr, "ERROR: Could not open KITTI frame: %s\n", filename);
        return false;
    }

    streamsize bytes = file.tellg();

    if (bytes <= 0 || bytes % (4 * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: Invalid KITTI frame size: %lld bytes\n",
                static_cast<long long>(bytes));
        return false;
    }

    file.seekg(0, ios::beg);

    size_t point_count = (size_t)bytes / (4 * sizeof(float));
    vector<float> raw(point_count * 4);

    if (!file.read(reinterpret_cast<char *>(raw.data()), bytes)) {
        fprintf(stderr, "ERROR: Failed while reading KITTI frame\n");
        return false;
    }

    dataset.resize(point_count * 3);

    for (size_t i = 0; i < point_count; i++) {
        float x = raw[i * 4 + 0];
        float y = raw[i * 4 + 1];
        float z = raw[i * 4 + 2];

        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
            fprintf(stderr, "ERROR: Non-finite coordinate at point %zu\n", i);
            return false;
        }

        dataset[i * 3 + 0] = x;
        dataset[i * 3 + 1] = y;
        dataset[i * 3 + 2] = z;
    }

    return true;
}

static void printProfile(const SamplingProfile *profile, uint64_t timer_overhead) {
    uint64_t setup = correctedCycles(profile->setup_cycles, 1, timer_overhead);

    uint64_t voxel_bound = correctedCycles(
        profile->voxel_bound_cycles,
        profile->voxel_bound_events,
        timer_overhead
    );

    uint64_t voxel_metadata = correctedCycles(
        profile->voxel_metadata_cycles,
        profile->voxel_metadata_events,
        timer_overhead
    );

    uint64_t point_prune = correctedCycles(
        profile->point_prune_cycles,
        profile->point_prune_events,
        timer_overhead
    );

    uint64_t full_distance = correctedCycles(
        profile->full_distance_cycles,
        profile->full_distance_events,
        timer_overhead
    );

    uint64_t local_best = correctedCycles(
        profile->local_best_cycles,
        profile->local_best_events,
        timer_overhead
    );

    uint64_t global_select = correctedCycles(
        profile->global_select_cycles,
        profile->global_select_events,
        timer_overhead
    );

    uint64_t measured_total =
        setup +
        voxel_bound +
        voxel_metadata +
        point_prune +
        full_distance +
        local_best +
        global_select;

    printf("\n================ SAMPLING PROFILE ================\n");

    printf("PROFILE_TIMER_OVERHEAD=%" PRIu64 "\n", timer_overhead);
    printf("PROFILE_TOTAL_RAW_CYCLES=%" PRIu64 "\n", profile->total_profiled_cycles);
    printf("PROFILE_MEASURED_CORRECTED_CYCLES=%" PRIu64 "\n", measured_total);

    printf("\n");

    printf("PROFILE_SETUP_CYCLES=%" PRIu64 " PCT=%.2f\n",
           setup, percentOf(setup, measured_total));

    printf("PROFILE_VOXEL_BOUND_CYCLES=%" PRIu64 " PCT=%.2f\n",
           voxel_bound, percentOf(voxel_bound, measured_total));

    printf("PROFILE_VOXEL_METADATA_CYCLES=%" PRIu64 " PCT=%.2f\n",
           voxel_metadata, percentOf(voxel_metadata, measured_total));

    printf("PROFILE_POINT_PRUNING_CYCLES=%" PRIu64 " PCT=%.2f\n",
           point_prune, percentOf(point_prune, measured_total));

    printf("PROFILE_FULL_DISTANCE_UPDATE_CYCLES=%" PRIu64 " PCT=%.2f\n",
           full_distance, percentOf(full_distance, measured_total));

    printf("PROFILE_LOCAL_BEST_CYCLES=%" PRIu64 " PCT=%.2f\n",
           local_best, percentOf(local_best, measured_total));

    printf("PROFILE_GLOBAL_SELECTION_CYCLES=%" PRIu64 " PCT=%.2f\n",
           global_select, percentOf(global_select, measured_total));

    printf("\n================ WORK COUNTERS ===================\n");

    printf("VOXEL_CHECKS=%" PRIu64 "\n", profile->voxel_checks);
    printf("VOXEL_SKIPS=%" PRIu64 "\n", profile->voxel_skips);
    printf("ACTIVE_VOXELS=%" PRIu64 "\n", profile->active_voxels);

    printf("POINT_CHECKS=%" PRIu64 "\n", profile->point_checks);
    printf("X_PRUNES=%" PRIu64 "\n", profile->x_prunes);
    printf("Y_PRUNES=%" PRIu64 "\n", profile->y_prunes);
    printf("Z_PRUNES=%" PRIu64 "\n", profile->z_prunes);
    printf("FULL_DISTANCE_CHECKS=%" PRIu64 "\n", profile->full_checks);

    printf("==================================================\n");
}

int main() {
    const char *filename =
        "/mnt/c/Users/lsadi/Documents/PCD Research/datasets/kitti/0000000000.bin";

    const int n = 100000;
    const int m = 128;
    const float voxel_size = 2.0f;

    vector<float> dataset;

    if (!loadKittiFrame(filename, dataset))
        return 1;

    int available_points = (int)(dataset.size() / 3);

    if (available_points < n) {
        fprintf(stderr, "ERROR: Frame has only %d points but N=%d\n",
                available_points, n);
        return 1;
    }

    float min_x = dataset[0];
    float min_y = dataset[1];
    float min_z = dataset[2];

    for (int i = 1; i < n; i++) {
        float x = dataset[i * 3 + 0];
        float y = dataset[i * 3 + 1];
        float z = dataset[i * 3 + 2];

        if (x < min_x)
            min_x = x;

        if (y < min_y)
            min_y = y;

        if (z < min_z)
            min_z = z;
    }

    vector<PointRecord> points((size_t)n);
    vector<PointRecord> scratch((size_t)n);
    vector<Voxel> voxels((size_t)n);
    vector<float> temp((size_t)n);
    vector<int> idxs((size_t)m);

    int voxel_count = buildVoxels(
        dataset.data(),
        n,
        voxel_size,
        points.data(),
        scratch.data(),
        voxels.data(),
        min_x,
        min_y,
        min_z
    );

    uint64_t timer_overhead = measureTimerOverhead();

    SamplingProfile profile = {};

    m5_reset_stats(0, 0);

    radiusFarthestPointSamplingProfile(
        n,
        m,
        points.data(),
        voxels.data(),
        voxel_count,
        temp.data(),
        idxs.data(),
        &profile
    );

    m5_dump_stats(0, 0);

    printf("POINT_COUNT=%d\n", n);
    printf("M=%d\n", m);
    printf("VOXEL_SIZE=%.2f\n", voxel_size);
    printf("VOXEL_COUNT=%d\n", voxel_count);

    printProfile(&profile, timer_overhead);

    printf("\n================ FPS INDICES =====================\n");

    for (int i = 0; i < m; i++)
        printf("FPS_INDEX[%d]=%d\n", i, idxs[i]);

    return 0;
}