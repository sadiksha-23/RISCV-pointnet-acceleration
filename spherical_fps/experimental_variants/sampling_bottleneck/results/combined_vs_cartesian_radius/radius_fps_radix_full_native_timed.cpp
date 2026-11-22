#include <chrono>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

using namespace std;

extern "C" {
    #ifdef ENABLE_GEM5
#include <gem5/m5ops.h>
#else
static inline void m5_reset_stats(uint64_t, uint64_t) {}
static inline void m5_dump_stats(uint64_t, uint64_t) {}
#endif
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

static void radixSortPoints(PointRecord *points, int n) {
    vector<PointRecord> buffer((size_t)n);

    uint64_t max_key = 0;
    for (int i = 0; i < n; i++) {
        if (points[i].voxel_key > max_key) max_key = points[i].voxel_key;
    }

    PointRecord *source = points;
    PointRecord *destination = buffer.data();

    for (int shift = 0; shift < 64; shift += 8) {
        if ((max_key >> shift) == 0) break;

        size_t count[256] = {0};
        size_t position[256];

        for (int i = 0; i < n; i++) {
            unsigned int bucket = (unsigned int)((source[i].voxel_key >> shift) & 0xFFULL);
            count[bucket]++;
        }

        position[0] = 0;
        for (int i = 1; i < 256; i++) position[i] = position[i - 1] + count[i - 1];

        for (int i = 0; i < n; i++) {
            unsigned int bucket = (unsigned int)((source[i].voxel_key >> shift) & 0xFFULL);
            destination[position[bucket]++] = source[i];
        }

        PointRecord *swap_pointer = source;
        source = destination;
        destination = swap_pointer;
    }

    if (source != points) {
        for (int i = 0; i < n; i++) points[i] = source[i];
    }
}

static int buildVoxels(const float *dataset, int n, float voxel_size, PointRecord *points, Voxel *voxels, float min_x, float min_y, float min_z) {
    int max_vy = 0;
    int max_vz = 0;

    // STAGE 3: voxel coordinate mapping
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

        if (points[i].vy > max_vy) max_vy = points[i].vy;
        if (points[i].vz > max_vz) max_vz = points[i].vz;
    }


    // STAGE 4: voxel key generation
    uint64_t ny = (uint64_t)max_vy + 1ULL;
    uint64_t nz = (uint64_t)max_vz + 1ULL;

    for (int i = 0; i < n; i++) {
        points[i].voxel_key = ((uint64_t)points[i].vx * ny + (uint64_t)points[i].vy) * nz + (uint64_t)points[i].vz;
    }


    // STAGE 5: radix sort
    radixSortPoints(points, n);


    // STAGE 6: voxel metadata construction
    int voxel_count = 0;
    int start = 0;

    while (start < n) {
        int vx = points[start].vx;
        int vy = points[start].vy;
        int vz = points[start].vz;

        int end = start + 1;

        while (end < n && points[end].vx == vx && points[end].vy == vy && points[end].vz == vz) end++;

        Voxel *v = &voxels[voxel_count];

        (*v).start = start;
        (*v).count = end - start;

        (*v).cx = min_x + ((float)vx + 0.5f) * voxel_size;
        (*v).cy = min_y + ((float)vy + 0.5f) * voxel_size;
        (*v).cz = min_z + ((float)vz + 0.5f) * voxel_size;

        (*v).radius = 0.5f * sqrtf(3.0f) * voxel_size;
        (*v).max_dist = 1e38f;
        (*v).best_point = start;
        (*v).best_original_index = points[start].original_index;

        voxel_count++;
        start = end;
    }


    return voxel_count;
}

void radiusFarthestPointSampling(int n, int m, PointRecord *points, Voxel *voxels, int voxel_count, float *temp, int *idxs) {
    if (m <= 0) return;

    for (int i = 0; i < n; i++) temp[i] = 1e38f;

    int old = 0;

    for (int i = 0; i < n; i++) {
        if (points[i].original_index == 0) {
            old = i;
            break;
        }
    }

    idxs[0] = 0;

    for (int j = 1; j < m; j++) {
        float x1 = points[old].x;
        float y1 = points[old].y;
        float z1 = points[old].z;

        for (int v = 0; v < voxel_count; v++) {
            Voxel *voxel = &voxels[v];

            float cx = (*voxel).cx - x1;
            float cy = (*voxel).cy - y1;
            float cz = (*voxel).cz - z1;

            float center_dist = sqrtf(cx * cx + cy * cy + cz * cz);
            float lower_bound = center_dist - (*voxel).radius;

            if (lower_bound < 0.0f) lower_bound = 0.0f;

            float lower_bound_sq = lower_bound * lower_bound;

            if (lower_bound_sq >= (*voxel).max_dist) continue;

            float new_voxel_max = -1.0f;
            int voxel_best_point = (*voxel).start;
            int voxel_best_original = points[(*voxel).start].original_index;
            int end = (*voxel).start + (*voxel).count;

            for (int k = (*voxel).start; k < end; k++) {
                float dx = points[k].x - x1;
                float dx2 = dx * dx;

                if (dx2 < temp[k]) {
                    float dy = points[k].y - y1;
                    float dy2 = dy * dy;

                    if (dy2 < temp[k]) {
                        float dz = points[k].z - z1;
                        float dz2 = dz * dz;

                        if (dz2 < temp[k]) {
                            float d = dx2 + dy2 + dz2;
                            if (d < temp[k]) temp[k] = d;
                        }
                    }
                }

                if (temp[k] > new_voxel_max) {
                    new_voxel_max = temp[k];
                    voxel_best_point = k;
                    voxel_best_original = points[k].original_index;
                } else if (temp[k] == new_voxel_max && points[k].original_index < voxel_best_original) {
                    voxel_best_point = k;
                    voxel_best_original = points[k].original_index;
                }
            }

            (*voxel).max_dist = new_voxel_max;
            (*voxel).best_point = voxel_best_point;
            (*voxel).best_original_index = voxel_best_original;
        }

        float best = -1.0f;
        int best_point = -1;
        int best_original = -1;

        for (int v = 0; v < voxel_count; v++) {
            if (voxels[v].max_dist > best) {
                best = voxels[v].max_dist;
                best_point = voxels[v].best_point;
                best_original = voxels[v].best_original_index;
            } else if (voxels[v].max_dist == best && (best_original < 0 || voxels[v].best_original_index < best_original)) {
                best_point = voxels[v].best_point;
                best_original = voxels[v].best_original_index;
            }
        }

        old = best_point;
        idxs[j] = points[old].original_index;
    }
}

static bool loadKittiFrame(const char *filename, vector<float> &dataset) {
    ifstream file(filename, ios::binary | ios::ate);

    if (!file) {
        fprintf(stderr, "ERROR: Could not open KITTI frame: %s\n", filename);
        return false;
    }

    streamsize bytes = file.tellg();

    if (bytes <= 0 || bytes % (4 * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: Invalid KITTI frame size\n");
        return false;
    }

    file.seekg(0, ios::beg);

    size_t point_count = (size_t)bytes / (4 * sizeof(float));
    vector<float> raw(point_count * 4);

    if (!file.read(reinterpret_cast<char *>(raw.data()), bytes)) return false;

    dataset.resize(point_count * 3);

    for (size_t i = 0; i < point_count; i++) {
        dataset[i * 3 + 0] = raw[i * 4 + 0];
        dataset[i * 3 + 1] = raw[i * 4 + 1];
        dataset[i * 3 + 2] = raw[i * 4 + 2];
    }

    return true;
}


int main(int argc, char **argv) {

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <KITTI frame> <N> <M>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int n = atoi(argv[2]);
    int m = atoi(argv[3]);

    const float voxel_size = 2.0f;

    vector<float> dataset;

    if (!loadKittiFrame(filename, dataset))
        return 1;

    if (n < 1 || m < 1 || m > n ||
        n > (int)(dataset.size() / 3)) {
        fprintf(stderr, "ERROR: Invalid N or M\n");
        return 1;
    }

    // Full algorithm timing begins.
    // Dataset loading is excluded.

    m5_reset_stats(0, 0);
    const auto prep_start = std::chrono::steady_clock::now();

    float min_x = dataset[0];
    float min_y = dataset[1];
    float min_z = dataset[2];

    for (int i = 1; i < n; i++) {
        float x = dataset[i * 3];
        float y = dataset[i * 3 + 1];
        float z = dataset[i * 3 + 2];

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (z < min_z) min_z = z;
    }

    vector<PointRecord> points((size_t)n);
    vector<Voxel> voxels((size_t)n);
    vector<float> temp((size_t)n);
    vector<int> idxs((size_t)m);

    int voxel_count = buildVoxels(
        dataset.data(),
        n,
        voxel_size,
        points.data(),
        voxels.data(),
        min_x,
        min_y,
        min_z
    );

    const auto prep_end = std::chrono::steady_clock::now();
    const auto sample_start = prep_end;
    radiusFarthestPointSampling(
        n, m,
        points.data(),
        voxels.data(),
        voxel_count,
        temp.data(),
        idxs.data()
    );

    const auto sample_end = std::chrono::steady_clock::now();
    m5_dump_stats(0, 0);

    // Results printed after the timing interval.

    const double prep_ms = std::chrono::duration<double, std::milli>(prep_end - prep_start).count();
    const double sample_ms = std::chrono::duration<double, std::milli>(sample_end - sample_start).count();
    printf("PREPARATION_MS=%.6f\nSAMPLING_MS=%.6f\nTOTAL_MS=%.6f\n", prep_ms, sample_ms, prep_ms + sample_ms);
    printf("ALGORITHM=RadiusFPS_Radix\n");
    printf("N=%d\n", n);
    printf("M=%d\n", m);
    printf("VOXEL_SIZE=%.2f\n", voxel_size);
    printf("VOXEL_COUNT=%d\n", voxel_count);

    for (int i = 0; i < m; i++)
        printf("FPS_INDEX[%d]=%d\n", i, idxs[i]);

    printf("RUN_STATUS=PASS\n");

    return 0;
}
