#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

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


static inline float vectorMinFloat(vfloat32m2_t values, size_t vl) {
    vfloat32m1_t initial = __riscv_vfmv_v_f_f32m1(FLT_MAX, 1);
    vfloat32m1_t reduced = __riscv_vfredmin_vs_f32m2_f32m1(values, initial, vl);
    return __riscv_vfmv_f_s_f32m1_f32(reduced);
}

static inline int vectorMaxInt(vint32m2_t values, size_t vl) {
    vint32m1_t initial = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t reduced = __riscv_vredmax_vs_i32m2_i32m1(values, initial, vl);
    return __riscv_vmv_x_s_i32m1_i32(reduced);
}

static inline void findVectorMaximum(vfloat32m2_t values, size_t vl,
                                     float &maximum_value, int &maximum_lane) {
    vfloat32m1_t initial = __riscv_vfmv_v_f_f32m1(-1.0f, 1);
    vfloat32m1_t reduced = __riscv_vfredmax_vs_f32m2_f32m1(values, initial, vl);
    maximum_value = __riscv_vfmv_f_s_f32m1_f32(reduced);
    vbool16_t mask = __riscv_vmfeq_vf_f32m2_b16(values, maximum_value, vl);
    maximum_lane = (int)__riscv_vfirst_m_b16(mask, vl);
}


static void radixSortPoints(PointRecord *points, int n) {
    vector<PointRecord> buffer((size_t)n);

    uint64_t max_key = 0;

    for (int i = 0; i < n; i++) {
        if (points[i].voxel_key > max_key)
            max_key = points[i].voxel_key;
    }

    PointRecord *source = points;
    PointRecord *destination = buffer.data();

    for (int shift = 0; shift < 64; shift += 8) {
        if ((max_key >> shift) == 0)
            break;

        size_t count[256] = {0};
        size_t position[256];

        for (int i = 0; i < n; i++) {
            unsigned int bucket = (unsigned int)((source[i].voxel_key >> shift) & 0xFFULL);
            count[bucket]++;
        }

        position[0] = 0;

        for (int i = 1; i < 256; i++)
            position[i] = position[i - 1] + count[i - 1];

        for (int i = 0; i < n; i++) {
            unsigned int bucket = (unsigned int)((source[i].voxel_key >> shift) & 0xFFULL);
            destination[position[bucket]++] = source[i];
        }

        PointRecord *swap_pointer = source;
        source = destination;
        destination = swap_pointer;
    }

    if (source != points) {
        for (int i = 0; i < n; i++)
            points[i] = source[i];
    }
}

static int buildVoxels(const float *dataset, int n, float voxel_size, PointRecord *points,
                       Voxel *voxels, float min_x, float min_y, float min_z) {

    const ptrdiff_t dataset_stride = 3 * sizeof(float);
    m5_reset_stats(0, 0);
    const ptrdiff_t point_stride = sizeof(PointRecord);

    int max_vy = 0;
    int max_vz = 0;

    // =========================================================
    // STAGE 3: RVV VOXEL COORDINATE MAPPING
    // =========================================================

    int remaining = n;
    int base = 0;

    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m2(remaining);

        // Load interleaved KITTI x/y/z coordinates.
        vfloat32m2_t vx = __riscv_vlse32_v_f32m2(dataset + base * 3 + 0, dataset_stride, vl);
        vfloat32m2_t vy = __riscv_vlse32_v_f32m2(dataset + base * 3 + 1, dataset_stride, vl);
        vfloat32m2_t vz = __riscv_vlse32_v_f32m2(dataset + base * 3 + 2, dataset_stride, vl);

        // Save coordinates into PointRecord array.
        __riscv_vsse32_v_f32m2(&points[base].x, point_stride, vx, vl);
        __riscv_vsse32_v_f32m2(&points[base].y, point_stride, vy, vl);
        __riscv_vsse32_v_f32m2(&points[base].z, point_stride, vz, vl);

        // Generate original point indices.
        vuint32m2_t lane_numbers = __riscv_vid_v_u32m2(vl);
        vuint32m2_t original_indices =
            __riscv_vadd_vx_u32m2(lane_numbers, (uint32_t)base, vl);

        __riscv_vsse32_v_u32m2(
            reinterpret_cast<uint32_t *>(&points[base].original_index),
            point_stride,
            original_indices,
            vl
        );

        // Translate coordinates so all values are nonnegative.
        vfloat32m2_t vdx = __riscv_vfsub_vf_f32m2(vx, min_x, vl);
        vfloat32m2_t vdy = __riscv_vfsub_vf_f32m2(vy, min_y, vl);
        vfloat32m2_t vdz = __riscv_vfsub_vf_f32m2(vz, min_z, vl);

        // Same calculation as scalar:
        // floor((coordinate - minimum) / voxel_size)
        //
        // Values are nonnegative, so RTZ integer conversion equals floor.
        vdx = __riscv_vfdiv_vf_f32m2(vdx, voxel_size, vl);
        vdy = __riscv_vfdiv_vf_f32m2(vdy, voxel_size, vl);
        vdz = __riscv_vfdiv_vf_f32m2(vdz, voxel_size, vl);

        vint32m2_t voxel_x = __riscv_vfcvt_rtz_x_f_v_i32m2(vdx, vl);
        vint32m2_t voxel_y = __riscv_vfcvt_rtz_x_f_v_i32m2(vdy, vl);
        vint32m2_t voxel_z = __riscv_vfcvt_rtz_x_f_v_i32m2(vdz, vl);

        __riscv_vsse32_v_i32m2(&points[base].vx, point_stride, voxel_x, vl);
        __riscv_vsse32_v_i32m2(&points[base].vy, point_stride, voxel_y, vl);
        __riscv_vsse32_v_i32m2(&points[base].vz, point_stride, voxel_z, vl);

        int chunk_max_y = vectorMaxInt(voxel_y, vl);
        int chunk_max_z = vectorMaxInt(voxel_z, vl);

        if (chunk_max_y > max_vy)
            max_vy = chunk_max_y;

        if (chunk_max_z > max_vz)
            max_vz = chunk_max_z;

        base += (int)vl;
        remaining -= (int)vl;
    }

    
    

    // =========================================================
    // STAGE 4: RVV VOXEL KEY GENERATION
    // =========================================================

    uint64_t ny = (uint64_t)max_vy + 1ULL;
    uint64_t nz = (uint64_t)max_vz + 1ULL;

    remaining = n;
    base = 0;

    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);

        vint32m1_t voxel_x =
            __riscv_vlse32_v_i32m1(&points[base].vx, point_stride, vl);

        vint32m1_t voxel_y =
            __riscv_vlse32_v_i32m1(&points[base].vy, point_stride, vl);

        vint32m1_t voxel_z =
            __riscv_vlse32_v_i32m1(&points[base].vz, point_stride, vl);

        // Widen 32-bit voxel coordinates to 64-bit.
        vint64m2_t vx64 = __riscv_vwcvt_x_x_v_i64m2(voxel_x, vl);
        vint64m2_t vy64 = __riscv_vwcvt_x_x_v_i64m2(voxel_y, vl);
        vint64m2_t vz64 = __riscv_vwcvt_x_x_v_i64m2(voxel_z, vl);

        // key = ((vx * ny) + vy) * nz + vz
        vint64m2_t key =
            __riscv_vmul_vx_i64m2(vx64, (int64_t)ny, vl);

        key = __riscv_vadd_vv_i64m2(key, vy64, vl);
        key = __riscv_vmul_vx_i64m2(key, (int64_t)nz, vl);
        key = __riscv_vadd_vv_i64m2(key, vz64, vl);

        __riscv_vsse64_v_i64m2(
            reinterpret_cast<int64_t *>(&points[base].voxel_key),
            point_stride,
            key,
            vl
        );

        base += (int)vl;
        remaining -= (int)vl;
    }

    
    

    // =========================================================

    // STAGE 5: RADIX SORT
    //
    // Kept scalar because histogram updates and scatter positions
    // contain loop-carried/data-dependent operations.
    // =========================================================

    radixSortPoints(points, n);

    
    

    // =========================================================
    // STAGE 6: VOXEL METADATA CONSTRUCTION
    //
    // Run boundaries are data-dependent, so this remains scalar.
    // =========================================================

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

static void radiusFarthestPointSamplingRVV(
    int n,
    int m,
    PointRecord *points,
    Voxel *voxels,
    int voxel_count,
    float *temp,
    int *idxs,
    int *active_voxels
) {
    if (m <= 0)
        return;

    const ptrdiff_t point_stride = sizeof(PointRecord);
    const ptrdiff_t voxel_stride = sizeof(Voxel);

    // =========================================================
    // RVV TEMP INITIALIZATION
    // =========================================================

    int remaining = n;
    int base = 0;

    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m2(remaining);

        vfloat32m2_t infinity =
            __riscv_vfmv_v_f_f32m2(1e38f, vl);

        __riscv_vse32_v_f32m2(temp + base, infinity, vl);

        base += (int)vl;
        remaining -= (int)vl;
    }

    // =========================================================
    // FIND SORTED POSITION OF ORIGINAL POINT 0 WITH RVV
    // =========================================================

    int old = 0;
    bool found = false;

    remaining = n;
    base = 0;

    while (remaining > 0 && !found) {
        size_t vl = __riscv_vsetvl_e32m2(remaining);

        vint32m2_t original_indices =
            __riscv_vlse32_v_i32m2(
                &points[base].original_index,
                point_stride,
                vl
            );

        vbool16_t match =
            __riscv_vmseq_vx_i32m2_b16(original_indices, 0, vl);

        long lane = __riscv_vfirst_m_b16(match, vl);

        if (lane >= 0) {
            old = base + (int)lane;
            found = true;
            break;
        }

        base += (int)vl;
        remaining -= (int)vl;
    }

    idxs[0] = 0;

    // =========================================================
    // FPS ITERATIONS
    // =========================================================

    for (int j = 1; j < m; j++) {
        float x1 = points[old].x;
        float y1 = points[old].y;
        float z1 = points[old].z;

        // -----------------------------------------------------
        // RVV VOXEL LOWER-BOUND TEST
        //
        // Instead of scalar sqrt/bound logic for every voxel,
        // calculate bounds for many voxels at once.
        // -----------------------------------------------------

        int active_count = 0;
        int voxel_base = 0;
        int remaining_voxels = voxel_count;

        while (remaining_voxels > 0) {
            size_t vl = __riscv_vsetvl_e32m2(remaining_voxels);

            vfloat32m2_t vcx =
                __riscv_vlse32_v_f32m2(
                    &voxels[voxel_base].cx,
                    voxel_stride,
                    vl
                );

            vfloat32m2_t vcy =
                __riscv_vlse32_v_f32m2(
                    &voxels[voxel_base].cy,
                    voxel_stride,
                    vl
                );

            vfloat32m2_t vcz =
                __riscv_vlse32_v_f32m2(
                    &voxels[voxel_base].cz,
                    voxel_stride,
                    vl
                );

            vfloat32m2_t vradius =
                __riscv_vlse32_v_f32m2(
                    &voxels[voxel_base].radius,
                    voxel_stride,
                    vl
                );

            vfloat32m2_t vmax_dist =
                __riscv_vlse32_v_f32m2(
                    &voxels[voxel_base].max_dist,
                    voxel_stride,
                    vl
                );

            vcx = __riscv_vfsub_vf_f32m2(vcx, x1, vl);
            vcy = __riscv_vfsub_vf_f32m2(vcy, y1, vl);
            vcz = __riscv_vfsub_vf_f32m2(vcz, z1, vl);

            vfloat32m2_t vcx2 =
                __riscv_vfmul_vv_f32m2(vcx, vcx, vl);

            vfloat32m2_t vcy2 =
                __riscv_vfmul_vv_f32m2(vcy, vcy, vl);

            vfloat32m2_t vcz2 =
                __riscv_vfmul_vv_f32m2(vcz, vcz, vl);

            vfloat32m2_t center_sq =
                __riscv_vfadd_vv_f32m2(vcx2, vcy2, vl);

            center_sq =
                __riscv_vfadd_vv_f32m2(center_sq, vcz2, vl);

            vfloat32m2_t center_dist =
                __riscv_vfsqrt_v_f32m2(center_sq, vl);

            vfloat32m2_t lower_bound =
                __riscv_vfsub_vv_f32m2(center_dist, vradius, vl);

            lower_bound =
                __riscv_vfmax_vf_f32m2(lower_bound, 0.0f, vl);

            vfloat32m2_t lower_bound_sq =
                __riscv_vfmul_vv_f32m2(
                    lower_bound,
                    lower_bound,
                    vl
                );

            // Scalar rule:
            // skip if lower_bound_sq >= max_dist
            //
            // Therefore voxel is active when:
            // lower_bound_sq < max_dist
            vbool16_t active_mask =
                __riscv_vmflt_vv_f32m2_b16(
                    lower_bound_sq,
                    vmax_dist,
                    vl
                );

            size_t matches =
                __riscv_vcpop_m_b16(active_mask, vl);

            if (matches > 0) {
                vuint32m2_t lane_numbers =
                    __riscv_vid_v_u32m2(vl);

                vuint32m2_t voxel_indices =
                    __riscv_vadd_vx_u32m2(
                        lane_numbers,
                        (uint32_t)voxel_base,
                        vl
                    );

                vuint32m2_t compressed =
                    __riscv_vcompress_vm_u32m2(
                        voxel_indices,
                        active_mask,
                        vl
                    );

                __riscv_vse32_v_u32m2(
                    reinterpret_cast<uint32_t *>(
                        active_voxels + active_count
                    ),
                    compressed,
                    matches
                );

                active_count += (int)matches;
            }

            voxel_base += (int)vl;
            remaining_voxels -= (int)vl;
        }

        // -----------------------------------------------------
        // PROCESS ONLY ACTIVE VOXELS
        // -----------------------------------------------------

        for (int av = 0; av < active_count; av++) {
            int v = active_voxels[av];
            Voxel *voxel = &voxels[v];

            float new_voxel_max = -1.0f;
            int voxel_best_point = voxel->start;
            int voxel_best_original =
                points[voxel->start].original_index;

            int point_base = voxel->start;
            int remaining_points = voxel->count;

            // -------------------------------------------------
            // RVV POINT PROCESSING
            //
            // Scalar RadiusFPS used:
            // X pruning -> Y pruning -> Z pruning -> distance
            //
            // RVV version calculates full d² vectorially and:
            //
            //     temp = min(temp, d²)
            //
            // This is mathematically equivalent to the scalar
            // pruning result but removes branch-heavy control flow.
            // -------------------------------------------------

            while (remaining_points > 0) {
                size_t vl =
                    __riscv_vsetvl_e32m2(remaining_points);

                vfloat32m2_t vx =
                    __riscv_vlse32_v_f32m2(
                        &points[point_base].x,
                        point_stride,
                        vl
                    );

                vfloat32m2_t vy =
                    __riscv_vlse32_v_f32m2(
                        &points[point_base].y,
                        point_stride,
                        vl
                    );

                vfloat32m2_t vz =
                    __riscv_vlse32_v_f32m2(
                        &points[point_base].z,
                        point_stride,
                        vl
                    );

                vfloat32m2_t vdx =
                    __riscv_vfsub_vf_f32m2(vx, x1, vl);

                vfloat32m2_t vdy =
                    __riscv_vfsub_vf_f32m2(vy, y1, vl);

                vfloat32m2_t vdz =
                    __riscv_vfsub_vf_f32m2(vz, z1, vl);

                vfloat32m2_t vdx2 =
                    __riscv_vfmul_vv_f32m2(vdx, vdx, vl);

                vfloat32m2_t vdy2 =
                    __riscv_vfmul_vv_f32m2(vdy, vdy, vl);

                vfloat32m2_t vdz2 =
                    __riscv_vfmul_vv_f32m2(vdz, vdz, vl);

                // Keep scalar-style arithmetic ordering:
                // (dx² + dy²) + dz²
                vfloat32m2_t distance =
                    __riscv_vfadd_vv_f32m2(
                        vdx2,
                        vdy2,
                        vl
                    );

                distance =
                    __riscv_vfadd_vv_f32m2(
                        distance,
                        vdz2,
                        vl
                    );

                vfloat32m2_t current_temp =
                    __riscv_vle32_v_f32m2(
                        temp + point_base,
                        vl
                    );

                vfloat32m2_t updated_temp =
                    __riscv_vfmin_vv_f32m2(
                        current_temp,
                        distance,
                        vl
                    );

                __riscv_vse32_v_f32m2(
                    temp + point_base,
                    updated_temp,
                    vl
                );

                // ---------------------------------------------
                // RVV LOCAL ARGMAX
                //
                // Find the farthest point in this vector chunk.
                // ---------------------------------------------

                float chunk_best;
                int chunk_best_lane;

                findVectorMaximum(
                    updated_temp,
                    vl,
                    chunk_best,
                    chunk_best_lane
                );

                int chunk_best_point =
                    point_base + chunk_best_lane;

                int chunk_best_original =
                    points[chunk_best_point].original_index;

                if (chunk_best > new_voxel_max) {
                    new_voxel_max = chunk_best;
                    voxel_best_point = chunk_best_point;
                    voxel_best_original = chunk_best_original;
                }
                else if (chunk_best == new_voxel_max &&
                         chunk_best_original < voxel_best_original) {
                    voxel_best_point = chunk_best_point;
                    voxel_best_original = chunk_best_original;
                }

                point_base += (int)vl;
                remaining_points -= (int)vl;
            }

            voxel->max_dist = new_voxel_max;
            voxel->best_point = voxel_best_point;
            voxel->best_original_index = voxel_best_original;
        }

        // -----------------------------------------------------
        // GLOBAL WINNER
        //
        // Only ~937 voxels and this was ~1% of scalar sampling,
        // so leave this scalar for now.
        // -----------------------------------------------------

        float best = -1.0f;
        int best_point = -1;
        int best_original = -1;

        for (int v = 0; v < voxel_count; v++) {
            if (voxels[v].max_dist > best) {
                best = voxels[v].max_dist;
                best_point = voxels[v].best_point;
                best_original = voxels[v].best_original_index;
            }
            else if (voxels[v].max_dist == best &&
                     (best_original < 0 ||
                      voxels[v].best_original_index < best_original)) {
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

    size_t point_count =
        (size_t)bytes / (4 * sizeof(float));

    vector<float> raw(point_count * 4);

    if (!file.read(
            reinterpret_cast<char *>(raw.data()),
            bytes)) {
        return false;
    }

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
        fprintf(stderr, "Usage: %s <KITTI> <N> <M>\n", argv[0]);
        return 2;
    }
    vector<float> dataset;

    const char *filename = argv[1];

    if (!loadKittiFrame(filename, dataset))
        return 1;

    int n = atoi(argv[2]);
    int m = atoi(argv[3]);
    float voxel_size = 2.0f;
    if (n < 1 || m < 1 || m > n) {
        fprintf(stderr, "Invalid N or M\n");
        return 2;
    }

    if ((int)(dataset.size() / 3) < n) {
        fprintf(stderr, "ERROR: Not enough KITTI points\n");
        return 1;
    }

    const ptrdiff_t dataset_stride = 3 * sizeof(float);

    // =========================================================
    // STAGE 1: RVV MIN COORDINATE SCAN
    // =========================================================

    

    float min_x = FLT_MAX;
    float min_y = FLT_MAX;
    float min_z = FLT_MAX;

    int remaining = n;
    int base = 0;

    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m2(remaining);

        vfloat32m2_t vx =
            __riscv_vlse32_v_f32m2(
                dataset.data() + base * 3 + 0,
                dataset_stride,
                vl
            );

        vfloat32m2_t vy =
            __riscv_vlse32_v_f32m2(
                dataset.data() + base * 3 + 1,
                dataset_stride,
                vl
            );

        vfloat32m2_t vz =
            __riscv_vlse32_v_f32m2(
                dataset.data() + base * 3 + 2,
                dataset_stride,
                vl
            );

        float chunk_min_x = vectorMinFloat(vx, vl);
        float chunk_min_y = vectorMinFloat(vy, vl);
        float chunk_min_z = vectorMinFloat(vz, vl);

        if (chunk_min_x < min_x)
            min_x = chunk_min_x;

        if (chunk_min_y < min_y)
            min_y = chunk_min_y;

        if (chunk_min_z < min_z)
            min_z = chunk_min_z;

        base += (int)vl;
        remaining -= (int)vl;
    }

    
    

    // =========================================================
    // STAGE 2: ALLOCATIONS
    // =========================================================

    vector<PointRecord> points((size_t)n);
    vector<Voxel> voxels((size_t)n);
    vector<float> temp((size_t)n);
    vector<int> idxs((size_t)m);

    // Extra workspace needed for compacted active voxel IDs.
    vector<int> active_voxels((size_t)n);

    
    

    int voxel_count =
        buildVoxels(
            dataset.data(),
            n,
            voxel_size,
            points.data(),
            voxels.data(),
            min_x,
            min_y,
            min_z
        );

    // =========================================================
    // STAGE 7: RVV RADIUSFPS SAMPLING
    // =========================================================

    radiusFarthestPointSamplingRVV(
        n,
        m,
        points.data(),
        voxels.data(),
        voxel_count,
        temp.data(),
        idxs.data(),
        active_voxels.data()
    );

    

    m5_dump_stats(0, 0);
    printf("POINT_COUNT=%d\n", n);
    printf("M=%d\n", m);
    printf("VOXEL_SIZE=%.2f\n", voxel_size);
    printf("VOXEL_COUNT=%d\n", voxel_count);

    // Print all indices so we can validate exactness against scalar.
    for (int i = 0; i < m; i++)
        printf("FPS_INDEX[%d]=%d\n", i, idxs[i]);

    return 0;
}