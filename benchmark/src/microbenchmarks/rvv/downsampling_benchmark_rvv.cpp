#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// 1. Vectorized Farthest Point Sampling (FPS)
static inline void find_vector_maximum(vfloat32m1_t values, size_t vl, float &maximum_value, int &maximum_lane) {
    // FPS distances are nonnegative, so -1 is a safe starting value.
    vfloat32m1_t initial = __riscv_vfmv_v_f_f32m1(-1.0f, 1);

    // Find the largest value across all active vector lanes.
    vfloat32m1_t reduced = __riscv_vfredmax_vs_f32m1_f32m1(values, initial, vl);
    maximum_value = __riscv_vfmv_f_s_f32m1_f32(reduced);

    // Mark all lanes containing the maximum value.
    vbool32_t maximum_mask = __riscv_vmfeq_vf_f32m1_b32(values, maximum_value, vl);

    // Select the first lane containing that maximum.
    maximum_lane = static_cast<int>(__riscv_vfirst_m_b32(maximum_mask, vl));
}

void farthestPointSampling_rvv(int b, int n, int m, const float * __restrict__ dataset, float * __restrict__ temp, int * __restrict__ idxs) {
    if (m <= 0) return;

    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        int old = 0;
        idxs[0] = old;

        // Initialize the minimum-distance buffer.
        int rem_init = n;
        int init_offset = 0;

        for (size_t vl; rem_init > 0; rem_init -= vl, init_offset += vl) {
            vl = __riscv_vsetvl_e32m1(rem_init);
            vfloat32m1_t v_inf = __riscv_vfmv_v_f_f32m1(1e38f, vl);
            __riscv_vse32_v_f32m1(temp + init_offset, v_inf, vl);
        }

        const float *cur_dataset = dataset + i * n * 3;

        // Select the remaining m - 1 centroid points.
        for (int j = 1; j < m; ++j) {
            int besti = 0;
            float best = -1.0f;

            // Coordinates of the previously selected centroid.
            float x1 = cur_dataset[old * 3 + 0];
            float y1 = cur_dataset[old * 3 + 1];
            float z1 = cur_dataset[old * 3 + 2];

            int rem_n = n;
            int k_base = 0;

            // Process candidate points in RVV chunks.
            for (size_t vl; rem_n > 0; rem_n -= vl, k_base += vl) {
                vl = __riscv_vsetvl_e32m1(rem_n);

                // Load candidate x, y and z coordinates.
                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 2, stride, vl);

                // Calculate coordinate differences.
                vfloat32m1_t vdx = __riscv_vfsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfsub_vf_f32m1(vz2, z1, vl);

                // Calculate squared Euclidean distances.
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Update each point's minimum distance to a selected centroid.
                vfloat32m1_t v_temp = __riscv_vle32_v_f32m1(temp + k_base, vl);
                v_temp = __riscv_vfmin_vv_f32m1(v_temp, vd, vl);
                __riscv_vse32_v_f32m1(temp + k_base, v_temp, vl);

                // Find the maximum updated distance and its lane using RVV.
                float chunk_best;
                int chunk_best_lane;
                find_vector_maximum(v_temp, vl, chunk_best, chunk_best_lane);

                // Compare this chunk's winner with the overall winner.
                if (chunk_best > best) {
                    best = chunk_best;
                    besti = k_base + chunk_best_lane;
                }
            }

            // The farthest point becomes the next centroid.
            old = besti;
            idxs[j] = old;
        }

        temp += n;
        idxs += m;
    }
}

// 2. Vectorized Point Gathering
void gatherPoint_rvv(int b, int n, int m, const float * __restrict__ inp, const int * __restrict__ idx, float * __restrict__ out) {
    for (int i = 0; i < b; ++i) {
        const float *cur_inp = inp + i * n * 3;
        const int   *cur_idx = idx + i * m;
        float       *cur_out = out + i * m * 3;

        for (int j = 0; j < m; ++j) {
            int a = cur_idx[j];
            size_t vl = __riscv_vsetvl_e32m1(3);
            vfloat32m1_t v_coords = __riscv_vle32_v_f32m1(cur_inp + a * 3, vl);
            __riscv_vse32_v_f32m1(cur_out + j * 3, v_coords, vl);
        }
    }
}

// 3. Vectorized Ball Query
static inline void fill_indices_rvv(int *output, int count, int value) {
    int remaining = count;
    int offset = 0;

    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t values = __riscv_vmv_v_x_u32m1(static_cast<uint32_t>(value), vl);
        __riscv_vse32_v_u32m1(reinterpret_cast<uint32_t *>(output + offset), values, vl);

        remaining -= static_cast<int>(vl);
        offset += static_cast<int>(vl);
    }
}

void query_ball_point_rvv(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    float radius2 = radius * radius;
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            int cnt = 0;
            int *current_output = idx + (i * m + j) * nsample;

            float x2 = xyz2[(i * m + j) * 3 + 0];
            float y2 = xyz2[(i * m + j) * 3 + 1];
            float z2 = xyz2[(i * m + j) * 3 + 2];

            const float *curr_xyz1 = xyz1 + i * n * 3;
            int k_base = 0;
            int rem_n = n;

            // Process candidate points in RVV chunks.
            for (size_t vl; rem_n > 0; rem_n -= vl, k_base += vl, curr_xyz1 += vl * 3) {
                if (cnt == nsample) break;

                vl = __riscv_vsetvl_e32m1(rem_n);

                // Load candidate coordinates.
                vfloat32m1_t vx1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 0, stride, vl);
                vfloat32m1_t vy1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 1, stride, vl);
                vfloat32m1_t vz1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 2, stride, vl);

                // Calculate coordinate differences.
                vfloat32m1_t vdx = __riscv_vfrsub_vf_f32m1(vx1, x2, vl);
                vfloat32m1_t vdy = __riscv_vfrsub_vf_f32m1(vy1, y2, vl);
                vfloat32m1_t vdz = __riscv_vfrsub_vf_f32m1(vz1, z2, vl);

                // Calculate squared distances.
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Mark lanes whose distance is smaller than radius².
                vbool32_t within_radius = __riscv_vmflt_vf_f32m1_b32(vd, radius2, vl);

                // Count matching lanes.
                size_t match_count = __riscv_vcpop_m_b32(within_radius, vl);
                if (match_count == 0) continue;

                // Generate complete candidate indices for this chunk.
                vuint32m1_t lane_numbers = __riscv_vid_v_u32m1(vl);
                vuint32m1_t candidate_indices = __riscv_vadd_vx_u32m1(lane_numbers, static_cast<uint32_t>(k_base), vl);

                // Move all matching indices to the beginning of the vector.
                vuint32m1_t matching_indices = __riscv_vcompress_vm_u32m1(candidate_indices, within_radius, vl);

                // Preserve PointNet++ behavior by filling empty positions with the first match.
                if (cnt == 0) {
                    uint32_t first_index = __riscv_vmv_x_s_u32m1_u32(matching_indices);
                    fill_indices_rvv(current_output, nsample, static_cast<int>(first_index));
                }

                // Store only as many matching indices as are still needed.
                int remaining_slots = nsample - cnt;
                int indices_to_store = std::min(remaining_slots, static_cast<int>(match_count));

                __riscv_vse32_v_u32m1(
                    reinterpret_cast<uint32_t *>(current_output + cnt),
                    matching_indices,
                    static_cast<size_t>(indices_to_store)
                );

                cnt += indices_to_store;
            }

            pts_cnt[i * m + j] = cnt;
        }
    }
}

// 4. Vectorized Feature Grouping
void group_point_rvv(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < nsample; ++k) {
                int ii = idx[(i * m + j) * nsample + k];

                const float *src_feature = points + (i * n + ii) * c;
                float *dst_feature = out + ((i * m + j) * nsample + k) * c;

                int rem_c = c;
                int c_offset = 0;

                // Strip-mine over feature channels
                for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                    vl = __riscv_vsetvl_e32m1(rem_c);
                    vfloat32m1_t v_feat = __riscv_vle32_v_f32m1(src_feature + c_offset, vl);
                    __riscv_vse32_v_f32m1(dst_feature + c_offset, v_feat, vl);
                }
            }
        }
    }
}

int main() {
    int b = 1;
    int n = 1024;
    int m = 128;
    int nsample = 8;
    int c       = 64;
    float radius = 0.2f;

    // Buffer allocations
    float *dataset     = new float[b * n * 3];
    float *points      = new float[b * n * c];
    float *temp        = new float[b * n];
    int   *fps_idx     = new int[b * m];
    float *new_xyz     = new float[b * m * 3];
    int   *ball_idx    = new int[b * m * nsample];
    int   *pts_cnt     = new int[b * m];
    float *grouped_out = new float[b * m * nsample * c];

    memset(fps_idx, 0, sizeof(int) * b * m);
    memset(ball_idx, 0, sizeof(int) * b * m * nsample);
    memset(pts_cnt, 0, sizeof(int) * b * m);
    memset(grouped_out, 0, sizeof(float) * b * m * nsample * c);

    // Fast deterministic setup
    for (int i = 0; i < b * n * 3; i++) dataset[i] = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * n * c; i++) points[i]  = (float)(i % 200) * 0.005f;

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    // Complete Downsampling & Grouping Pipeline
    farthestPointSampling_rvv(b, n, m, dataset, temp, fps_idx);
    gatherPoint_rvv(b, n, m, dataset, fps_idx, new_xyz);
    query_ball_point_rvv(b, n, m, radius, nsample, dataset, new_xyz, ball_idx, pts_cnt);
    group_point_rvv(b, n, c, m, nsample, points, ball_idx, grouped_out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Prevent dead code elimination
    printf("Pipeline check grouped_out[0]: %f\n", grouped_out[0]);

    delete[] dataset;
    delete[] points;
    delete[] temp;
    delete[] fps_idx;
    delete[] new_xyz;
    delete[] ball_idx;
    delete[] pts_cnt;
    delete[] grouped_out;

    return 0;
}