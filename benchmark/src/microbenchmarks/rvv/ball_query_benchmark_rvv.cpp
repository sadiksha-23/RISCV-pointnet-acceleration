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

// Fill the complete neighbor list with the first matching index.
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

// Vectorized Ball Query using RVV.
void query_ball_point(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
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

// Vectorized Group Point across feature channels.
void group_point(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < nsample; ++k) {
                int ii = idx[(i * m + j) * nsample + k];

                const float *src = points + (i * n + ii) * c;
                float *dst = out + ((i * m + j) * nsample + k) * c;

                int rem_c = c;
                int c_offset = 0;

                for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                    vl = __riscv_vsetvl_e32m1(rem_c);
                    vfloat32m1_t values = __riscv_vle32_v_f32m1(src + c_offset, vl);
                    __riscv_vse32_v_f32m1(dst + c_offset, values, vl);
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
    int c = 64;
    float radius = 0.2f;

    float *xyz1 = new float[b * n * 3];
    float *xyz2 = new float[b * m * 3];
    float *points = new float[b * n * c];
    int *ball_idx = new int[b * m * nsample];
    int *pts_cnt = new int[b * m];
    float *grouped_out = new float[b * m * nsample * c];

    memset(ball_idx, 0, sizeof(int) * b * m * nsample);
    memset(pts_cnt, 0, sizeof(int) * b * m);
    memset(grouped_out, 0, sizeof(float) * b * m * nsample * c);

    // Keep the existing deterministic input.
    for (int i = 0; i < b * n * 3; ++i) xyz1[i] = static_cast<float>(i % 100) * 0.01f;
    for (int i = 0; i < b * m * 3; ++i) xyz2[i] = static_cast<float>(i % 50) * 0.02f;
    for (int i = 0; i < b * n * c; ++i) points[i] = static_cast<float>(i % 200) * 0.005f;

    // This measurement currently includes Ball Query and Group Point.
    m5_reset_stats(0, 0);

    query_ball_point(b, n, m, radius, nsample, xyz1, xyz2, ball_idx, pts_cnt);
    group_point(b, n, c, m, nsample, points, ball_idx, grouped_out);

    m5_dump_stats(0, 0);

    // Output checks.
    printf("First centroid neighbor count: %d\n", pts_cnt[0]);
    printf("First neighbor index: %d\n", ball_idx[0]);
    printf("Last stored neighbor index: %d\n", ball_idx[nsample - 1]);
    printf("Sample grouped value: %f\n", grouped_out[0]);

    delete[] xyz1;
    delete[] xyz2;
    delete[] points;
    delete[] ball_idx;
    delete[] pts_cnt;
    delete[] grouped_out;

    return 0;
}