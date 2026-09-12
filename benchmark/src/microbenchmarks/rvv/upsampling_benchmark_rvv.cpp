#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// 1. Vectorized 3-Nearest Neighbor Search
static inline vfloat32m1_t take_vector_minimum(vfloat32m1_t values, size_t vl, float &minimum_value, int &minimum_lane) {
    const float infinity = 1e30f;

    // Find the smallest value across the active vector lanes.
    vfloat32m1_t initial = __riscv_vfmv_v_f_f32m1(infinity, 1);
    vfloat32m1_t reduced = __riscv_vfredmin_vs_f32m1_f32m1(values, initial, vl);
    minimum_value = __riscv_vfmv_f_s_f32m1_f32(reduced);

    // Find the first lane containing the minimum value.
    vbool32_t minimum_mask = __riscv_vmfeq_vf_f32m1_b32(values, minimum_value, vl);
    minimum_lane = static_cast<int>(__riscv_vfirst_m_b32(minimum_mask, vl));

    // Select only that lane.
    vuint32m1_t lane_numbers = __riscv_vid_v_u32m1(vl);
    vbool32_t selected_lane = __riscv_vmseq_vx_u32m1_b32(lane_numbers, static_cast<uint32_t>(minimum_lane), vl);

    // Remove the selected lane so the next call finds the next minimum.
    return __riscv_vfmerge_vfm_f32m1(values, infinity, selected_lane, vl);
}

void threenn_rvv(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            // Load the current target point.
            float x1 = xyz1[j * 3 + 0];
            float y1 = xyz1[j * 3 + 1];
            float z1 = xyz1[j * 3 + 2];

            // Store the three smallest distances and their indices.
            float best1 = 1e30f;
            float best2 = 1e30f;
            float best3 = 1e30f;

            int besti1 = 0;
            int besti2 = 0;
            int besti3 = 0;

            const float *curr_xyz2 = xyz2;
            int k_base = 0;
            int rem_m = m;

            // Process candidate points in RVV chunks.
            for (size_t vl; rem_m > 0; rem_m -= vl, k_base += vl, curr_xyz2 += vl * 3) {
                vl = __riscv_vsetvl_e32m1(rem_m);

                // Load candidate x, y and z coordinates.
                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 2, stride, vl);

                // Calculate coordinate differences.
                vfloat32m1_t vdx = __riscv_vfrsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfrsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfrsub_vf_f32m1(vz2, z1, vl);

                // Calculate squared distances: dx² + dy² + dz².
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Keep the distances in the vector and extract up to three minima.
                vfloat32m1_t remaining_distances = vd;
                int candidates_to_check = (vl < 3) ? static_cast<int>(vl) : 3;

                for (int candidate = 0; candidate < candidates_to_check; ++candidate) {
                    float d;
                    int minimum_lane;

                    remaining_distances = take_vector_minimum(remaining_distances, vl, d, minimum_lane);
                    int curr_k = k_base + minimum_lane;

                    // Original top-three update logic.
                    if (d < best1) {
                        best3 = best2;
                        besti3 = besti2;
                        best2 = best1;
                        besti2 = besti1;
                        best1 = d;
                        besti1 = curr_k;
                    } else if (d < best2) {
                        best3 = best2;
                        besti3 = besti2;
                        best2 = d;
                        besti2 = curr_k;
                    } else if (d < best3) {
                        best3 = d;
                        besti3 = curr_k;
                    }
                }
            }

            // Store the final three nearest neighbors.
            dist[j * 3 + 0] = best1;
            idx[j * 3 + 0] = besti1;

            dist[j * 3 + 1] = best2;
            idx[j * 3 + 1] = besti2;

            dist[j * 3 + 2] = best3;
            idx[j * 3 + 2] = besti3;
        }

        // Move pointers to the next batch.
        xyz1 += n * 3;
        xyz2 += m * 3;
        dist += n * 3;
        idx += n * 3;
    }
}

// 2. Vectorized Inverse Distance Weighting: w_k = (1/d_k) / sum(1/d_j)
void get_weights_rvv(int b, int n, const float *dist, float *weight) {
    const ptrdiff_t stride = 3 * sizeof(float);
    const float eps = 1e-10f;

    for (int i = 0; i < b; ++i) {
        const float *curr_dist = dist + (i * n * 3);
        float *curr_weight = weight + (i * n * 3);
        int rem_n = n;

        for (size_t vl; rem_n > 0; rem_n -= vl, curr_dist += vl * 3, curr_weight += vl * 3) {
            vl = __riscv_vsetvl_e32m1(rem_n);

            // Strided load for 3 neighbor distances across points
            vfloat32m1_t vd0 = __riscv_vlse32_v_f32m1(curr_dist + 0, stride, vl);
            vfloat32m1_t vd1 = __riscv_vlse32_v_f32m1(curr_dist + 1, stride, vl);
            vfloat32m1_t vd2 = __riscv_vlse32_v_f32m1(curr_dist + 2, stride, vl);

            // Guard against division by zero: max(d, eps)
            vfloat32m1_t veps = __riscv_vfmv_v_f_f32m1(eps, vl);
            vd0 = __riscv_vfmax_vv_f32m1(vd0, veps, vl);
            vd1 = __riscv_vfmax_vv_f32m1(vd1, veps, vl);
            vd2 = __riscv_vfmax_vv_f32m1(vd2, veps, vl);

            // Inverse distance: w = 1.0f / d
            vfloat32m1_t vw0 = __riscv_vfrdiv_vf_f32m1(vd0, 1.0f, vl);
            vfloat32m1_t vw1 = __riscv_vfrdiv_vf_f32m1(vd1, 1.0f, vl);
            vfloat32m1_t vw2 = __riscv_vfrdiv_vf_f32m1(vd2, 1.0f, vl);

            // Sum of inverse distances: sum = w0 + w1 + w2
            vfloat32m1_t vsum = __riscv_vfadd_vv_f32m1(vw0, vw1, vl);
            vsum = __riscv_vfadd_vv_f32m1(vsum, vw2, vl);

            // Normalize weights
            vw0 = __riscv_vfdiv_vv_f32m1(vw0, vsum, vl);
            vw1 = __riscv_vfdiv_vv_f32m1(vw1, vsum, vl);
            vw2 = __riscv_vfdiv_vv_f32m1(vw2, vsum, vl);

            // Strided store back to weight buffer
            __riscv_vsse32_v_f32m1(curr_weight + 0, stride, vw0, vl);
            __riscv_vsse32_v_f32m1(curr_weight + 1, stride, vw1, vl);
            __riscv_vsse32_v_f32m1(curr_weight + 2, stride, vw2, vl);
        }
    }
}

// 3. Vectorized Feature Interpolation across Channels
void interpolate_rvv(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            float w1 = weight[j * 3 + 0];
            float w2 = weight[j * 3 + 1];
            float w3 = weight[j * 3 + 2];

            int i1 = idx[j * 3 + 0];
            int i2 = idx[j * 3 + 1];
            int i3 = idx[j * 3 + 2];

            const float *p1 = points + i1 * c;
            const float *p2 = points + i2 * c;
            const float *p3 = points + i3 * c;
            float *out_pt   = out + j * c;

            int rem_c = c;
            int c_offset = 0;

            // Strip-mine over feature channels
            for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                vl = __riscv_vsetvl_e32m1(rem_c);

                vfloat32m1_t vf1 = __riscv_vle32_v_f32m1(p1 + c_offset, vl);
                vfloat32m1_t vf2 = __riscv_vle32_v_f32m1(p2 + c_offset, vl);
                vfloat32m1_t vf3 = __riscv_vle32_v_f32m1(p3 + c_offset, vl);

                // Weighted sum: out = (p1 * w1) + (p2 * w2) + (p3 * w3)
                vfloat32m1_t v_out = __riscv_vfmul_vf_f32m1(vf1, w1, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w2, vf2, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w3, vf3, vl);

                __riscv_vse32_v_f32m1(out_pt + c_offset, v_out, vl);
            }
        }

        points += m * c;
        idx    += n * 3;
        weight += n * 3;
        out    += n * c;
    }
}

int main() {
    int b = 1, n = 1024, m = 128, c = 64;

    float *xyz1   = new float[b * n * 3];
    float *xyz2   = new float[b * m * 3];
    float *dist   = new float[b * n * 3];
    int   *idx    = new int[b * n * 3];
    float *weight = new float[b * n * 3];
    float *points = new float[b * m * c];
    float *out    = new float[b * n * c];

    memset(idx, 0, sizeof(int) * b * n * 3);

    // Fast deterministic setup
    for (int i = 0; i < b * n * 3; i++) xyz1[i]   = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * m * 3; i++) xyz2[i]   = (float)(i % 50) * 0.02f;
    for (int i = 0; i < b * m * c; i++) points[i] = (float)(i % 200) * 0.005f;

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    // Complete Feature Propagation Pipeline Execution
    threenn_rvv(b, n, m, xyz1, xyz2, dist, idx);
    get_weights_rvv(b, n, dist, weight);
    interpolate_rvv(b, m, c, n, points, idx, weight, out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Verify calculation output
    printf("FP check output point 0, channel 0: %f\n", out[0]);

    // Cleanup memory
    delete[] xyz1;
    delete[] xyz2;
    delete[] dist;
    delete[] idx;
    delete[] weight;
    delete[] points;
    delete[] out;

    return 0;
}