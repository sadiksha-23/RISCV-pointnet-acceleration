#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Find one vector minimum and replace only its selected lane with infinity.
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

// PointNet++ 3NN with RVV distance calculation and RVV minimum selection.
void threenn_cpu(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
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

int main() {
    int b = 1;
    int n = 1024;
    int m = 128;

    // Allocate memory.
    float *xyz1 = new float[b * n * 3];
    float *xyz2 = new float[b * m * 3];
    float *dist = new float[b * n * 3];
    int *idx = new int[b * n * 3];

    // Generate deterministic coordinates.
    for (int i = 0; i < b * n * 3; ++i) {
        xyz1[i] = static_cast<float>(i % 100) * 0.01f;
    }

    for (int i = 0; i < b * m * 3; ++i) {
        xyz2[i] = static_cast<float>(i % 50) * 0.02f;
    }

    // Reset gem5 statistics before the kernel.
    m5_reset_stats(0, 0);

    threenn_cpu(b, n, m, xyz1, xyz2, dist, idx);

    // Dump gem5 statistics after the kernel.
    m5_dump_stats(0, 0);

    // Prevent the compiler from removing the result.
    printf("Point 0 -> 1st NN idx: %d, dist: %f\n", idx[0], dist[0]);
    printf("Point 0 -> 2nd NN idx: %d, dist: %f\n", idx[1], dist[1]);
    printf("Point 0 -> 3rd NN idx: %d, dist: %f\n", idx[2], dist[2]);

    // Release memory.
    delete[] xyz1;
    delete[] xyz2;
    delete[] dist;
    delete[] idx;

    return 0;
}