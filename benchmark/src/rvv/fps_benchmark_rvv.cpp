#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Vectorized Farthest Point Sampling (FPS) using RVV
void farthestPointSampling_rvv(int b, int n, int m, const float * __restrict__ dataset, float * __restrict__ temp, int * __restrict__ idxs) {
    if (m <= 0) return;
    const ptrdiff_t stride = 3 * sizeof(float); // 12-byte stride between (x, y, z) points

    for (int i = 0; i < b; ++i) {
        int old = 0;      // Start by picking point 0
        idxs[0] = old;

        // Initialize distance scratchpad to a very large number (1e38)
        int rem_init = n;
        int init_offset = 0;
        for (size_t vl; rem_init > 0; rem_init -= vl, init_offset += vl) {
            vl = __riscv_vsetvl_e32m1(rem_init);
            vfloat32m1_t v_inf = __riscv_vfmv_v_f_f32m1(1e38f, vl);
            __riscv_vse32_v_f32m1(temp + init_offset, v_inf, vl);
        }

        const float *cur_dataset = dataset + i * n * 3;

        // Find remaining m - 1 farthest points
        for (int j = 1; j < m; ++j) {
            int besti = 0;
            float best = -1.0f;

            // Coordinates of the newly selected sample point (x1, y1, z1)
            float x1 = cur_dataset[old * 3 + 0];
            float y1 = cur_dataset[old * 3 + 1];
            float z1 = cur_dataset[old * 3 + 2];

            int rem_n = n;
            int k_base = 0;

            // Process candidate points in vector chunks
            for (size_t vl; rem_n > 0; rem_n -= vl, k_base += vl) {
                vl = __riscv_vsetvl_e32m1(rem_n);

                // Load (x2, y2, z2) for 'vl' candidate points
                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 2, stride, vl);

                // Compute distance components: dx = x2 - x1, dy = y2 - y1, dz = z2 - z1
                vfloat32m1_t vdx = __riscv_vfsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfsub_vf_f32m1(vz2, z1, vl);

                // Compute squared Euclidean distance: d = dx*dx + dy*dy + dz*dz
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Load existing minimum distances from temp
                vfloat32m1_t v_temp = __riscv_vle32_v_f32m1(temp + k_base, vl);

                // Update min distance: temp[k] = min(temp[k], d)
                v_temp = __riscv_vfmin_vv_f32m1(v_temp, vd, vl);
                __riscv_vse32_v_f32m1(temp + k_base, v_temp, vl);

                // Check this chunk to see if any point is the new farthest
                float chunk_temp[vl];
                __riscv_vse32_v_f32m1(chunk_temp, v_temp, vl);
                for (size_t elem = 0; elem < vl; ++elem) {
                    if (chunk_temp[elem] > best) {
                        best = chunk_temp[elem];
                        besti = k_base + elem;
                    }
                }
            }

            // The farthest point becomes the next sample
            old = besti;
            idxs[j] = old;
        }

        temp += n;
        idxs += m;
    }
}

// Vectorized Gather Point using RVV
void gatherPoint_rvv(int b, int n, int m, const float * __restrict__ inp, const int * __restrict__ idx, float * __restrict__ out) {
    for (int i = 0; i < b; ++i) {
        const float *cur_inp = inp + i * n * 3;
        const int   *cur_idx = idx + i * m;
        float       *cur_out = out + i * m * 3;

        // Gather 3D coordinates for all m sampled points
        for (int j = 0; j < m; ++j) {
            int a = cur_idx[j]; // Sampled point index

            // Load 3 floats (x, y, z) using unit vector length = 3
            size_t vl = __riscv_vsetvl_e32m1(3);
            vfloat32m1_t v_coords = __riscv_vle32_v_f32m1(cur_inp + a * 3, vl);

            // Store (x, y, z) directly into the downsampled output array
            __riscv_vse32_v_f32m1(cur_out + j * 3, v_coords, vl);
        }
    }
}

int main() {
    int b = 1, n = 32, m = 8;

    // Allocate memory
    float *dataset = new float[b * n * 3]; // Input 3D points
    float *temp    = new float[b * n];     // Distance scratch buffer for FPS
    int   *idxs    = new int[b * m];       // Output indices from FPS
    float *out     = new float[b * m * 3]; // Final gathered 3D coordinates

    // Fast deterministic setup
    for (int i = 0; i < b * n * 3; ++i) {
        dataset[i] = (float)(i % 100) * 0.01f;
    }

    memset(temp, 0, sizeof(float) * b * n);
    memset(idxs, 0, sizeof(int) * b * m);
    memset(out, 0, sizeof(float) * b * m * 3);

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    // FPS + Gather Execution
    farthestPointSampling_rvv(b, n, m, dataset, temp, idxs);
    gatherPoint_rvv(b, n, m, dataset, idxs, out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Demo calculation check
    printf("Sample check gathered point 0: %f\n", out[0]);

    // Memory cleanup
    delete[] dataset;
    delete[] temp;
    delete[] idxs;
    delete[] out;

    return 0;
}