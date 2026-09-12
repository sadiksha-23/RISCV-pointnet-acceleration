#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Find the largest value in one vector chunk and its first lane.
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

// Vectorized Farthest Point Sampling using RVV.
void farthestPointSampling_rvv(int b, int n, int m, const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
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

// Vectorized Gather Point using RVV.
void gatherPoint_rvv(int b, int n, int m, const float *__restrict__ inp, const int *__restrict__ idx, float *__restrict__ out) {
    for (int i = 0; i < b; ++i) {
        const float *cur_inp = inp + i * n * 3;
        const int *cur_idx = idx + i * m;
        float *cur_out = out + i * m * 3;

        // Copy the x, y and z coordinates of every selected point.
        for (int j = 0; j < m; ++j) {
            int selected_index = cur_idx[j];

            // Each point contains three consecutive float values.
            size_t vl = __riscv_vsetvl_e32m1(3);
            vfloat32m1_t coordinates = __riscv_vle32_v_f32m1(cur_inp + selected_index * 3, vl);
            __riscv_vse32_v_f32m1(cur_out + j * 3, coordinates, vl);
        }
    }
}

int main() {
    int b = 1;
    int n = 1024;
    int m = 128;

    // Allocate memory.
    float *dataset = new float[b * n * 3];
    float *temp = new float[b * n];
    int *idxs = new int[b * m];
    float *out = new float[b * m * 3];

    // Keep the existing deterministic input.
    for (int i = 0; i < b * n * 3; ++i) {
        dataset[i] = static_cast<float>(i % 100) * 0.01f;
    }

    memset(temp, 0, sizeof(float) * b * n);
    memset(idxs, 0, sizeof(int) * b * m);
    memset(out, 0, sizeof(float) * b * m * 3);

    // This region currently measures FPS and Gather together.
    m5_reset_stats(0, 0);

    farthestPointSampling_rvv(b, n, m, dataset, temp, idxs);
    gatherPoint_rvv(b, n, m, dataset, idxs, out);

    m5_dump_stats(0, 0);

    // Display selected indices and gathered coordinates.
    printf("FPS centroid 0: %d\n", idxs[0]);
    printf("FPS centroid 1: %d\n", idxs[1]);
    printf("FPS centroid 2: %d\n", idxs[2]);
    printf("FPS final centroid: %d\n", idxs[m - 1]);

    printf("Gathered point 0: (%f, %f, %f)\n", out[0], out[1], out[2]);

    // Release memory.
    delete[] dataset;
    delete[] temp;
    delete[] idxs;
    delete[] out;

    return 0;
}