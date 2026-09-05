#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Vectorized Feature Interpolation using RVV
void interpolate_rvv(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            // Load scalar weights for the 3 nearest neighbors
            float w1 = weight[j * 3 + 0];
            float w2 = weight[j * 3 + 1];
            float w3 = weight[j * 3 + 2];

            // Load neighbor indices
            int i1 = idx[j * 3 + 0];
            int i2 = idx[j * 3 + 1];
            int i3 = idx[j * 3 + 2];

            // Base memory pointers to the 3 neighbors' feature channels
            const float *p1 = points + i1 * c;
            const float *p2 = points + i2 * c;
            const float *p3 = points + i3 * c;
            float *out_pt   = out + j * c;

            int rem_c = c;
            int c_offset = 0;

            // Strip-mine across feature channels
            for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                vl = __riscv_vsetvl_e32m1(rem_c);

                // Load feature chunks for the 3 neighbors
                vfloat32m1_t vf1 = __riscv_vle32_v_f32m1(p1 + c_offset, vl);
                vfloat32m1_t vf2 = __riscv_vle32_v_f32m1(p2 + c_offset, vl);
                vfloat32m1_t vf3 = __riscv_vle32_v_f32m1(p3 + c_offset, vl);

                // Weighted sum: out = (p1 * w1) + (p2 * w2) + (p3 * w3)
                vfloat32m1_t v_out = __riscv_vfmul_vf_f32m1(vf1, w1, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w2, vf2, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w3, vf3, vl);

                // Store interpolated vector directly into memory
                __riscv_vse32_v_f32m1(out_pt + c_offset, v_out, vl);
            }
        } 

        // Advance pointers for next batch
        points += m * c;
        idx    += n * 3;
        weight += n * 3;
        out    += n * c;
    }
}

int main() {
    int b = 1;   // batch size
    int n = 1024;  // target points per batch
    int m = 128;   // candidate points per batch
    int c = 64;  // feature channels per point

    // Allocating memory
    float *points = new float[b * m * c];
    int   *idx    = new int[b * n * 3];
    float *weight = new float[b * n * 3];
    float *out    = new float[b * n * c];

    // Fast deterministic setup (Zero rand() overhead)
    for (int i = 0; i < b * m * c; i++) {
        points[i] = (float)(i % 100) * 0.01f;
    }
    for (int i = 0; i < b * n * 3; i++) {
        idx[i]    = i % m;         // Deterministic candidate index within [0, m-1]
        weight[i] = 1.0f / 3.0f;   // Uniform weight = 0.3333f
    }

    // --- RESET STATS BEFORE KERNEL ---
    m5_reset_stats(0, 0);

    // Core Kernel Execution
    interpolate_rvv(b, m, c, n, points, idx, weight, out);

    // --- DUMP STATS AFTER KERNEL ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away execution)
    printf("Point 0 -> Feature Channel 0 out: %f\n", out[0]);
    printf("Point 0 -> Feature Channel 1 out: %f\n", out[1]);

    // Memory cleanup
    delete[] points;
    delete[] idx;
    delete[] weight;
    delete[] out;

    return 0;
}