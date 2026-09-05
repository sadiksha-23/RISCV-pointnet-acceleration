#include <cstdio>
#include <cstring>
#include <algorithm>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Vectorized inverse distance weight calculation using RISC-V Vector intrinsics
void get_weights_rvv(int b, int n, const float *dist, float *weight) {
    const ptrdiff_t stride = 3 * sizeof(float); // 12-byte stride between coordinates

    for (int i = 0; i < b; ++i) {
        const float *curr_dist = dist + (i * n * 3);
        float *curr_weight = weight + (i * n * 3);
        int rem_n = n;

        // Process points in vector chunks (vl)
        for (size_t vl; rem_n > 0; rem_n -= vl, curr_dist += vl * 3, curr_weight += vl * 3) {
            vl = __riscv_vsetvl_e32m1(rem_n);

            // 1. Strided load: grab d0, d1, d2 across 'vl' points
            vfloat32m1_t vd0 = __riscv_vlse32_v_f32m1(curr_dist + 0, stride, vl);
            vfloat32m1_t vd1 = __riscv_vlse32_v_f32m1(curr_dist + 1, stride, vl);
            vfloat32m1_t vd2 = __riscv_vlse32_v_f32m1(curr_dist + 2, stride, vl);

            // 2. Vector reciprocal division: w = 1.0f / d
            vfloat32m1_t vw0 = __riscv_vfrdiv_vf_f32m1(vd0, 1.0f, vl);
            vfloat32m1_t vw1 = __riscv_vfrdiv_vf_f32m1(vd1, 1.0f, vl);
            vfloat32m1_t vw2 = __riscv_vfrdiv_vf_f32m1(vd2, 1.0f, vl);

            // 3. Vector sum: vsum = vw0 + vw1 + vw2
            vfloat32m1_t vsum = __riscv_vfadd_vv_f32m1(vw0, vw1, vl);
            vsum = __riscv_vfadd_vv_f32m1(vsum, vw2, vl);

            // 4. Vector normalize: w = w / vsum
            vw0 = __riscv_vfdiv_vv_f32m1(vw0, vsum, vl);
            vw1 = __riscv_vfdiv_vv_f32m1(vw1, vsum, vl);
            vw2 = __riscv_vfdiv_vv_f32m1(vw2, vsum, vl);

            // 5. Strided store: write back normalized weights
            __riscv_vsse32_v_f32m1(curr_weight + 0, stride, vw0, vl);
            __riscv_vsse32_v_f32m1(curr_weight + 1, stride, vw1, vl);
            __riscv_vsse32_v_f32m1(curr_weight + 2, stride, vw2, vl);
        }
    }
}

int main() {
    int b = 1;   // Batch size
    int n = 1024;  // Target points per batch
    int total_elements = b * n * 3;

    float *dist   = new float[total_elements];
    float *weight = new float[total_elements];

    // Fast deterministic distance initialization
    for (int i = 0; i < total_elements; ++i) {
        dist[i] = (float)(i % 100) * 0.01f + 0.05f;
    }

    // --- RESET STATS BEFORE KERNEL ---
    m5_reset_stats(0, 0);

    get_weights_rvv(b, n, dist, weight);

    // --- DUMP STATS AFTER KERNEL ---
    m5_dump_stats(0, 0);

    // Prevent compiler dead-code elimination
    printf("Vector Output Check -> Point 0: [w0=%.4f, w1=%.4f, w2=%.4f]\n", 
           weight[0], weight[1], weight[2]);

    delete[] dist;
    delete[] weight;

    return 0;
}