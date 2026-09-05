#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Neighborhood Max Pooling using RISC-V Vector instructions
void maxpool_rvv(int b, int n, int k, int c, const float *input_features, float *output_features) {
    for (int batch = 0; batch < b; ++batch) {
        for (int pt = 0; pt < n; ++pt) {
            
            int rem_c = c;      // channels left to process
            int ch_offset = 0;  // current channel position

            // Process feature channels in vector chunks
            for (size_t vl; rem_c > 0; rem_c -= vl, ch_offset += vl) {
                vl = __riscv_vsetvl_e32m1(rem_c);

                // start with -infinity for the max search
                vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-1e10f, vl);

                // look through all K neighbors for this point
                for (int nb = 0; nb < k; ++nb) {
                    // Find memory location of this neighbor's channels
                    const float *src = input_features + batch * (n * k * c) + pt * (k * c) + nb * c + ch_offset;

                    // Load a chunk of channels from this neighbor
                    vfloat32m1_t v_in = __riscv_vle32_v_f32m1(src, vl);

                    // Keep the higher value between current max and neighbor
                    vmax = __riscv_vfmax_vv_f32m1(vmax, v_in, vl);
                }

                // Find memory location where the output channels go
                float *dst = output_features + batch * (n * c) + pt * c + ch_offset;

                // Save the maximum values into the output array
                __riscv_vse32_v_f32m1(dst, vmax, vl);
            }
        }
    }
}

int main() {
    int b = 1;   // Batch size
    int n = 1024;  // Number of points
    int k = 8;   // Number of neighbors per point
    int c = 64;  // Number of feature channels per point

    int total_input_elements = b * n * k * c;
    int total_output_elements = b * n * c;

    // Allocate memory for inputs and outputs
    float *input_features = new float[total_input_elements];
    float *output_features = new float[total_output_elements];

    // Fill input with deterministic test data
    for (int i = 0; i < total_input_elements; i++) {
        input_features[i] = (float)(i % 100) * 0.01f;
    }

    // --- RESET STATS BEFORE RUNNING KERNEL ---
    m5_reset_stats(0, 0);

    // Run the vectorized maxpool function
    maxpool_rvv(b, n, k, c, input_features, output_features);

    // --- DUMP STATS AFTER RUNNING KERNEL ---
    m5_dump_stats(0, 0);

    // Print one value to verify calculation worked
    printf("Point 0, Channel 0 max value: %f\n", output_features[0]);

    // Free allocated memory
    delete[] input_features;
    delete[] output_features;

    return 0;
}