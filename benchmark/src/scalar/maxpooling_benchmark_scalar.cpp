#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

extern "C" {
    #include <gem5/m5ops.h>
}

// PointNet++ Neighborhood Max Pooling
void maxpool_cpu(int b, int n, int k, int c, const float *input_features, float *output_features) {
    for (int batch = 0; batch < b; ++batch) {
        for (int pt = 0; pt < n; ++pt) {
            for (int ch = 0; ch < c; ++ch) {
                
                float max_val = -1e10f; 
                
                for (int nb = 0; nb < k; ++nb) {
                    // Indexing into [batch][pt][nb][ch]
                    int idx = batch * (n * k * c) + pt * (k * c) + nb * c + ch;
                    if (input_features[idx] > max_val) {
                        max_val = input_features[idx];
                    }
                }
                
                // Storing into [batch][pt][ch]
                int out_idx = batch * (n * c) + pt * c + ch;
                output_features[out_idx] = max_val;
            }
        }
    }
}

int main() {
    int b = 1;   // batch size
    int n = 32; // number of points
    int k = 8;   // neighbors per point (nsample)
    int c = 64;   // feature channels

    int total_input_elements = b * n * k * c;
    int total_output_elements = b * n * c;

    // Allocating memory
    float *input_features = new float[total_input_elements];
    float *output_features = new float[total_output_elements];

    // Fast deterministic setup (Zero rand() overhead)
    for (int i = 0; i < total_input_elements; i++) {
        input_features[i] = (float)(i % 100) * 0.01f;
    }

    // --- RESET STATS BEFORE KERNEL ---
    m5_reset_stats(0, 0);

    // Core Kernel Execution
    maxpool_cpu(b, n, k, c, input_features, output_features);

    // --- DUMP STATS AFTER KERNEL ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away execution)
    printf("Point 0, Channel 0 max value: %f\n", output_features[0]);

    // Memory cleanup
    delete[] input_features;
    delete[] output_features;

    return 0;
}