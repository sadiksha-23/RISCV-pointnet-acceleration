#include <cstdio>
#include <cstring>
#include <algorithm>

extern "C" {
    #include <gem5/m5ops.h>
}

// Scalar inverse distance weight calculation for 3NN interpolation
void get_weights_cpu(int b, int n, const float *dist, float *weight) {
    const float eps = 1e-10f; // Prevent division by zero

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            int base = (i * n + j) * 3;

            // Read neighbor distances
            float d0 = dist[base + 0];
            float d1 = dist[base + 1];
            float d2 = dist[base + 2];

            // Inverse distances (1.0 / dist)
            float w0 = 1.0f / std::max(d0, eps);
            float w1 = 1.0f / std::max(d1, eps);
            float w2 = 1.0f / std::max(d2, eps);

            // Sum of weights
            float sum = w0 + w1 + w2;

            // Normalize weights
            weight[base + 0] = w0 / sum;
            weight[base + 1] = w1 / sum;
            weight[base + 2] = w2 / sum;
        }
    }
}

int main() {
    int b = 1;   // Batch size
    int n = 32;  // Target points per batch
    int total_elements = b * n * 3;

    float *dist   = new float[total_elements];
    float *weight = new float[total_elements];

    // Fast deterministic distance initialization
    for (int i = 0; i < total_elements; ++i) {
        dist[i] = (float)(i % 100) * 0.01f + 0.05f;
    }

    // --- RESET STATS BEFORE KERNEL ---
    m5_reset_stats(0, 0);

    get_weights_cpu(b, n, dist, weight);

    // --- DUMP STATS AFTER KERNEL ---
    m5_dump_stats(0, 0);

    // Prevent compiler dead-code elimination
    printf("Scalar Output Check -> Point 0: [w0=%.4f, w1=%.4f, w2=%.4f]\n", 
           weight[0], weight[1], weight[2]);

    delete[] dist;
    delete[] weight;

    return 0;
}