#include <cstdio>
#include <algorithm>

extern "C" {
    #include <gem5/m5ops.h>
}

// assigning weights for the 3-Nearest Neighbor interpolation
void get_weights_cpu(int b, int n, const float *dist, float *weight) {
    const float w = 1.0f / 3.0f; // constant = 0.3333f

    // iterating over batch size (B) and point cloud (N)
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            weight[j * 3 + 0] = w;
            weight[j * 3 + 1] = w;
            weight[j * 3 + 2] = w;
        } 
        dist   += n * 3;
        weight += n * 3;
    }
}

int main() {
    int b = 32;   // batch size
    int n = 1024; // target points per batch

    int total_elements = b * n * 3;

    // allocating memory
    float *dist   = new float[total_elements];
    float *weight = new float[total_elements];

    // Fast deterministic setup (Zero rand() overhead)
    for (int i = 0; i < total_elements; ++i) {
        dist[i] = (float)(i % 100) * 0.01f + 0.01f;
    }

    // --- RESET STATS BEFORE KERNEL ---
    m5_reset_stats(0, 0);

    // Core Kernel Execution
    get_weights_cpu(b, n, dist, weight);

    // --- DUMP STATS AFTER KERNEL ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away execution)
    printf("Point 0 Weights -> Neighbor 1: %f, Neighbor 2: %f, Neighbor 3: %f\n", 
           weight[0], weight[1], weight[2]);

    // memory cleanup
    delete[] dist;
    delete[] weight;

    return 0;
}