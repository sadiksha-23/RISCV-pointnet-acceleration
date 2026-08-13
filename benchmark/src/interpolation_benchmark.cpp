#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
    #include <gem5/m5ops.h>
}

void interpolate_cpu(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out) {
     float w1, w2, w3;
     int i1, i2, i3;

     for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            w1 = weight[j * 3 + 0];
            w2 = weight[j * 3 + 1];
            w3 = weight[j * 3 + 2]; 
            i1 = idx[j * 3 + 0];
            i2 = idx[j * 3 + 1];
            i3 = idx[j * 3 + 2];

            for (int l = 0; l < c; ++l) {
                out[j * c + l] = points[i1 * c + l] * w1 + points[i2 * c + l] * w2 + points[i3 * c + l] * w3;
            }
        } 

        points += m * c;
        idx    += n * 3;
        weight += n * 3;
        out    += n * c;
    }
}

int main() {
    int b = 32;   // batch size
    int n = 1024; // target points per batch
    int m = 128;  // candidate points per batch
    int c = 64;   // feature channels per point

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
        idx[i]    = i % m;        // Deterministic candidate index within [0, m-1]
        weight[i] = 1.0f / 3.0f;  // Uniform weight = 0.3333f
    }

    // --- RESET STATS BEFORE KERNEL ---
    m5_reset_stats(0, 0);

    // Core Kernel Execution
    interpolate_cpu(b, m, c, n, points, idx, weight, out);

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