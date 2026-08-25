#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

extern "C" {
    #include <gem5/m5ops.h>
}

void farthestPointSampling (int b, int n, int m, const float * __restrict__ dataset, float * __restrict__ temp, int * __restrict__ idxs) {
    if (m <= 0) return;

    for (int i = 0; i < b; ++i) {
        int old = 0;
        idxs[0] = old;

        for (int a = 0; a < n; ++a) {
            temp[a] = 1e38f;
        }

        for (int j = 1; j < m; ++j) {
            int besti = 0;
            float best = -1;

            float x1 = dataset[i * n * 3 + old * 3 + 0];
            float y1 = dataset[i * n * 3 + old * 3 + 1];
            float z1 = dataset[i * n * 3 + old * 3 + 2];

            for (int k = 0; k < n; ++k) {
                float x2 = dataset[i * n * 3 + k * 3 + 0];
                float y2 = dataset[i * n * 3 + k * 3 + 1];
                float z2 = dataset[i * n * 3 + k * 3 + 2];

                float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

                if (d < temp[k]) {
                    temp[k] = d;
                }
                if (temp[k] > best) {
                    best = temp[k];
                    besti = k;
                }
            }
            old = besti;
            idxs[j] = old;
        }

        temp += n;
        idxs += m;
    }
}

void gatherPoint (int b, int n, int m, const float * __restrict__ inp, const int * __restrict__ idx, float * __restrict__ out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            int a = idx[i * m + j]; 
            
            out[(i * m + j) * 3 + 0] = inp[(i * n + a) * 3 + 0];
            out[(i * m + j) * 3 + 1] = inp[(i * n + a) * 3 + 1];
            out[(i * m + j) * 3 + 2] = inp[(i * n + a) * 3 + 2];
        }
    }
}

int main() {
    int b = 1, n = 32, m = 8;

    // Memory allocation
    float *dataset = new float[b * n * 3]; // Input 3D points
    float *temp    = new float[b * n];     // Scratch buffer for FPS distance tracking
    int   *idxs    = new int[b * m];       // Output indices from FPS
    float *out     = new float[b * m * 3]; // Final gathered 3D coordinates

    // Fast deterministic setup (Zero rand() overhead)
    for (int i = 0; i < b * n * 3; ++i) {
        dataset[i] = (float)(i % 100) * 0.01f;
    }

    memset(temp, 0, sizeof(float) * b * n);
    memset(idxs, 0, sizeof(int) * b * m);
    memset(out, 0, sizeof(float) * b * m * 3);

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    // FPS + Gather Execution
    farthestPointSampling(b, n, m, dataset, temp, idxs);
    gatherPoint(b, n, m, dataset, idxs, out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away the loop)
    printf("Sample check gathered point 0: %f\n", out[0]);

    // Memory cleanup
    delete[] dataset;
    delete[] temp;
    delete[] idxs;
    delete[] out;

    return 0;
}