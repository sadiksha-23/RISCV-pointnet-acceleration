#include <cstdio>
#include <cstring>
#include <cmath>

extern "C" {
    #include <gem5/m5ops.h>
}

// Find three nearest neighbors with square distance
// input: xyz1 (b,n,3), xyz2(b,m,3)
// output: dist (b,n,3), idx (b,n,3)
void threenn_cpu(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            float x1 = xyz1[j * 3 + 0];
            float y1 = xyz1[j * 3 + 1];
            float z1 = xyz1[j * 3 + 2];

            double best1 = 1e40; double best2 = 1e40; double best3 = 1e40;
            int besti1 = 0; int besti2 = 0; int besti3 = 0;

            for (int k = 0; k < m; ++k) {
                float x2 = xyz2[k * 3 + 0];
                float y2 = xyz2[k * 3 + 1];
                float z2 = xyz2[k * 3 + 2];

                float dx = x1 - x2;
                float dy = y1 - y2;
                float dz = z1 - z2;
                double d = dx * dx + dy * dy + dz * dz;

                if (d < best1) {
                    best3  = best2;
                    besti3 = besti2;
                    best2  = best1;
                    besti2 = besti1;
                    best1  = d;
                    besti1 = k;
                } else if (d < best2) {
                    best3  = best2;
                    besti3 = besti2;
                    best2  = d;
                    besti2 = k;
                } else if (d < best3) {
                    best3  = d;
                    besti3 = k;
                }
            } 
            dist[j * 3 + 0] = best1;
            idx[j * 3 + 0]  = besti1;
            dist[j * 3 + 1] = best2;
            idx[j * 3 + 1]  = besti2;
            dist[j * 3 + 2] = best3;
            idx[j * 3 + 2]  = besti3;
        } 
        xyz1 += n * 3;
        xyz2 += m * 3;
        dist += n * 3;
        idx  += n * 3;
    }
} 

// CONSTANT WEIGHT
// input: dist (b,n,3)
// output: weight (b,n,3)
void get_weights_cpu(int b, int n, const float *dist, float *weight) {
    const float w = 1.0f / 3.0f;
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

// input: points (b,m,c), idx (b,n,3), weight (b,n,3)
// output: out (b,n,c)
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
    int b = 32, n = 1024, m = 128, c = 64;

    float *xyz1   = new float[b * n * 3];
    float *xyz2   = new float[b * m * 3];
    float *dist   = new float[b * n * 3];
    int   *idx    = new int[b * n * 3];
    float *weight = new float[b * n * 3];
    float *points = new float[b * m * c];
    float *out    = new float[b * n * c];

    memset(idx, 0, sizeof(int) * b * n * 3);

    // Fast deterministic setup (Zero rand() overhead)
    for (int i = 0; i < b * n * 3; i++) xyz1[i]   = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * m * 3; i++) xyz2[i]   = (float)(i % 50) * 0.02f;
    for (int i = 0; i < b * m * c; i++) points[i] = (float)(i % 200) * 0.005f;

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    // Complete Feature Propagation Pipeline Execution
    threenn_cpu(b, n, m, xyz1, xyz2, dist, idx);
    get_weights_cpu(b, n, dist, weight);
    interpolate_cpu(b, m, c, n, points, idx, weight, out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away execution)
    printf("FP check output point 0, channel 0: %f\n", out[0]);

    // Cleanup memory
    delete[] xyz1;
    delete[] xyz2;
    delete[] dist;
    delete[] idx;
    delete[] weight;
    delete[] points;
    delete[] out;

    return 0;
}