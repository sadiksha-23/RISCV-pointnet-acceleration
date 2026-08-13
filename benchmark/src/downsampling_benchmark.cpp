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

void query_ball_point(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    float radius2 = radius * radius;

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            int cnt = 0;

            float x2 = xyz2[(i * m + j) * 3 + 0];
            float y2 = xyz2[(i * m + j) * 3 + 1];
            float z2 = xyz2[(i * m + j) * 3 + 2];

            for (int k = 0; k < n; ++k) {
                if (cnt == nsample) break;

                float x1 = xyz1[(i * n + k) * 3 + 0];
                float y1 = xyz1[(i * n + k) * 3 + 1];
                float z1 = xyz1[(i * n + k) * 3 + 2];

                float d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

                if (d2 < radius2) {
                    if (cnt == 0) {
                        for (int l = 0; l < nsample; ++l) {
                            idx[(i * m + j) * nsample + l] = k;
                        }
                    }
                    idx[(i * m + j) * nsample + cnt] = k;
                    cnt++;
                }
            }

            pts_cnt[i * m + j] = cnt;
        }
    }
}

void group_point(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < nsample; ++k) {
                int ii = idx[(i * m + j) * nsample + k];

                for (int l = 0; l < c; ++l) {
                    out[((i * m + j) * nsample + k) * c + l] = points[(i * n + ii) * c + l];
                }
            }
        }
    }
}

int main() {
    int b       = 32;
    int n       = 1024;
    int m       = 128;
    int nsample = 32;
    int c       = 64;
    float radius = 0.2f;

    float *dataset     = new float[b * n * 3];
    float *points      = new float[b * n * c];
    float *temp        = new float[b * n];
    int   *fps_idx     = new int[b * m];
    float *new_xyz     = new float[b * m * 3];
    int   *ball_idx    = new int[b * m * nsample];
    int   *pts_cnt     = new int[b * m];
    float *grouped_out = new float[b * m * nsample * c];

    memset(fps_idx, 0, sizeof(int) * b * m);
    memset(ball_idx, 0, sizeof(int) * b * m * nsample);
    memset(pts_cnt, 0, sizeof(int) * b * m);
    memset(grouped_out, 0, sizeof(float) * b * m * nsample * c);

    // Fast deterministic setup (Zero rand() overhead)
    for (int i = 0; i < b * n * 3; i++) dataset[i] = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * n * c; i++) points[i]  = (float)(i % 200) * 0.005f;

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    // Complete Downsampling & Grouping Pipeline
    farthestPointSampling(b, n, m, dataset, temp, fps_idx);
    gatherPoint(b, n, m, dataset, fps_idx, new_xyz);
    query_ball_point(b, n, m, radius, nsample, dataset, new_xyz, ball_idx, pts_cnt);
    group_point(b, n, c, m, nsample, points, ball_idx, grouped_out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away execution)
    printf("Pipeline check grouped_out[0]: %f\n", grouped_out[0]);

    delete[] dataset;
    delete[] points;
    delete[] temp;
    delete[] fps_idx;
    delete[] new_xyz;
    delete[] ball_idx;
    delete[] pts_cnt;
    delete[] grouped_out;

    return 0;
}