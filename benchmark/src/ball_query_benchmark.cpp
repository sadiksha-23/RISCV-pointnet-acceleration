#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

extern "C" {
    #include <gem5/m5ops.h>
}

// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
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

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
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

    float *xyz1        = new float[b * n * 3];
    float *xyz2        = new float[b * m * 3];
    float *points      = new float[b * n * c];
    int   *ball_idx    = new int[b * m * nsample];
    int   *pts_cnt     = new int[b * m];
    float *grouped_out = new float[b * m * nsample * c];

    memset(ball_idx, 0, sizeof(int) * b * m * nsample);
    memset(pts_cnt, 0, sizeof(int) * b * m);
    memset(grouped_out, 0, sizeof(float) * b * m * nsample * c);

    // Fast deterministic setup (Zero rand() overhead)
    for (int i = 0; i < b * n * 3; i++) xyz1[i]   = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * m * 3; i++) xyz2[i]   = (float)(i % 50) * 0.02f;
    for (int i = 0; i < b * n * c; i++) points[i] = (float)(i % 200) * 0.005f;

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    query_ball_point(b, n, m, radius, nsample, xyz1, xyz2, ball_idx, pts_cnt);
    group_point(b, n, c, m, nsample, points, ball_idx, grouped_out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away the loop)
    printf("Sample check grouped_out[0]: %f\n", grouped_out[0]);

    delete[] xyz1;
    delete[] xyz2;
    delete[] points;
    delete[] ball_idx;
    delete[] pts_cnt;
    delete[] grouped_out;

    return 0;
}