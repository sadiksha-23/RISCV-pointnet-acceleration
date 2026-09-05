#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Vectorized query_ball_point using RVV
// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
void query_ball_point(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    float radius2 = radius * radius;
    const ptrdiff_t stride = 3 * sizeof(float); // 12-byte jump between coordinates

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            int cnt = 0;

            float x2 = xyz2[(i * m + j) * 3 + 0];
            float y2 = xyz2[(i * m + j) * 3 + 1];
            float z2 = xyz2[(i * m + j) * 3 + 2];

            const float *curr_xyz1 = xyz1 + (i * n * 3); // starting pointer for candidates
            int k_base = 0; // base point index of this chunk
            int rem_n = n; // remaining candidate points left to check in this batch

            // Process candidate points in chunks of vector length (vl)
            for (size_t vl; rem_n > 0; rem_n -= vl, k_base += vl, curr_xyz1 += vl * 3) {
                if (cnt == nsample) break;

                // Set vector length for float32 lanes
                vl = __riscv_vsetvl_e32m1(rem_n);

                // Load x1, y1, z1 coordinates using strided loads
                vfloat32m1_t vx1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 0, stride, vl);
                vfloat32m1_t vy1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 1, stride, vl);
                vfloat32m1_t vz1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 2, stride, vl);

                // Calculate differences: (x2 - x1), (y2 - y1), (z2 - z1)
                vfloat32m1_t vdx = __riscv_vfrsub_vf_f32m1(vx1, x2, vl);
                vfloat32m1_t vdy = __riscv_vfrsub_vf_f32m1(vy1, y2, vl);
                vfloat32m1_t vdz = __riscv_vfrsub_vf_f32m1(vz1, z2, vl);

                // Compute squared distance: vd = (dx*dx) + (dy*dy) + (dz*dz)
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Save computed distances from vector register to stack buffer
                float temp_d[64];
                __riscv_vse32_v_f32m1(temp_d, vd, vl);

                // Check chunk elements to find points within ball query radius
                for (size_t elem = 0; elem < vl; ++elem) {
                    float d2 = temp_d[elem];
                    int curr_k = k_base + elem;

                    if (d2 < radius2) {
                        if (cnt == 0) {
                            for (int l = 0; l < nsample; ++l) {
                                idx[(i * m + j) * nsample + l] = curr_k;
                            }
                        }
                        idx[(i * m + j) * nsample + cnt] = curr_k;
                        cnt++;
                        if (cnt == nsample) break;
                    }
                }
            }

            pts_cnt[i * m + j] = cnt;
        }
    }
}

// Vectorized group_point across feature channels (c) using RVV
// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
void group_point(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < nsample; ++k) {
                int ii = idx[(i * m + j) * nsample + k];

                const float *src = points + (i * n + ii) * c;
                float *dst = out + ((i * m + j) * nsample + k) * c;

                int rem_c = c;
                int c_offset = 0;

                // Vectorized feature channel copy
                for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                    vl = __riscv_vsetvl_e32m1(rem_c);
                    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + c_offset, vl);
                    __riscv_vse32_v_f32m1(dst + c_offset, v, vl);
                }
            }
        }
    }
}

int main() {
    int b = 1;
    int n = 1024;
    int m = 128;
    int nsample = 8;
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

    // Fast deterministic setup
    for (int i = 0; i < b * n * 3; i++) xyz1[i]   = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * m * 3; i++) xyz2[i]   = (float)(i % 50) * 0.02f;
    for (int i = 0; i < b * n * c; i++) points[i] = (float)(i % 200) * 0.005f;

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    query_ball_point(b, n, m, radius, nsample, xyz1, xyz2, ball_idx, pts_cnt);
    group_point(b, n, c, m, nsample, points, ball_idx, grouped_out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Demo calculation check
    printf("Sample check grouped_out[0]: %f\n", grouped_out[0]);

    delete[] xyz1;
    delete[] xyz2;
    delete[] points;
    delete[] ball_idx;
    delete[] pts_cnt;
    delete[] grouped_out;

    return 0;
}