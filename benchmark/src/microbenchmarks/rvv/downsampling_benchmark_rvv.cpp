#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// 1. Vectorized Farthest Point Sampling (FPS)
void farthestPointSampling_rvv(int b, int n, int m, const float * __restrict__ dataset, float * __restrict__ temp, int * __restrict__ idxs) {
    if (m <= 0) return;
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        int old = 0;
        idxs[0] = old;

        // Initialize temp distance buffer with infinity (1e38)
        int rem_init = n;
        int init_offset = 0;
        for (size_t vl; rem_init > 0; rem_init -= vl, init_offset += vl) {
            vl = __riscv_vsetvl_e32m1(rem_init);
            vfloat32m1_t v_inf = __riscv_vfmv_v_f_f32m1(1e38f, vl);
            __riscv_vse32_v_f32m1(temp + init_offset, v_inf, vl);
        }

        const float *cur_dataset = dataset + i * n * 3;

        for (int j = 1; j < m; ++j) {
            int besti = 0;
            float best = -1.0f;

            float x1 = cur_dataset[old * 3 + 0];
            float y1 = cur_dataset[old * 3 + 1];
            float z1 = cur_dataset[old * 3 + 2];

            int rem_n = n;
            int k_base = 0;

            for (size_t vl; rem_n > 0; rem_n -= vl, k_base += vl) {
                vl = __riscv_vsetvl_e32m1(rem_n);

                // Load candidate points (x2, y2, z2)
                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 2, stride, vl);

                // Distance calculation
                vfloat32m1_t vdx = __riscv_vfsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfsub_vf_f32m1(vz2, z1, vl);

                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Update shortest distance across samples
                vfloat32m1_t v_temp = __riscv_vle32_v_f32m1(temp + k_base, vl);
                v_temp = __riscv_vfmin_vv_f32m1(v_temp, vd, vl);
                __riscv_vse32_v_f32m1(temp + k_base, v_temp, vl);

                // Find local maximum distance
                float chunk_temp[vl];
                __riscv_vse32_v_f32m1(chunk_temp, v_temp, vl);
                for (size_t elem = 0; elem < vl; ++elem) {
                    if (chunk_temp[elem] > best) {
                        best = chunk_temp[elem];
                        besti = k_base + elem;
                    }
                }
            }

            old = besti;
            idxs[j] = old;
        }

        temp += n;
        idxs += m;
    }
}

// 2. Vectorized Point Gathering
void gatherPoint_rvv(int b, int n, int m, const float * __restrict__ inp, const int * __restrict__ idx, float * __restrict__ out) {
    for (int i = 0; i < b; ++i) {
        const float *cur_inp = inp + i * n * 3;
        const int   *cur_idx = idx + i * m;
        float       *cur_out = out + i * m * 3;

        for (int j = 0; j < m; ++j) {
            int a = cur_idx[j];
            size_t vl = __riscv_vsetvl_e32m1(3);
            vfloat32m1_t v_coords = __riscv_vle32_v_f32m1(cur_inp + a * 3, vl);
            __riscv_vse32_v_f32m1(cur_out + j * 3, v_coords, vl);
        }
    }
}

// 3. Vectorized Ball Query
void query_ball_point_rvv(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    float radius2 = radius * radius;
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            int cnt = 0;
            int *cur_idx = idx + (i * m + j) * nsample;

            float x2 = xyz2[(i * m + j) * 3 + 0];
            float y2 = xyz2[(i * m + j) * 3 + 1];
            float z2 = xyz2[(i * m + j) * 3 + 2];

            int rem_n = n;
            int k_base = 0;

            for (size_t vl; rem_n > 0; rem_n -= vl, k_base += vl) {
                if (cnt == nsample) break;
                vl = __riscv_vsetvl_e32m1(rem_n);

                // Load candidate points from dataset
                vfloat32m1_t vx1 = __riscv_vlse32_v_f32m1(xyz1 + (i * n + k_base) * 3 + 0, stride, vl);
                vfloat32m1_t vy1 = __riscv_vlse32_v_f32m1(xyz1 + (i * n + k_base) * 3 + 1, stride, vl);
                vfloat32m1_t vz1 = __riscv_vlse32_v_f32m1(xyz1 + (i * n + k_base) * 3 + 2, stride, vl);

                // Compute squared Euclidean distance
                vfloat32m1_t vdx = __riscv_vfsub_vf_f32m1(vx1, x2, vl);
                vfloat32m1_t vdy = __riscv_vfsub_vf_f32m1(vy1, y2, vl);
                vfloat32m1_t vdz = __riscv_vfsub_vf_f32m1(vz1, z2, vl);

                vfloat32m1_t vd2 = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd2 = __riscv_vfmacc_vv_f32m1(vd2, vdy, vdy, vl);
                vd2 = __riscv_vfmacc_vv_f32m1(vd2, vdz, vdz, vl);

                float temp_d[vl];
                __riscv_vse32_v_f32m1(temp_d, vd2, vl);

                // Check distance against radius threshold
                for (size_t elem = 0; elem < vl; ++elem) {
                    if (temp_d[elem] < radius2) {
                        if (cnt == 0) {
                            for (int l = 0; l < nsample; ++l) {
                                cur_idx[l] = k_base + elem;
                            }
                        }
                        cur_idx[cnt] = k_base + elem;
                        cnt++;
                        if (cnt == nsample) break;
                    }
                }
            }
            pts_cnt[i * m + j] = cnt;
        }
    }
}

// 4. Vectorized Feature Grouping
void group_point_rvv(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < nsample; ++k) {
                int ii = idx[(i * m + j) * nsample + k];

                const float *src_feature = points + (i * n + ii) * c;
                float *dst_feature = out + ((i * m + j) * nsample + k) * c;

                int rem_c = c;
                int c_offset = 0;

                // Strip-mine over feature channels
                for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                    vl = __riscv_vsetvl_e32m1(rem_c);
                    vfloat32m1_t v_feat = __riscv_vle32_v_f32m1(src_feature + c_offset, vl);
                    __riscv_vse32_v_f32m1(dst_feature + c_offset, v_feat, vl);
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

    // Buffer allocations
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

    // Fast deterministic setup
    for (int i = 0; i < b * n * 3; i++) dataset[i] = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * n * c; i++) points[i]  = (float)(i % 200) * 0.005f;

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    // Complete Downsampling & Grouping Pipeline
    farthestPointSampling_rvv(b, n, m, dataset, temp, fps_idx);
    gatherPoint_rvv(b, n, m, dataset, fps_idx, new_xyz);
    query_ball_point_rvv(b, n, m, radius, nsample, dataset, new_xyz, ball_idx, pts_cnt);
    group_point_rvv(b, n, c, m, nsample, points, ball_idx, grouped_out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Prevent dead code elimination
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