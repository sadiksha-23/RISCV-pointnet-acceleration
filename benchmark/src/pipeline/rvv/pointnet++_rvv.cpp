#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// -------------------------------------------------------------
// 1. DOWNSAMPLING & GROUPING KERNELS (RVV)
// -------------------------------------------------------------

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

// -------------------------------------------------------------
// 2. FEATURE EXTRACTION & REDUCTION KERNELS (RVV)
// -------------------------------------------------------------

void conv2d_mlp_bn_relu_rvv(int b, int n, int k, int c_in, int c_out, 
                            const float *X, const float *W, const float *bias,
                            const float *scale, const float *shift, 
                            float *Y) {
    
    int total_points = b * n * k;

    for (int p = 0; p < total_points; ++p) {
        const float *x_pt = X + p * c_in;
        float *y_pt = Y + p * c_out;

        int rem_co = c_out;
        int co_offset = 0;

        // Strip-mine across output feature channels
        for (size_t vl; rem_co > 0; rem_co -= vl, co_offset += vl) {
            vl = __riscv_vsetvl_e32m1(rem_co);

            // 1. Initialize vector accumulator with bias values
            vfloat32m1_t v_acc = __riscv_vle32_v_f32m1(bias + co_offset, vl);

            // 2. Perform GEMM: sum += X[p, ci] * W[ci, co]
            for (int ci = 0; ci < c_in; ++ci) {
                float x_val = x_pt[ci];
                const float *w_row = W + ci * c_out + co_offset;
                
                vfloat32m1_t v_w = __riscv_vle32_v_f32m1(w_row, vl);
                v_acc = __riscv_vfmacc_vf_f32m1(v_acc, x_val, v_w, vl);
            }

            // 3. Apply Batch Normalization: bn_out = sum * scale + shift
            vfloat32m1_t v_scale = __riscv_vle32_v_f32m1(scale + co_offset, vl);
            vfloat32m1_t v_shift = __riscv_vle32_v_f32m1(shift + co_offset, vl);
            
            vfloat32m1_t v_bn = __riscv_vfmul_vv_f32m1(v_acc, v_scale, vl);
            v_bn = __riscv_vfadd_vv_f32m1(v_bn, v_shift, vl);

            // 4. Apply ReLU Activation: max(0.0f, bn_out)
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_out = __riscv_vfmax_vv_f32m1(v_bn, v_zero, vl);

            // 5. Store final features to output array
            __riscv_vse32_v_f32m1(y_pt + co_offset, v_out, vl);
        }
    }
}

void maxpool_rvv(int b, int n, int k, int c, const float *input_features, float *output_features) {
    for (int batch = 0; batch < b; ++batch) {
        for (int pt = 0; pt < n; ++pt) {
            
            int rem_c = c;
            int ch_offset = 0;

            // Process feature channels in vector chunks
            for (size_t vl; rem_c > 0; rem_c -= vl, ch_offset += vl) {
                vl = __riscv_vsetvl_e32m1(rem_c);

                vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-1e10f, vl);

                for (int nb = 0; nb < k; ++nb) {
                    const float *src = input_features + batch * (n * k * c) + pt * (k * c) + nb * c + ch_offset;
                    vfloat32m1_t v_in = __riscv_vle32_v_f32m1(src, vl);
                    vmax = __riscv_vfmax_vv_f32m1(vmax, v_in, vl);
                }

                float *dst = output_features + batch * (n * c) + pt * c + ch_offset;
                __riscv_vse32_v_f32m1(dst, vmax, vl);
            }
        }
    }
}

// -------------------------------------------------------------
// 3. FEATURE PROPAGATION (UPSAMPLING) KERNELS (RVV)
// -------------------------------------------------------------

void threenn_rvv(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            float x1 = xyz1[j * 3 + 0]; 
            float y1 = xyz1[j * 3 + 1];
            float z1 = xyz1[j * 3 + 2];
                
            float best1 = 1e30f; float best2 = 1e30f; float best3 = 1e30f;
            int besti1 = 0; int besti2 = 0; int besti3 = 0;

            const float *curr_xyz2 = xyz2;
            int k_base = 0;
            int rem_m = m;

            for (size_t vl; rem_m > 0; rem_m -= vl, k_base += vl, curr_xyz2 += vl * 3) {
                vl = __riscv_vsetvl_e32m1(rem_m);

                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 2, stride, vl);

                vfloat32m1_t vdx = __riscv_vfrsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfrsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfrsub_vf_f32m1(vz2, z1, vl);

                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                float temp_d[64];
                __riscv_vse32_v_f32m1(temp_d, vd, vl);

                for (size_t elem = 0; elem < vl; ++elem) {
                    float d = temp_d[elem];
                    int curr_k = k_base + elem;

                    if (d < best1) {
                        best3 = best2; besti3 = besti2;
                        best2 = best1; besti2 = besti1;
                        best1 = d;     besti1 = curr_k;
                    } else if (d < best2) {
                        best3 = best2; besti3 = besti2;
                        best2 = d;     besti2 = curr_k;
                    } else if (d < best3) {
                        best3 = d;     besti3 = curr_k;
                    }
                }
            }

            dist[j * 3 + 0] = best1; idx[j * 3 + 0] = besti1;
            dist[j * 3 + 1] = best2; idx[j * 3 + 1] = besti2;
            dist[j * 3 + 2] = best3; idx[j * 3 + 2] = besti3;
        }

        xyz1 += n * 3;
        xyz2 += m * 3;
        dist += n * 3;
        idx  += n * 3;
    }
}

void get_weights_rvv(int b, int n, const float *dist, float *weight) {
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        const float *curr_dist = dist + (i * n * 3);
        float *curr_weight = weight + (i * n * 3);
        int rem_n = n;

        for (size_t vl; rem_n > 0; rem_n -= vl, curr_dist += vl * 3, curr_weight += vl * 3) {
            vl = __riscv_vsetvl_e32m1(rem_n);

            vfloat32m1_t vd0 = __riscv_vlse32_v_f32m1(curr_dist + 0, stride, vl);
            vfloat32m1_t vd1 = __riscv_vlse32_v_f32m1(curr_dist + 1, stride, vl);
            vfloat32m1_t vd2 = __riscv_vlse32_v_f32m1(curr_dist + 2, stride, vl);

            vfloat32m1_t vw0 = __riscv_vfrdiv_vf_f32m1(vd0, 1.0f, vl);
            vfloat32m1_t vw1 = __riscv_vfrdiv_vf_f32m1(vd1, 1.0f, vl);
            vfloat32m1_t vw2 = __riscv_vfrdiv_vf_f32m1(vd2, 1.0f, vl);

            vfloat32m1_t vsum = __riscv_vfadd_vv_f32m1(vw0, vw1, vl);
            vsum = __riscv_vfadd_vv_f32m1(vsum, vw2, vl);

            vw0 = __riscv_vfdiv_vv_f32m1(vw0, vsum, vl);
            vw1 = __riscv_vfdiv_vv_f32m1(vw1, vsum, vl);
            vw2 = __riscv_vfdiv_vv_f32m1(vw2, vsum, vl);

            __riscv_vsse32_v_f32m1(curr_weight + 0, stride, vw0, vl);
            __riscv_vsse32_v_f32m1(curr_weight + 1, stride, vw1, vl);
            __riscv_vsse32_v_f32m1(curr_weight + 2, stride, vw2, vl);
        }
    }
}

void interpolate_rvv(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            float w1 = weight[j * 3 + 0];
            float w2 = weight[j * 3 + 1];
            float w3 = weight[j * 3 + 2];

            int i1 = idx[j * 3 + 0];
            int i2 = idx[j * 3 + 1];
            int i3 = idx[j * 3 + 2];

            const float *p1 = points + i1 * c;
            const float *p2 = points + i2 * c;
            const float *p3 = points + i3 * c;
            float *out_pt   = out + j * c;

            int rem_c = c;
            int c_offset = 0;

            for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                vl = __riscv_vsetvl_e32m1(rem_c);

                vfloat32m1_t vf1 = __riscv_vle32_v_f32m1(p1 + c_offset, vl);
                vfloat32m1_t vf2 = __riscv_vle32_v_f32m1(p2 + c_offset, vl);
                vfloat32m1_t vf3 = __riscv_vle32_v_f32m1(p3 + c_offset, vl);

                vfloat32m1_t v_out = __riscv_vfmul_vf_f32m1(vf1, w1, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w2, vf2, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w3, vf3, vl);

                __riscv_vse32_v_f32m1(out_pt + c_offset, v_out, vl);
            }
        } 

        points += m * c;
        idx    += n * 3;
        weight += n * 3;
        out    += n * c;
    }
}

// -------------------------------------------------------------
// MAIN DRIVER & PROFILING PIPELINE
// -------------------------------------------------------------

int main() {
    int b = 5;
    int n = 1024;
    int m = 128;
    int nsample = 8;
    int c_in    = 64;
    int c_out   = 128;
    int classes = 4;
    float radius = 0.2f;
    float epsilon = 1e-5f;

    // Buffer Allocations
    float *dataset      = new float[b * n * 3];
    float *points       = new float[b * n * c_in];
    float *temp         = new float[b * n];
    int   *fps_idx      = new int[b * m];
    float *new_xyz      = new float[b * m * 3];
    int   *ball_idx     = new int[b * m * nsample];
    int   *pts_cnt      = new int[b * m];
    float *grouped_out  = new float[b * m * nsample * c_in];

    // Stage 1 MLP Weights & Precomputed BN Parameters (64 -> 128)
    float *W1     = new float[c_in * c_out];
    float *bias1  = new float[c_out];
    float *mean1  = new float[c_out];
    float *var1   = new float[c_out];
    float *gamma1 = new float[c_out];
    float *beta1  = new float[c_out];
    float *scale1 = new float[c_out];
    float *shift1 = new float[c_out];
    float *mlp_out = new float[b * m * nsample * c_out];

    // Max Pooling Output
    float *pooled_summary = new float[b * m * c_out];

    // Feature Propagation Buffers
    float *nn_dist   = new float[b * n * 3];
    int   *nn_idx    = new int[b * n * 3];
    float *nn_weight = new float[b * n * 3];
    float *interpolated_out = new float[b * n * c_out];

    // Stage 2 Classification MLP Weights & Precomputed BN Parameters (128 -> 4)
    float *W2     = new float[c_out * classes];
    float *bias2  = new float[classes];
    float *mean2  = new float[classes];
    float *var2   = new float[classes];
    float *gamma2 = new float[classes];
    float *beta2  = new float[classes];
    float *scale2 = new float[classes];
    float *shift2 = new float[classes];
    float *final_logits = new float[b * n * classes];

    // Fast deterministic test data setup
    for (int i = 0; i < b * n * 3; ++i) dataset[i] = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * n * c_in; ++i) points[i] = (float)(i % 200) * 0.005f;

    for (int i = 0; i < c_in * c_out; ++i) W1[i] = (float)(i % 50) * 0.02f;
    for (int i = 0; i < c_out; ++i) {
        bias1[i]  = 0.01f;
        mean1[i]  = 0.05f;
        var1[i]   = 1.00f;
        gamma1[i] = 1.00f;
        beta1[i]  = 0.00f;
        
        float inv_std = 1.0f / std::sqrt(var1[i] + epsilon);
        scale1[i] = gamma1[i] * inv_std;
        shift1[i] = beta1[i] - (mean1[i] * scale1[i]);
    }

    for (int i = 0; i < c_out * classes; ++i) W2[i] = (float)(i % 30) * 0.03f;
    for (int i = 0; i < classes; ++i) {
        bias2[i]  = 0.01f;
        mean2[i]  = 0.05f;
        var2[i]   = 1.00f;
        gamma2[i] = 1.00f;
        beta2[i]  = 0.00f;

        float inv_std = 1.0f / std::sqrt(var2[i] + epsilon);
        scale2[i] = gamma2[i] * inv_std;
        shift2[i] = beta2[i] - (mean2[i] * scale2[i]);
    }

    memset(fps_idx, 0, sizeof(int) * b * m);
    memset(ball_idx, 0, sizeof(int) * b * m * nsample);
    memset(pts_cnt, 0, sizeof(int) * b * m);
    memset(nn_idx, 0, sizeof(int) * b * n * 3);

    // =========================================================
    // --- gem5 STATS RESET: START END-TO-END RVV PIPELINE ---
    // =========================================================
    m5_reset_stats(0, 0);

    // 1. Spatial Downsampling & Grouping (RVV)
    farthestPointSampling_rvv(b, n, m, dataset, temp, fps_idx);
    gatherPoint_rvv(b, n, m, dataset, fps_idx, new_xyz);
    query_ball_point_rvv(b, n, m, radius, nsample, dataset, new_xyz, ball_idx, pts_cnt);
    group_point_rvv(b, n, c_in, m, nsample, points, ball_idx, grouped_out);

    // 2. Feature Extraction & Reduction (RVV Set Abstraction MLP)
    conv2d_mlp_bn_relu_rvv(b, m, nsample, c_in, c_out, grouped_out, W1, bias1, scale1, shift1, mlp_out);
    maxpool_rvv(b, m, nsample, c_out, mlp_out, pooled_summary);

    // 3. Spatial Upsampling & Feature Propagation (RVV)
    threenn_rvv(b, n, m, dataset, new_xyz, nn_dist, nn_idx);
    get_weights_rvv(b, n, nn_dist, nn_weight);
    interpolate_rvv(b, m, c_out, n, pooled_summary, nn_idx, nn_weight, interpolated_out);

    // 4. Final Classification Head (RVV Feature Propagation MLP)
    conv2d_mlp_bn_relu_rvv(b, n, 1, c_out, classes, interpolated_out, W2, bias2, scale2, shift2, final_logits);

    // =========================================================
    // --- gem5 STATS DUMP: END END-TO-END RVV PIPELINE ---
    // =========================================================
    m5_dump_stats(0, 0);

    // Prevent Dead Code Elimination
    printf("PointNet++ RVV End-to-End Pipeline Completed.\n");
    printf("Sample Check: Point 0 Class 0 Logit: %f\n", final_logits[0]);

    // Memory Cleanup
    delete[] dataset;
    delete[] points;
    delete[] temp;
    delete[] fps_idx;
    delete[] new_xyz;
    delete[] ball_idx;
    delete[] pts_cnt;
    delete[] grouped_out;
    delete[] W1;
    delete[] bias1;
    delete[] mean1;
    delete[] var1;
    delete[] gamma1;
    delete[] beta1;
    delete[] scale1;
    delete[] shift1;
    delete[] mlp_out;
    delete[] pooled_summary;
    delete[] nn_dist;
    delete[] nn_idx;
    delete[] nn_weight;
    delete[] interpolated_out;
    delete[] W2;
    delete[] bias2;
    delete[] mean2;
    delete[] var2;
    delete[] gamma2;
    delete[] beta2;
    delete[] scale2;
    delete[] shift2;
    delete[] final_logits;

    return 0;
}