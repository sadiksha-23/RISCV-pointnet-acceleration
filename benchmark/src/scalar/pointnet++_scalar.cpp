#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

extern "C" {
    #include <gem5/m5ops.h>
}

// -------------------------------------------------------------
// 1. DOWNSAMPLING & GROUPING KERNELS
// -------------------------------------------------------------

void farthestPointSampling(int b, int n, int m, const float * __restrict__ dataset, float * __restrict__ temp, int * __restrict__ idxs) {
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

void gatherPoint(int b, int n, int m, const float * __restrict__ inp, const int * __restrict__ idx, float * __restrict__ out) {
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

// -------------------------------------------------------------
// 2. FEATURE EXTRACTION & REDUCTION KERNELS
// -------------------------------------------------------------

void conv2d_mlp_bn_relu_cpu(int b, int n, int k, int c_in, int c_out, 
                            const float *X, const float *W, const float *bias,
                            const float *mean, const float *var, 
                            const float *gamma, const float *beta, 
                            float *Y) {
    
    int total_points = b * n * k;
    float epsilon = 1e-5f;

    for (int p = 0; p < total_points; ++p) {
        for (int co = 0; co < c_out; ++co) {
            
            // Step 1: MLP / GEMM
            float sum = bias[co];
            for (int ci = 0; ci < c_in; ++ci) {
                sum += X[p * c_in + ci] * W[ci * c_out + co];
            }
            
            // Step 2: Batch Normalization
            float norm = (sum - mean[co]) / std::sqrt(var[co] + epsilon);
            float bn_out = gamma[co] * norm + beta[co];

            // Step 3: ReLU Activation
            Y[p * c_out + co] = (bn_out > 0.0f) ? bn_out : 0.0f;
        }
    }
}

void maxpool_cpu(int b, int n, int k, int c, const float *input_features, float *output_features) {
    for (int batch = 0; batch < b; ++batch) {
        for (int pt = 0; pt < n; ++pt) {
            for (int ch = 0; ch < c; ++ch) {
                
                float max_val = -1e10f; 
                
                for (int nb = 0; nb < k; ++nb) {
                    int idx = batch * (n * k * c) + pt * (k * c) + nb * c + ch;
                    if (input_features[idx] > max_val) {
                        max_val = input_features[idx];
                    }
                }
                
                int out_idx = batch * (n * c) + pt * c + ch;
                output_features[out_idx] = max_val;
            }
        }
    }
}

// -------------------------------------------------------------
// 3. FEATURE PROPAGATION (UPSAMPLING) KERNELS
// -------------------------------------------------------------

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

void get_weights_cpu(int b, int n, const float *dist, float *weight) {
    const float eps = 1e-10f;
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            float d0 = dist[j * 3 + 0];
            float d1 = dist[j * 3 + 1];
            float d2 = dist[j * 3 + 2];

            float w0 = 1.0f / std::max(d0, eps);
            float w1 = 1.0f / std::max(d1, eps);
            float w2 = 1.0f / std::max(d2, eps);

            float sum = w0 + w1 + w2;
            weight[j * 3 + 0] = w0 / sum;
            weight[j * 3 + 1] = w1 / sum;
            weight[j * 3 + 2] = w2 / sum;
        } 
        dist   += n * 3;
        weight += n * 3;
    }
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

// -------------------------------------------------------------
// MAIN DRIVER & PROFILING PIPELINE
// -------------------------------------------------------------

int main() {
    // Hyperparameters
    int b       = 1;
    int n       = 32;
    int m       = 8;
    int nsample = 8;
    int c_in    = 64;
    int c_out   = 128;
    int classes = 4;
    float radius = 0.2f;

    // Buffer Allocations
    float *dataset      = new float[b * n * 3];
    float *points       = new float[b * n * c_in];
    float *temp         = new float[b * n];
    int   *fps_idx      = new int[b * m];
    float *new_xyz      = new float[b * m * 3];
    int   *ball_idx     = new int[b * m * nsample];
    int   *pts_cnt      = new int[b * m];
    float *grouped_out  = new float[b * m * nsample * c_in];

    // Stage 1 MLP Weights & BatchNorm (64 -> 128)
    float *W1     = new float[c_in * c_out];
    float *bias1  = new float[c_out];
    float *mean1  = new float[c_out];
    float *var1   = new float[c_out];
    float *gamma1 = new float[c_out];
    float *beta1  = new float[c_out];
    float *mlp_out = new float[b * m * nsample * c_out];

    // Max Pooling Output (M summary points with c_out channels)
    float *pooled_summary = new float[b * m * c_out];

    // Feature Propagation Buffers
    float *nn_dist   = new float[b * n * 3];
    int   *nn_idx    = new int[b * n * 3];
    float *nn_weight = new float[b * n * 3];
    float *interpolated_out = new float[b * n * c_out];

    // Stage 2 MLP Weights & BatchNorm (128 -> 4 classes)
    float *W2     = new float[c_out * classes];
    float *bias2  = new float[classes];
    float *mean2  = new float[classes];
    float *var2   = new float[classes];
    float *gamma2 = new float[classes];
    float *beta2  = new float[classes];
    float *final_logits = new float[b * n * classes];

    // Fast deterministic setup
    for (int i = 0; i < b * n * 3; ++i) dataset[i] = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * n * c_in; ++i) points[i] = (float)(i % 200) * 0.005f;

    for (int i = 0; i < c_in * c_out; ++i) W1[i] = (float)(i % 50) * 0.02f;
    for (int i = 0; i < c_out; ++i) {
        bias1[i]  = 0.01f;
        mean1[i]  = 0.05f;
        var1[i]   = 1.00f;
        gamma1[i] = 1.00f;
        beta1[i]  = 0.00f;
    }

    for (int i = 0; i < c_out * classes; ++i) W2[i] = (float)(i % 30) * 0.03f;
    for (int i = 0; i < classes; ++i) {
        bias2[i]  = 0.01f;
        mean2[i]  = 0.05f;
        var2[i]   = 1.00f;
        gamma2[i] = 1.00f;
        beta2[i]  = 0.00f;
    }

    memset(fps_idx, 0, sizeof(int) * b * m);
    memset(ball_idx, 0, sizeof(int) * b * m * nsample);
    memset(pts_cnt, 0, sizeof(int) * b * m);
    memset(nn_idx, 0, sizeof(int) * b * n * 3);

    // =========================================================
    // --- gem5 STATS RESET: START END-TO-END PIPELINE ---
    // =========================================================
    m5_reset_stats(0, 0);

    // 1. Spatial Downsampling & Grouping
    farthestPointSampling(b, n, m, dataset, temp, fps_idx);
    gatherPoint(b, n, m, dataset, fps_idx, new_xyz);
    query_ball_point(b, n, m, radius, nsample, dataset, new_xyz, ball_idx, pts_cnt);
    group_point(b, n, c_in, m, nsample, points, ball_idx, grouped_out);

    // 2. Feature Extraction & Reduction (Set Abstraction MLP)
    conv2d_mlp_bn_relu_cpu(b, m, nsample, c_in, c_out, grouped_out, W1, bias1, mean1, var1, gamma1, beta1, mlp_out);
    maxpool_cpu(b, m, nsample, c_out, mlp_out, pooled_summary);

    // 3. Spatial Upsampling & Feature Propagation
    threenn_cpu(b, n, m, dataset, new_xyz, nn_dist, nn_idx);
    get_weights_cpu(b, n, nn_dist, nn_weight);
    interpolate_cpu(b, m, c_out, n, pooled_summary, nn_idx, nn_weight, interpolated_out);

    // 4. Final Classification Head (Feature Propagation MLP)
    conv2d_mlp_bn_relu_cpu(b, n, 1, c_out, classes, interpolated_out, W2, bias2, mean2, var2, gamma2, beta2, final_logits);

    // =========================================================
    // --- gem5 STATS DUMP: END END-TO-END PIPELINE ---
    // =========================================================
    m5_dump_stats(0, 0);

    // Prevent Dead Code Elimination
    printf("PointNet++ End-to-End Pipeline Completed.\n");
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
    delete[] final_logits;

    return 0;
}