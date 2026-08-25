#include <cstdio>
#include <cstring>
#include <cmath>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// 1. Vectorized 3-Nearest Neighbor Search
void threenn_rvv(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    const ptrdiff_t stride = 3 * sizeof(float); // 12-byte stride between (x, y, z)

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            // Target point coordinates (x1, y1, z1)
            float x1 = xyz1[j * 3 + 0];
            float y1 = xyz1[j * 3 + 1];
            float z1 = xyz1[j * 3 + 2];

            // Initialize top-3 best distances and their indices
            float best1 = 1e38f, best2 = 1e38f, best3 = 1e38f;
            int besti1 = 0, besti2 = 0, besti3 = 0;

            int rem_m = m;
            int k_base = 0;

            // Process source candidate points in vector chunks
            for (size_t vl; rem_m > 0; rem_m -= vl, k_base += vl) {
                vl = __riscv_vsetvl_e32m1(rem_m);

                // Load candidate points (x2, y2, z2)
                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(xyz2 + k_base * 3 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(xyz2 + k_base * 3 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(xyz2 + k_base * 3 + 2, stride, vl);

                // Compute squared Euclidean distance: d = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
                vfloat32m1_t vdx = __riscv_vfsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfsub_vf_f32m1(vz2, z1, vl);

                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Store chunk distances to inspect top 3 nearest
                float d_buf[vl];
                __riscv_vse32_v_f32m1(d_buf, vd, vl);

                for (size_t elem = 0; elem < vl; ++elem) {
                    float d = d_buf[elem];
                    int k = k_base + elem;

                    if (d < best1) {
                        best3 = best2;   besti3 = besti2;
                        best2 = best1;   besti2 = besti1;
                        best1 = d;       besti1 = k;
                    } else if (d < best2) {
                        best3 = best2;   besti3 = besti2;
                        best2 = d;       besti2 = k;
                    } else if (d < best3) {
                        best3 = d;       besti3 = k;
                    }
                }
            }

            // Save top 3 distances and indices for point j
            dist[j * 3 + 0] = best1;  idx[j * 3 + 0] = besti1;
            dist[j * 3 + 1] = best2;  idx[j * 3 + 1] = besti2;
            dist[j * 3 + 2] = best3;  idx[j * 3 + 2] = besti3;
        }

        xyz1 += n * 3;
        xyz2 += m * 3;
        dist += n * 3;
        idx  += n * 3;
    }
}

// 2. Vectorized Uniform Weights Setup
void get_weights_rvv(int b, int n, const float *dist, float *weight) {
    const float w = 1.0f / 3.0f;
    int total_weights = b * n * 3;
    int offset = 0;

    // Fill the weight array with 1/3 in vector chunks
    for (size_t vl; total_weights > 0; total_weights -= vl, offset += vl) {
        vl = __riscv_vsetvl_e32m1(total_weights);
        vfloat32m1_t vw = __riscv_vfmv_v_f_f32m1(w, vl);
        __riscv_vse32_v_f32m1(weight + offset, vw, vl);
    }
}

// 3. Vectorized Feature Interpolation across Channels
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

            // Strip-mine over feature channels
            for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                vl = __riscv_vsetvl_e32m1(rem_c);

                // Load feature slices for the 3 nearest neighbors
                vfloat32m1_t vf1 = __riscv_vle32_v_f32m1(p1 + c_offset, vl);
                vfloat32m1_t vf2 = __riscv_vle32_v_f32m1(p2 + c_offset, vl);
                vfloat32m1_t vf3 = __riscv_vle32_v_f32m1(p3 + c_offset, vl);

                // Weighted sum: out = (p1 * w1) + (p2 * w2) + (p3 * w3)
                vfloat32m1_t v_out = __riscv_vfmul_vf_f32m1(vf1, w1, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w2, vf2, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w3, vf3, vl);

                // Write interpolated channel vector to output
                __riscv_vse32_v_f32m1(out_pt + c_offset, v_out, vl);
            }
        }

        points += m * c;
        idx    += n * 3;
        weight += n * 3;
        out    += n * c;
    }
}

int main() {
    int b = 1, n = 32, m = 8, c = 64;

    float *xyz1   = new float[b * n * 3];
    float *xyz2   = new float[b * m * 3];
    float *dist   = new float[b * n * 3];
    int   *idx    = new int[b * n * 3];
    float *weight = new float[b * n * 3];
    float *points = new float[b * m * c];
    float *out    = new float[b * n * c];

    memset(idx, 0, sizeof(int) * b * n * 3);

    // Fast deterministic setup
    for (int i = 0; i < b * n * 3; i++) xyz1[i]   = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * m * 3; i++) xyz2[i]   = (float)(i % 50) * 0.02f;
    for (int i = 0; i < b * m * c; i++) points[i] = (float)(i % 200) * 0.005f;

    // --- RESET STATS BEFORE PIPELINE ---
    m5_reset_stats(0, 0);

    // Complete Feature Propagation Pipeline Execution
    threenn_rvv(b, n, m, xyz1, xyz2, dist, idx);
    get_weights_rvv(b, n, dist, weight);
    interpolate_rvv(b, m, c, n, points, idx, weight, out);

    // --- DUMP STATS AFTER PIPELINE ---
    m5_dump_stats(0, 0);

    // Verify calculation output
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