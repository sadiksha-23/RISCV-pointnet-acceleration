#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Vectorized MLP (GEMM) + Fused Batch Normalization + ReLU
void conv2d_mlp_bn_relu_rvv(int b, int n, int k, int c_in, int c_out, 
                            const float *X, const float *W, const float *bias,
                            const float *mean, const float *var, 
                            const float *gamma, const float *beta, 
                            float *Y) {
    
    int total_points = b * n * k;
    float epsilon = 1e-5f;

    // Precompute combined BN scale and shift parameters for inference
    std::vector<float> scale(c_out);
    std::vector<float> shift(c_out);
    for (int co = 0; co < c_out; ++co) {
        float inv_std = 1.0f / std::sqrt(var[co] + epsilon);
        scale[co] = gamma[co] * inv_std;
        shift[co] = beta[co] - (mean[co] * scale[co]);
    }

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
            vfloat32m1_t v_scale = __riscv_vle32_v_f32m1(scale.data() + co_offset, vl);
            vfloat32m1_t v_shift = __riscv_vle32_v_f32m1(shift.data() + co_offset, vl);
            
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

int main() {
    int b     = 1;
    int n     = 32;
    int k     = 8;
    int c_in  = 64;
    int c_out = 128;

    int total_points = b * n * k;

    // Buffer allocations
    float *X     = new float[total_points * c_in];
    float *W     = new float[c_in * c_out];
    float *bias  = new float[c_out];
    
    float *mean  = new float[c_out];
    float *var   = new float[c_out];
    float *gamma = new float[c_out];
    float *beta  = new float[c_out];

    float *Y = new float[total_points * c_out];

    // Fast deterministic setup
    for (int i = 0; i < total_points * c_in; i++) X[i] = (float)(i % 100) * 0.01f;
    for (int i = 0; i < c_in * c_out; i++)         W[i] = (float)(i % 50) * 0.02f;
    for (int i = 0; i < c_out; i++) {
        bias[i]  = 0.01f;
        mean[i]  = 0.05f;
        var[i]   = 1.00f;
        gamma[i] = 1.00f;
        beta[i]  = 0.00f;
    }

    // --- RESET STATS BEFORE KERNEL ---
    m5_reset_stats(0, 0);

    // Run vectorized kernel
    conv2d_mlp_bn_relu_rvv(b, n, k, c_in, c_out, X, W, bias, mean, var, gamma, beta, Y);

    // --- DUMP STATS AFTER KERNEL ---
    m5_dump_stats(0, 0);

    // Verify output to prevent dead-code elimination
    printf("Point 0, Channel 0 final output: %f\n", Y[0]);

    delete[] X;
    delete[] W;
    delete[] bias;
    delete[] mean;
    delete[] var;
    delete[] gamma;
    delete[] beta;
    delete[] Y;

    return 0;
}