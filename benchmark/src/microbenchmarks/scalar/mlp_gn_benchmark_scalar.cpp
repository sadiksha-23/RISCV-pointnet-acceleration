#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

extern "C" {
    #include <gem5/m5ops.h>
}

// Full tf_util.conv2d pipeline: MLP (GEMM) + Batch Normalization + ReLU
void conv2d_mlp_bn_relu_cpu(int b, int n, int k, int c_in, int c_out, 
                            const float *X, const float *W, const float *bias,
                            const float *mean, const float *var, 
                            const float *gamma, const float *beta, 
                            float *Y) {
    
    int total_points = b * n * k; // Total spatial instances (B * N * K)
    float epsilon = 1e-5f;

    for (int p = 0; p < total_points; ++p) {
        for (int co = 0; co < c_out; ++co) {
            
            // --- Step 1: MLP / GEMM (X * W + Bias) ---
            float sum = bias[co];
            for (int ci = 0; ci < c_in; ++ci) {
                sum += X[p * c_in + ci] * W[ci * c_out + co];
            }
            
            // --- Step 2: Batch Normalization ---
            float norm = (sum - mean[co]) / std::sqrt(var[co] + epsilon);
            float bn_out = gamma[co] * norm + beta[co];

            // --- Step 3: ReLU Activation ---
            Y[p * c_out + co] = (bn_out > 0.0f) ? bn_out : 0.0f;
        }
    }
}

int main() {
    // 5, 1024, 32, 64, 128
    int b = 1;        // Batch size
    int n = 1024;     // Target points
    int k     = 8;       // Neighbors per point (nsample)
    int c_in  = 64;       // Input feature channels
    int c_out = 128;      // Output feature channels

    int total_points = b * n * k;

    // Allocating memory for tensors
    float *X    = new float[total_points * c_in];
    float *W    = new float[c_in * c_out];
    float *bias = new float[c_out];
    
    // Batch Norm parameters (per output channel)
    float *mean  = new float[c_out];
    float *var   = new float[c_out];
    float *gamma = new float[c_out];
    float *beta  = new float[c_out];

    float *Y = new float[total_points * c_out];

    // Fast deterministic setup (Zero rand() overhead)
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

    // Core Pipeline Execution
    conv2d_mlp_bn_relu_cpu(b, n, k, c_in, c_out, X, W, bias, mean, var, gamma, beta, Y);

    // --- DUMP STATS AFTER KERNEL ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away execution)
    printf("Point 0, Channel 0 final output: %f\n", Y[0]);

    // Memory cleanup
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