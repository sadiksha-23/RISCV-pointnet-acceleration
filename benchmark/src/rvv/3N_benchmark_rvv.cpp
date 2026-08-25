#include <iostream>
#include <vector>
#include <cmath>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// original PointNet++ 3NN (3-Nearest Neighbors) Code
void threenn_cpu(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    const ptrdiff_t stride = 3 * sizeof(float); // offset for finding same x, y, or z

     for (int i=0;i<b;++i) { // going through each batch
        for (int j=0;j<n;++j) { // going through every target point
            // loading x, y, z values from xyz1
            float x1=xyz1[j*3+0]; 
            float y1=xyz1[j*3+1];
            float z1=xyz1[j*3+2];
                
            float best1 = 1e30f; float best2 = 1e30f; float best3 = 1e30f; // placeholder for 3 smallest distances
            int besti1 = 0; int besti2 = 0; int besti3 = 0;                 // storing the index of closest points

            const float *curr_xyz2 = xyz2; // moving pointer for the current chunk
            int k_base = 0; // stores the starting index of current chunk 
            int rem_m = m; // remaining points left to process

            // Loop through candidate points in chunks of vector length (vl)
            for (size_t vl; rem_m > 0; rem_m -= vl, k_base += vl, curr_xyz2 += vl * 3) {
                // Configure vector length for active float32 lanes (LMUL=1)
                vl = __riscv_vsetvl_e32m1(rem_m);

                // Load x2, y2, z2 coordinates into separate vector registers
                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 2, stride, vl);

                // Calculate differences: (x1 - x2), (y1 - y2), (z1 - z2)
                vfloat32m1_t vdx = __riscv_vfrsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfrsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfrsub_vf_f32m1(vz2, z1, vl);

                // Compute distance: vd = (dx*dx) + (dy*dy) + (dz*dz)
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Save computed distances from vector register to an array
                float temp_d[64];
                __riscv_vse32_v_f32m1(temp_d, vd, vl);

                // Check this chunk to find and update the 3 closest points
                for (size_t elem = 0; elem < vl; ++elem) {
                    float d = temp_d[elem];
                    int curr_k = k_base + elem;

                    if (d < best1) {
                        best3 = best2;
                        besti3 = besti2;
                        best2 = best1;
                        besti2 = besti1;
                        best1 = d;
                        besti1 = curr_k;
                    } else if (d < best2) {
                        best3 = best2;
                        besti3 = besti2;
                        best2 = d;
                        besti2 = curr_k;
                    } else if (d < best3) {
                        best3 = d;
                        besti3 = curr_k;
                    }
                }
            }

            // Storing top 3 smallest distances and candidate indices
            dist[j * 3 + 0] = best1;
            idx[j * 3 + 0]  = besti1;
            dist[j * 3 + 1] = best2;
            idx[j * 3 + 1]  = besti2;
            dist[j * 3 + 2] = best3;
            idx[j * 3 + 2]  = besti3;
        }

        // Moving pointers forward to next batch
        xyz1 += n * 3;
        xyz2 += m * 3;
        dist += n * 3;
        idx  += n * 3;
    }
}

int main() {
    int b = 1;   // batch size
    int n = 32; // target points per batch
    int m = 8;  // candidate points per batch

    // allocating memory
    float *xyz1 = new float[b * n * 3];
    float *xyz2 = new float[b * m * 3];
    float *dist = new float[b * n * 3];
    int *idx    = new int[b * n * 3];

   
    for (int i = 0; i < b * n * 3; i++) {
        xyz1[i] = (float)(i % 100) * 0.01f;
    }
    for (int i = 0; i < b * m * 3; i++) {
        xyz2[i] = (float)(i % 50) * 0.02f;
    }

    // --- RESET STATS BEFORE KERNEL ---
    m5_reset_stats(0, 0);

    // Function call (Pure hardware execution)
    threenn_cpu(b, n, m, xyz1, xyz2, dist, idx);

    // --- DUMP STATS AFTER KERNEL ---
    m5_dump_stats(0, 0);

    // Demo calculation check (Ensures compiler does not optimize away the loop)
    printf("Point 0 -> 1st NN idx: %d, dist: %f\n", idx[0], dist[0]);
    printf("Point 0 -> 2nd NN idx: %d, dist: %f\n", idx[1], dist[1]);
    printf("Point 0 -> 3rd NN idx: %d, dist: %f\n", idx[2], dist[2]);

    // memory cleanup
    delete[] xyz1;
    delete[] xyz2;
    delete[] dist;
    delete[] idx;

    return 0;
}



