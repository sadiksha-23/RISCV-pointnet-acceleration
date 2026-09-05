#include <iostream>
#include <vector>
#include <cmath>
extern "C" {
    #include <gem5/m5ops.h>
}

// original PointNet++ 3NN (3-Nearest Neighbors) Code
void threenn_cpu(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
     for (int i=0;i<b;++i) { // going through each batch
        for (int j=0;j<n;++j) { // going through every target point
            // loading x, y, z values from xyz1
            float x1=xyz1[j*3+0]; 
            float y1=xyz1[j*3+1];
            float z1=xyz1[j*3+2];
                
            double best1=1e40; double best2=1e40; double best3=1e40; // placeholder for 3 smallest distance
            int besti1=0; int besti2=0; int besti3=0; // storing the index of the closest points

            for (int k=0;k<m;++k) { // going through every candidate point
                // loading x, y, z values from xyz1
                float x2=xyz2[k*3+0];
                float y2=xyz2[k*3+1];
                float z2=xyz2[k*3+2];
                
                // calculating the distance
                float dx=x1-x2;
                float dy=y1-y2;
                float dz=z1-z2;
                double d=dx*dx+dy*dy+dz*dz;

                // updating the top 3 neighbor tracking list
                if (d<best1) {
                    best3=best2;
                    besti3=besti2;
                    best2=best1;
                    besti2=besti1;
                    best1=d;
                    besti1=k;
                } else if (d<best2) {
                    best3=best2;
                    besti3=besti2;
                    best2=d;
                    besti2=k;
                } else if (d<best3) {
                    best3=d;
                    besti3=k;
                }
            } 
            // storing the best distance and their indices
            dist[j*3]=best1;
            idx[j*3]=besti1;
            dist[j*3+1]=best2;
            idx[j*3+1]=besti2;
            dist[j*3+2]=best3;
            idx[j*3+2]=besti3;
        } 
        // moving memory pointers forward to the next batch
        xyz1+=n*3;
        xyz2+=m*3;
        dist+=n*3;
        idx+=n*3;
    }
} 

int main() {
    int b = 1;   // batch size
    int n = 1024; // target points per batch
    int m = 128;  // candidate points per batch

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



