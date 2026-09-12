#include <cstdio>
#include <cstring>

extern "C" {
    #include <gem5/m5ops.h>
}

void farthestPointSampling (int b, int n, int m, const float * __restrict__ dataset, float * __restrict__ temp, int * __restrict__ idxs) {
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

int main() {
    int b = 1, n = 1024, m = 128;

    float *dataset = new float[b * n * 3];
    float *temp = new float[b * n];
    int *idxs = new int[b * m];

    for (int i = 0; i < b * n * 3; ++i) {
        dataset[i] = (float)(i % 100) * 0.01f;
    }

    memset(temp, 0, sizeof(float) * b * n);
    memset(idxs, 0, sizeof(int) * b * m);

    m5_reset_stats(0, 0);

    farthestPointSampling(b, n, m, dataset, temp, idxs);

    m5_dump_stats(0, 0);

    printf("First FPS index: %d\n", idxs[0]);
    printf("Last FPS index: %d\n", idxs[m - 1]);

    delete[] dataset;
    delete[] temp;
    delete[] idxs;

    return 0;
}