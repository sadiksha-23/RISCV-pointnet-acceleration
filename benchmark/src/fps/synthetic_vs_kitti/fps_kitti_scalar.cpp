#include <cmath>
#include <fstream>
#include <vector>
#include <string>
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

static bool loadKittiFrame(const char *filename, std::vector<float> &dataset) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file) {
        std::fprintf(stderr, "ERROR: Could not open KITTI frame: %s\n", filename);
        return false;
    }

    std::streamsize bytes = file.tellg();

    if (bytes <= 0 || bytes % (4 * sizeof(float)) != 0) {
        std::fprintf(stderr, "ERROR: Invalid KITTI frame size: %lld bytes\n", static_cast<long long>(bytes));
        return false;
    }

    file.seekg(0, std::ios::beg);

    size_t point_count = static_cast<size_t>(bytes) / (4 * sizeof(float));
    std::vector<float> raw(point_count * 4);

    if (!file.read(reinterpret_cast<char *>(raw.data()), bytes)) {
        std::fprintf(stderr, "ERROR: Failed while reading KITTI frame\n");
        return false;
    }

    dataset.resize(point_count * 3);

    for (size_t i = 0; i < point_count; ++i) {
        float x = raw[i * 4 + 0];
        float y = raw[i * 4 + 1];
        float z = raw[i * 4 + 2];

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            std::fprintf(stderr, "ERROR: Non-finite coordinate at point %zu\n", i);
            return false;
        }

        dataset[i * 3 + 0] = x;
        dataset[i * 3 + 1] = y;
        dataset[i * 3 + 2] = z;
    }

    return true;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <KITTI_frame.bin> <N> <M>\n", argv[0]);
        return 1;
    }

    std::vector<float> dataset;

    if (!loadKittiFrame(argv[1], dataset)) {
        return 1;
    }

    int b = 1;
    int available_n = static_cast<int>(dataset.size() / 3);
    int n = std::atoi(argv[2]);
    int m = std::atoi(argv[3]);

    if (n <= 0 || m <= 0) {
        std::fprintf(stderr, "ERROR: N and M must both be positive.\n");
        return 1;
    }

    if (n > available_n) {
        std::fprintf(stderr, "ERROR: Requested N=%d but frame has only %d points.\n", n, available_n);
        return 1;
    }

    if (m > n) {
        std::fprintf(stderr, "ERROR: M=%d cannot exceed N=%d.\n", m, n);
        return 1;
    }

    std::vector<float> temp(static_cast<size_t>(n));
    std::vector<int> idxs(static_cast<size_t>(m));

    m5_reset_stats(0, 0);

    farthestPointSampling(b, n, m, dataset.data(), temp.data(), idxs.data());

    m5_dump_stats(0, 0);

    std::printf("AVAILABLE_POINT_COUNT=%d\n", available_n);
    std::printf("POINT_COUNT=%d\n", n);
    std::printf("SAMPLE_COUNT=%d\n", m);

    for (int i = 0; i < m; ++i) {
        std::printf("FPS_INDEX[%d]=%d\n", i, idxs[i]);
    }

    return 0;
}