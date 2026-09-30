#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// Find the largest value in one vector chunk and its first lane.
static inline void find_vector_maximum(vfloat32m1_t values, size_t vl, float &maximum, long &lane) {
    vfloat32m1_t initial = __riscv_vfmv_s_f_f32m1(-1.0f, vl);
    vfloat32m1_t reduced = __riscv_vfredmax_vs_f32m1_f32m1(values, initial, vl);

    maximum = __riscv_vfmv_f_s_f32m1_f32(reduced);

    vbool32_t maximum_mask = __riscv_vmfeq_vf_f32m1_b32(values, maximum, vl);
    lane = __riscv_vfirst_m_b32(maximum_mask, vl);
}

// RVV Farthest Point Sampling
void farthestPointSampling_rvv(int b, int n, int m, const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    if (m <= 0) return;

    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        const float *cur_dataset = dataset + i * n * 3;

        int old = 0;
        idxs[0] = old;

        // Initialize every stored minimum distance to a large value.
        int rem_init = n;
        int init_offset = 0;

        while (rem_init > 0) {
            size_t vl = __riscv_vsetvl_e32m1(rem_init);
            vfloat32m1_t large_values = __riscv_vfmv_v_f_f32m1(1e38f, vl);
            __riscv_vse32_v_f32m1(temp + init_offset, large_values, vl);

            rem_init -= vl;
            init_offset += vl;
        }

        // Select the remaining m - 1 centroids.
        for (int j = 1; j < m; ++j) {
            float best = -1.0f;
            int besti = 0;

            float x1 = cur_dataset[old * 3 + 0];
            float y1 = cur_dataset[old * 3 + 1];
            float z1 = cur_dataset[old * 3 + 2];

            int rem_n = n;
            int k_base = 0;

            while (rem_n > 0) {
                size_t vl = __riscv_vsetvl_e32m1(rem_n);

                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 2, stride, vl);

                vfloat32m1_t vdx = __riscv_vfsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfsub_vf_f32m1(vz2, z1, vl);

                vfloat32m1_t distances = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                distances = __riscv_vfmacc_vv_f32m1(distances, vdy, vdy, vl);
                distances = __riscv_vfmacc_vv_f32m1(distances, vdz, vdz, vl);

                vfloat32m1_t minimum_distances = __riscv_vle32_v_f32m1(temp + k_base, vl);
                minimum_distances = __riscv_vfmin_vv_f32m1(minimum_distances, distances, vl);
                __riscv_vse32_v_f32m1(temp + k_base, minimum_distances, vl);

                float chunk_best;
                long chunk_lane;

                find_vector_maximum(minimum_distances, vl, chunk_best, chunk_lane);

                // Only one winner from each vector chunk is checked here.
                if (chunk_lane >= 0 && chunk_best > best) {
                    best = chunk_best;
                    besti = k_base + static_cast<int>(chunk_lane);
                }

                rem_n -= vl;
                k_base += vl;
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
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <KITTI_frame.bin>\n", argv[0]);
        return 1;
    }

    std::vector<float> dataset;

    if (!loadKittiFrame(argv[1], dataset)) {
        return 1;
    }

    int b = 1;
    int n = static_cast<int>(dataset.size() / 3);
    int m = 128;

    if (n < m) {
        std::fprintf(stderr, "ERROR: Frame has only %d points but M=%d\n", n, m);
        return 1;
    }

    std::vector<float> temp(static_cast<size_t>(n));
    std::vector<int> idxs(static_cast<size_t>(m));

    m5_reset_stats(0, 0);

    farthestPointSampling_rvv(b, n, m, dataset.data(), temp.data(), idxs.data());

    m5_dump_stats(0, 0);

    std::printf("POINT_COUNT=%d\n", n);

    for (int i = 0; i < m; ++i) {
        std::printf("FPS_INDEX[%d]=%d\n", i, idxs[i]);
    }

    return 0;
}