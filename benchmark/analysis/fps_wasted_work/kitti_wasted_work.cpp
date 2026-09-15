#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>

struct WorkCounts {
    unsigned long long total = 0;
    unsigned long long effective = 0;
    unsigned long long ineffective = 0;
};

bool loadKittiFrame(const char *path, std::vector<float> &dataset) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        std::fprintf(stderr, "ERROR: Could not open %s\n", path);
        return false;
    }

    std::streamsize bytes = file.tellg();

    if (bytes <= 0 || bytes % 16 != 0) {
        std::fprintf(
            stderr,
            "ERROR: KITTI frame size must be a positive multiple of 16 bytes.\n"
        );
        return false;
    }

    file.seekg(0, std::ios::beg);

    size_t available_points = static_cast<size_t>(bytes / 16);
    std::vector<float> raw(available_points * 4);

    if (!file.read(
            reinterpret_cast<char *>(raw.data()),
            static_cast<std::streamsize>(raw.size() * sizeof(float))
        )) {
        std::fprintf(stderr, "ERROR: Could not read the complete KITTI frame.\n");
        return false;
    }

    dataset.resize(available_points * 3);

    for (size_t i = 0; i < available_points; ++i) {
        float x = raw[i * 4 + 0];
        float y = raw[i * 4 + 1];
        float z = raw[i * 4 + 2];

        if (!std::isfinite(x) ||
            !std::isfinite(y) ||
            !std::isfinite(z)) {
            std::fprintf(
                stderr,
                "ERROR: Non-finite coordinate at point %zu.\n",
                i
            );
            return false;
        }

        dataset[i * 3 + 0] = x;
        dataset[i * 3 + 1] = y;
        dataset[i * 3 + 2] = z;
    }

    return true;
}

WorkCounts analyzeFPS(
    int n,
    int m,
    const float *dataset,
    float *temp,
    int *idxs,
    const char *csv_path
) {
    WorkCounts totals;

    std::ofstream csv(csv_path);
    if (!csv) {
        std::fprintf(stderr, "ERROR: Could not create %s\n", csv_path);
        std::exit(1);
    }

    csv << "iteration,reference_index,selected_index,total_checks,"
           "effective_updates,ineffective_updates,"
           "effective_percent,ineffective_percent\n";

    int old = 0;
    idxs[0] = old;

    for (int k = 0; k < n; ++k) {
        temp[k] = 1e38f;
    }

    for (int j = 1; j < m; ++j) {
        int reference_index = old;
        int besti = 0;
        float best = -1.0f;

        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];

        unsigned long long effective = 0;
        unsigned long long ineffective = 0;

        for (int k = 0; k < n; ++k) {
            float x2 = dataset[k * 3 + 0];
            float y2 = dataset[k * 3 + 1];
            float z2 = dataset[k * 3 + 2];

            float dx = x2 - x1;
            float dy = y2 - y1;
            float dz = z2 - z1;
            float d = dx * dx + dy * dy + dz * dz;

            if (d < temp[k]) {
                temp[k] = d;
                ++effective;
            } else {
                ++ineffective;
            }

            if (temp[k] > best) {
                best = temp[k];
                besti = k;
            }
        }

        old = besti;
        idxs[j] = old;

        unsigned long long checks = effective + ineffective;
        double effective_percent =
            100.0 * static_cast<double>(effective) /
            static_cast<double>(checks);
        double ineffective_percent =
            100.0 * static_cast<double>(ineffective) /
            static_cast<double>(checks);

        csv << j << ','
            << reference_index << ','
            << besti << ','
            << checks << ','
            << effective << ','
            << ineffective << ','
            << std::fixed << std::setprecision(6)
            << effective_percent << ','
            << ineffective_percent << '\n';

        totals.total += checks;
        totals.effective += effective;
        totals.ineffective += ineffective;
    }

    return totals;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(
            stderr,
            "Usage: %s <KITTI_frame.bin> <N> <M> <output.csv>\n",
            argv[0]
        );
        return 1;
    }

    const char *frame_path = argv[1];
    int n = std::atoi(argv[2]);
    int m = std::atoi(argv[3]);
    const char *csv_path = argv[4];

    std::vector<float> dataset;

    if (!loadKittiFrame(frame_path, dataset)) {
        return 1;
    }

    int available_n = static_cast<int>(dataset.size() / 3);

    if (n <= 0 || m <= 0 || m > n) {
        std::fprintf(stderr, "ERROR: Require N > 0 and 0 < M <= N.\n");
        return 1;
    }

    if (n > available_n) {
        std::fprintf(
            stderr,
            "ERROR: Requested N=%d but frame has only %d points.\n",
            n,
            available_n
        );
        return 1;
    }

    std::vector<float> temp(static_cast<size_t>(n));
    std::vector<int> idxs(static_cast<size_t>(m));

    WorkCounts totals = analyzeFPS(
        n,
        m,
        dataset.data(),
        temp.data(),
        idxs.data(),
        csv_path
    );

    double effective_percent =
        100.0 * static_cast<double>(totals.effective) /
        static_cast<double>(totals.total);
    double ineffective_percent =
        100.0 * static_cast<double>(totals.ineffective) /
        static_cast<double>(totals.total);

    std::printf("DATASET=kitti\n");
    std::printf("FRAME=%s\n", frame_path);
    std::printf("AVAILABLE_N=%d\n", available_n);
    std::printf("N=%d\n", n);
    std::printf("M=%d\n", m);
    std::printf("TOTAL_CHECKS=%llu\n", totals.total);
    std::printf("EFFECTIVE_UPDATES=%llu\n", totals.effective);
    std::printf("INEFFECTIVE_UPDATES=%llu\n", totals.ineffective);
    std::printf("EFFECTIVE_PERCENT=%.6f\n", effective_percent);
    std::printf("INEFFECTIVE_PERCENT=%.6f\n", ineffective_percent);
    std::printf("FIRST_INDEX=%d\n", idxs.front());
    std::printf("LAST_INDEX=%d\n", idxs.back());
    std::printf("CSV=%s\n", csv_path);

    return 0;
}
