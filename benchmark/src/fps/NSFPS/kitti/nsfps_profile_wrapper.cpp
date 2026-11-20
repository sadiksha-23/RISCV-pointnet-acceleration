// Benchmark adapter for the unmodified NS-FPS CPU algorithm:
// https://github.com/satreeby/ns-fps
// Uses upstream src/yuezu_fps.cpp + src/yuezu_fps.h, not a reimplementation.
// Input: KITTI xyzi float32 records; all results report original zero-based indices.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <vector>
#include "yuezu_fps.h"
extern "C" void ns_bbox_begin();
extern "C" void ns_bbox_end();
extern "C" void ns_report();

#ifdef NSFPS_GEM5
extern "C" {
#include <gem5/m5ops.h>
}
#endif

static int parse_int(const char *s, const char *what) {
    char *end = nullptr;
    long x = std::strtol(s, &end, 10);
    if (!s[0] || *end || x <= 0 || x > 100000000) {
        std::fprintf(stderr, "Invalid %s: %s\n", what, s);
        std::exit(2);
    }
    return (int)x;
}

#ifndef NSFPS_XBLOCKS
#define NSFPS_XBLOCKS 16
#endif
#ifndef NSFPS_YBLOCKS
#define NSFPS_YBLOCKS 16
#endif
#ifndef NSFPS_ZBLOCKS
#define NSFPS_ZBLOCKS 16
#endif

int main(int argc, char **argv) {
    if (argc != 1 && argc != 7) {
        std::fprintf(stderr,
          "Usage: %s KITTI.bin N M X_BLOCKS Y_BLOCKS Z_BLOCKS\n", argv[0]);
        return 2;
    }
    // No arguments is used inside gem5 because the project's launch script
    // currently accepts only the executable path. CLI arguments work natively.
    const char *path = argc == 1
        ? "/mnt/c/Users/lsadi/Documents/PCD Research/datasets/kitti/0000000000.bin"
        : argv[1];
    const int n  = argc == 1 ? 100000 : parse_int(argv[2], "N");
    const int m  = argc == 1 ? 128 : parse_int(argv[3], "M");
    const int xb = argc == 1 ? NSFPS_XBLOCKS : parse_int(argv[4], "X_BLOCKS");
    const int yb = argc == 1 ? NSFPS_YBLOCKS : parse_int(argv[5], "Y_BLOCKS");
    const int zb = argc == 1 ? NSFPS_ZBLOCKS : parse_int(argv[6], "Z_BLOCKS");
    auto is_power_of_two = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
    if (m > n || !is_power_of_two(xb) || !is_power_of_two(yb) || !is_power_of_two(zb)) {
        std::fprintf(stderr, "Require 1<=M<=N and power-of-two spatial block counts.\n");
        return 2;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { std::perror(path); return 1; }
    const std::streamoff bytes = file.tellg();
    if (bytes < (std::streamoff)n * 4 * (std::streamoff)sizeof(float)) {
        std::fprintf(stderr, "Input contains fewer than %d KITTI points.\n", n);
        return 1;
    }
    file.seekg(0, std::ios::beg);
    std::vector<Point3D> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i) {
        float xyzi[4];
        if (!file.read(reinterpret_cast<char *>(xyzi), sizeof(xyzi))) {
            std::fprintf(stderr, "Could not read point %d.\n", i);
            return 1;
        }
        pts.emplace_back(xyzi[0], xyzi[1], xyzi[2], (size_t)i);
    }
    // Start benchmark AFTER disk I/O and input decoding. Count bounding-box
    // preparation, NS-FPS structure allocation, cache building and all samples.
#ifdef NSFPS_GEM5
    m5_reset_stats(0, 0);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    // compute_bbox is an upstream function and initializes its Morton LUTs.
    ns_bbox_begin();
    SpaceRange range = compute_bbox(pts, (uint32_t)xb, (uint32_t)yb, (uint32_t)zb);
    ns_bbox_end();
    std::vector<size_t> indices = yuezu_fps(pts, (size_t)m, range);
    const auto t1 = std::chrono::steady_clock::now();
#ifdef NSFPS_GEM5
    m5_dump_stats(0, 0);
#endif
    const double host_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ns_report();
    std::printf("METHOD=NSFPS_UPSTREAM_CPU\n");
    std::printf("N=%d\nM=%d\nBLOCKS=%d,%d,%d\n", n, m, xb, yb, zb);
    std::printf("SAMPLES_RETURNED=%zu\n", indices.size());
#ifdef NSFPS_GEM5
    std::printf("TIME_SOURCE=gem5_stats_simSeconds\n");
#else
    std::printf("NATIVE_MS=%.6f\n", host_elapsed_ms);
#endif
    for (size_t i = 0; i < indices.size(); ++i)
        std::printf("FPS_INDEX[%zu]=%zu\n", i, indices[i]);
    if (indices.size() != (size_t)m) {
        std::fprintf(stderr, "WARNING: NS-FPS returned fewer than M samples.\n");
        return 3;
    }
    return 0;
}
