#include "yuezu_fps.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

// 简单的参考FPS实现
std::vector<size_t> fps_reference(const std::vector<Point3D>& points, size_t n_samples) {
  size_t N = points.size();
  if (n_samples >= N) {
    std::vector<size_t> result(N);
    for (size_t i = 0; i < N; ++i) result[i] = i;
    return result;
  }

  std::vector<size_t> indices(n_samples);
  std::vector<float> min_dist(N, 1e30f);

  // 固定第一个采样点为0
  indices[0] = 0;

  for (size_t i = 1; i < n_samples; ++i) {
    const Point3D& last = points[indices[i - 1]];

    // 更新距离
    for (size_t j = 0; j < N; ++j) {
      float dx = points[j].x - last.x;
      float dy = points[j].y - last.y;
      float dz = points[j].z - last.z;
      float dist_sq = dx * dx + dy * dy + dz * dz;
      if (dist_sq < min_dist[j]) {
        min_dist[j] = dist_sq;
      }
    }

    // 找最远点
    float max_d = -1.0f;
    size_t max_idx = 0;
    for (size_t j = 0; j < N; ++j) {
      if (min_dist[j] > max_d) {
        max_d = min_dist[j];
        max_idx = j;
      }
    }

    if (max_d <= 0.0f) {
      indices.resize(i);
      break;
    }

    indices[i] = max_idx;
  }

  return indices;
}

int main(int argc, char** argv) {
  std::cout << "==========================================" << std::endl;
  std::cout << "CPU vs Reference Exact Comparison" << std::endl;
  std::cout << "==========================================" << std::endl;

  size_t num_points = 1000;
  size_t num_samples = 100;

  if (argc >= 2) num_points = std::stoul(argv[1]);
  if (argc >= 3) num_samples = std::stoul(argv[2]);

  std::cout << "Point count: " << num_points << std::endl;
  std::cout << "Sample count: " << num_samples << std::endl;

  // 生成随机点云
  std::cout << "\nGenerating points..." << std::endl;
  std::vector<Point3D> points;
  points.reserve(num_points);

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 10.0f);

  float min_x = 1e10f, max_x = -1e10f;
  float min_y = 1e10f, max_y = -1e10f;
  float min_z = 1e10f, max_z = -1e10f;

  for (size_t i = 0; i < num_points; ++i) {
    float x = dist(rng);
    float y = dist(rng);
    float z = dist(rng);
    points.emplace_back(x, y, z, i);

    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
  }

  // 创建SpaceRange
  float eps = 1e-6f;
  SpaceRange range = make_range(
      min_x - eps, max_x + eps,
      min_y - eps, max_y + eps,
      min_z - eps, max_z + eps,
      16, 16, 16);

  // ================== Reference ==================
  std::cout << "\n--- Reference ---" << std::endl;

  auto ref_start = std::chrono::high_resolution_clock::now();
  std::vector<size_t> ref_indices = fps_reference(points, num_samples);
  auto ref_end = std::chrono::high_resolution_clock::now();

  double ref_time = std::chrono::duration<double, std::milli>(ref_end - ref_start).count();
  std::cout << "Time: " << ref_time << " ms" << std::endl;
  std::cout << "Samples: " << ref_indices.size() << std::endl;

  // ================== CPU版本 ==================
  std::cout << "\n--- CPU Version ---" << std::endl;

  auto cpu_start = std::chrono::high_resolution_clock::now();
  std::vector<size_t> cpu_indices = yuezu_fps(points, num_samples, range);
  auto cpu_end = std::chrono::high_resolution_clock::now();

  double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  std::cout << "Time: " << cpu_time << " ms" << std::endl;
  std::cout << "Samples: " << cpu_indices.size() << std::endl;

  // ================== 精确对比索引 ==================
  std::cout << "\n--- Exact Index Comparison ---" << std::endl;

  bool all_match = true;
  size_t first_mismatch = size_t(-1);
  size_t min_len = std::min(ref_indices.size(), cpu_indices.size());

  if (ref_indices.size() != cpu_indices.size()) {
    std::cout << "✗ Sample count mismatch! Ref: " << ref_indices.size()
              << ", CPU: " << cpu_indices.size() << std::endl;
    all_match = false;
  }

  for (size_t i = 0; i < min_len; ++i) {
    if (ref_indices[i] != cpu_indices[i]) {
      all_match = false;
      if (first_mismatch == size_t(-1)) {
        first_mismatch = i;
      }
    }
  }

  if (all_match) {
    std::cout << "✓ All indices match exactly!" << std::endl;
  } else {
    std::cout << "✗ Indices differ!" << std::endl;
    if (first_mismatch != size_t(-1)) {
      std::cout << "  First mismatch at step " << first_mismatch << std::endl;
      std::cout << "  Ref[" << first_mismatch << "] = " << ref_indices[first_mismatch] << std::endl;
      std::cout << "  CPU[" << first_mismatch << "] = " << cpu_indices[first_mismatch] << std::endl;
    }

    // 打印前20个索引对比
    std::cout << "\n  First 20 indices:" << std::endl;
    std::cout << "  Step |  Ref  |  CPU  | Match?" << std::endl;
    std::cout << "  -----+-------+-------+--------" << std::endl;
    for (size_t i = 0; i < std::min(min_len, size_t(20)); ++i) {
      const char* match = (ref_indices[i] == cpu_indices[i]) ? "✓" : "✗";
      std::cout << "  " << std::setw(4) << i << " | "
                << std::setw(5) << ref_indices[i] << " | "
                << std::setw(5) << cpu_indices[i] << " |   "
                << match << std::endl;
    }
  }

  // ================== 速度对比 ==================
  std::cout << "\n--- Speed Comparison ---" << std::endl;
  double speedup = ref_time / cpu_time;
  std::cout << "Reference time: " << ref_time << " ms" << std::endl;
  std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
  std::cout << "Speedup: " << speedup << "x" << std::endl;

  std::cout << "\n==========================================" << std::endl;
  std::cout << "Done!" << std::endl;
  std::cout << "==========================================" << std::endl;

  return all_match ? 0 : 1;
}
