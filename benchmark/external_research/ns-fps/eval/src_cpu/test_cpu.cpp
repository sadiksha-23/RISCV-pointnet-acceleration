#include "yuezu_fps.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
  std::cout << "==========================================" << std::endl;
  std::cout << "Yuezu FPS CPU Test" << std::endl;
  std::cout << "==========================================" << std::endl;

  // 参数
  size_t num_points = 10000;
  size_t num_samples = 1000;

  if (argc >= 2) {
    num_points = std::stoul(argv[1]);
  }
  if (argc >= 3) {
    num_samples = std::stoul(argv[2]);
  }

  std::cout << "Point count: " << num_points << std::endl;
  std::cout << "Sample count: " << num_samples << std::endl;

  // 生成随机点云
  std::cout << "\nGenerating random point cloud..." << std::endl;
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

  std::cout << "  Bounding box: [" << min_x << ", " << max_x << "] x ["
            << min_y << ", " << max_y << "] x [" << min_z << ", " << max_z << "]" << std::endl;

  // 创建SpaceRange
  float eps = 1e-6f;
  SpaceRange range = make_range(
      min_x - eps, max_x + eps,
      min_y - eps, max_y + eps,
      min_z - eps, max_z + eps,
      16, 16, 16);

  // 运行FPS
  std::cout << "\nRunning FPS..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<size_t> result = yuezu_fps(points, num_samples, range);
  auto end = std::chrono::high_resolution_clock::now();

  double total_time = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "  Total time: " << total_time << " ms" << std::endl;
  std::cout << "  Samples obtained: " << result.size() << std::endl;
  std::cout << "  Throughput: " << (num_points * num_samples / total_time / 1000.0) << " M point-samples/sec" << std::endl;

  if (!result.empty()) {
    std::cout << "\nFirst 10 sample indices:" << std::endl;
    for (size_t i = 0; i < std::min(result.size(), size_t(10)); ++i) {
      std::cout << "  [" << i << "] = " << result[i];
      if (i < result.size() - 1) std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "\n==========================================" << std::endl;
  std::cout << "Test completed!" << std::endl;
  std::cout << "==========================================" << std::endl;

  return 0;
}
