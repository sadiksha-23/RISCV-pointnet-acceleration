#include "yuezu_fps.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <filesystem>

// 简单的参考FPS实现（纯CPU，用于对比验证）
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

// ============ 新增：读取.bin点云文件 ============
/**
 * @brief 读取二进制点云文件（每点4个float: x,y,z,intensity 或 3个float: x,y,z）
 * @param filename 输入文件路径
 * @param points 输出点云容器
 * @param stride 每个点的浮点数个数，默认4（KITTI格式），可设为3
 * @return 是否读取成功
 */
bool read_bin_pointcloud(const std::string& filename, std::vector<Point3D>& points, int stride = 4) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file: " << filename << std::endl;
    return false;
  }

  // 获取文件大小
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (file_size == 0) {
    std::cerr << "Error: File is empty: " << filename << std::endl;
    return false;
  }

  // 计算点数
  size_t num_floats = file_size / sizeof(float);
  if (num_floats % stride != 0) {
    std::cerr << "Warning: File size not divisible by stride (" << stride << "), may be corrupted." << std::endl;
  }
  size_t num_points = num_floats / stride;

  std::cout << "Reading " << num_points << " points from " << filename 
            << " (stride=" << stride << ", " << num_floats << " floats)" << std::endl;

  points.reserve(num_points);

  // 批量读取所有浮点数
  std::vector<float> buffer(num_floats);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
    std::cerr << "Error: Failed to read file content." << std::endl;
    return false;
  }
  file.close();

  // 解析为Point3D
  float min_x = 1e10f, max_x = -1e10f;
  float min_y = 1e10f, max_y = -1e10f;
  float min_z = 1e10f, max_z = -1e10f;

  for (size_t i = 0; i < num_points; ++i) {
    size_t base = i * stride;
    float x = buffer[base];
    float y = buffer[base + 1];
    float z = buffer[base + 2];
    
    points.emplace_back(x, y, z, i);  // 假设Point3D构造函数为 (x, y, z, index)

    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
  }

  std::cout << "Point cloud bounds: "
            << "X[" << min_x << ", " << max_x << "], "
            << "Y[" << min_y << ", " << max_y << "], "
            << "Z[" << min_z << ", " << max_z << "]" << std::endl;

  return true;
}

// ============ 辅助函数：生成随机点云（保留用于测试） ============
std::vector<Point3D> generate_random_points(size_t num_points, unsigned seed = 42) {
  std::vector<Point3D> points;
  points.reserve(num_points);

  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 10.0f);

  for (size_t i = 0; i < num_points; ++i) {
    float x = dist(rng);
    float y = dist(rng);
    float z = dist(rng);
    points.emplace_back(x, y, z, i);
  }
  return points;
}

int main(int argc, char** argv) {
  std::cout << "==========================================" << std::endl;
  std::cout << "CPU vs Reference Exact Comparison" << std::endl;
  std::cout << "==========================================" << std::endl;

  // ============ 命令行参数解析 ============
  std::string bin_file;          // .bin点云文件路径（可选）
  size_t num_points = 1000;      // 随机生成时的点数（仅当未指定文件时生效）
  size_t num_samples = 100;      // 采样点数
  int stride = 4;                // 每个点的float数量: 3或4

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
      bin_file = argv[++i];
    } else if ((arg == "-n" || arg == "--num-points") && i + 1 < argc) {
      num_points = std::stoul(argv[++i]);
    } else if ((arg == "-s" || arg == "--num-samples") && i + 1 < argc) {
      num_samples = std::stoul(argv[++i]);
    } else if ((arg == "--stride") && i + 1 < argc) {
      stride = std::stoi(argv[++i]);
      if (stride != 3 && stride != 4) {
        std::cerr << "Error: stride must be 3 or 4" << std::endl;
        return 1;
      }
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -f, --file <path>      Path to .bin point cloud file (optional)" << std::endl;
      std::cout << "  -n, --num-points <N>   Number of random points if no file given (default: 1000)" << std::endl;
      std::cout << "  -s, --num-samples <M>  Number of FPS samples (default: 100)" << std::endl;
      std::cout << "  --stride <3|4>         Floats per point in .bin file (default: 4, KITTI format)" << std::endl;
      std::cout << "  -h, --help             Show this help message" << std::endl;
      return 0;
    }
  }

  std::cout << "Sample count: " << num_samples << std::endl;

  // ============ 加载点云数据 ============
  std::vector<Point3D> points;
  
  if (!bin_file.empty()) {
    // 读取指定.bin文件
    std::cout << "\nLoading point cloud from file: " << bin_file << std::endl;
    if (!read_bin_pointcloud(bin_file, points, stride)) {
      std::cerr << "Failed to load point cloud!" << std::endl;
      return 1;
    }

    // 删除z<-5的点
    size_t valid_count = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        // 保留 z >= -5 的点
        if (points[i].z >= -5.0f) {
            if (valid_count != i) {
                points[valid_count] = points[i];
            }
            valid_count++;
        }
    }
    // 截断vector到有效点数量
    points.resize(valid_count);
    
    num_points = points.size();  // 更新实际点数
  } else {
    // 生成随机点云（原逻辑）
    std::cout << "\nGenerating " << num_points << " random points..." << std::endl;
    points = generate_random_points(num_points);
  }

  std::cout << "Point count: " << points.size() << std::endl;

  // ============ 计算SpaceRange ============
  if (points.empty()) {
    std::cerr << "Error: No points to process!" << std::endl;
    return 1;
  }

  float min_x = points[0].x, max_x = points[0].x;
  float min_y = points[0].y, max_y = points[0].y;
  float min_z = points[0].z, max_z = points[0].z;
  for (const auto& p : points) {
    min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
    min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
  }

  float eps = 1e-6f;
  SpaceRange range = make_range(
      min_x - eps, max_x + eps,
      min_y - eps, max_y + eps,
      min_z - eps, max_z + eps,
      128, 128, 2);

  // ================== Reference ==================
  std::cout << "\n--- Reference ---" << std::endl;

  auto ref_start = std::chrono::high_resolution_clock::now();
  std::vector<size_t> ref_indices = fps_reference(points, num_samples);
  auto ref_end = std::chrono::high_resolution_clock::now();

  double ref_time = std::chrono::duration<double, std::milli>(ref_end - ref_start).count();
  std::cout << "Time: " << ref_time << " ms" << std::endl;
  std::cout << "Samples: " << ref_indices.size() << std::endl;

  // ================== CPU版本 ==================
  std::cout << "\n--- CPU Version (yuezu_fps) ---" << std::endl;

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
  double speedup = (cpu_time > 1e-6) ? ref_time / cpu_time : 0;
  std::cout << "Reference time: " << ref_time << " ms" << std::endl;
  std::cout << "CPU (yuezu_fps) time: " << cpu_time << " ms" << std::endl;
  if (cpu_time > 1e-6) {
    std::cout << "Speedup: " << speedup << "x" << std::endl;
  } else {
    std::cout << "Speedup: N/A (CPU time too small)" << std::endl;
  }

  std::cout << "\n==========================================" << std::endl;
  std::cout << "Done!" << std::endl;
  std::cout << "==========================================" << std::endl;

  return all_match ? 0 : 1;
}