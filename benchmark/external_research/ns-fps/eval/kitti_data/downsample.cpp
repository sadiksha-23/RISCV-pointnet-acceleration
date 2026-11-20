// downsample_bin.cpp
// 功能：读取 .bin 点云文件，随机下采样，保存为新文件
// 编译: g++ -std=c++17 -O3 downsample_bin.cpp -o downsample_bin

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <unordered_set>
#include <cstring>
#include <filesystem>

/**
 * @brief 随机下采样 .bin 点云文件
 * @param input_file  输入文件路径
 * @param output_file 输出文件路径
 * @param stride      每点浮点数 (3=XYZ, 4=XYZ+intensity)
 * @param target_num  下采样目标点数 (0=保留全部)
 * @param seed        随机种子 (保证可复现)
 * @return 是否成功
 */
bool downsample_bin(const std::string& input_file, 
                    const std::string& output_file,
                    int stride,
                    size_t target_num,
                    unsigned seed = 42) {
  // ========== 1. 打开输入文件 ==========
  std::ifstream infile(input_file, std::ios::binary | std::ios::ate);
  if (!infile.is_open()) {
    std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
    return false;
  }
  
  std::streamsize file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);
  
  if (file_size == 0) {
    std::cerr << "Error: Input file is empty" << std::endl;
    return false;
  }
  
  size_t num_floats = file_size / sizeof(float);
  if (num_floats % stride != 0) {
    std::cerr << "Warning: File size not divisible by stride (" << stride 
              << "), may be corrupted." << std::endl;
  }
  size_t total_points = num_floats / stride;
  std::cout << "Input: " << input_file << " | " 
            << total_points << " points | stride=" << stride << std::endl;

  // ========== 2. 确定实际读取点数 ==========
  size_t read_count = total_points;
  if (target_num > 0 && target_num < total_points) {
    read_count = target_num;
    std::cout << "Downsampling: " << total_points << " -> " << target_num 
              << " points (seed=" << seed << ")" << std::endl;
  } else if (target_num >= total_points) {
    std::cout << "Skip downsampling: target (" << target_num 
              << ") >= original (" << total_points << ")" << std::endl;
  }

  // ========== 3. 生成随机索引集合（如需下采样） ==========
  std::unordered_set<size_t> sample_indices;
  if (read_count < total_points) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, total_points - 1);
    while (sample_indices.size() < read_count) {
      sample_indices.insert(dist(rng));
    }
    std::cout << "Generated " << sample_indices.size() << " random indices" << std::endl;
  }

  // ========== 4. 读取并筛选点 ==========
  std::vector<float> point_buffer(stride);  // 单点缓冲区
  std::vector<float> output_data;
  output_data.reserve(read_count * stride);
  
  size_t accepted = 0;
  for (size_t i = 0; i < total_points; ++i) {
    // 如需下采样且当前点不在集合中，跳过
    if (!sample_indices.empty() && sample_indices.find(i) == sample_indices.end()) {
      infile.seekg(stride * sizeof(float), std::ios::cur);
      continue;
    }
    // 读取当前点
    if (!infile.read(reinterpret_cast<char*>(point_buffer.data()), stride * sizeof(float))) {
      std::cerr << "Error: Failed to read point " << i << std::endl;
      return false;
    }
    // 存入输出缓冲区
    for (int j = 0; j < stride; ++j) {
      output_data.push_back(point_buffer[j]);
    }
    ++accepted;
  }
  infile.close();
  
  if (accepted != read_count) {
    std::cerr << "Warning: Expected " << read_count << " points, but got " << accepted << std::endl;
  }
  std::cout << "Loaded " << accepted << " points" << std::endl;

  // ========== 5. 写入输出文件 ==========
  std::ofstream outfile(output_file, std::ios::binary);
  if (!outfile.is_open()) {
    std::cerr << "Error: Cannot create output file: " << output_file << std::endl;
    return false;
  }
  
  outfile.write(reinterpret_cast<const char*>(output_data.data()), 
                output_data.size() * sizeof(float));
  outfile.close();
  
  std::cout << "Saved " << output_data.size() << " floats to " << output_file << std::endl;
  return true;
}

// ========== 命令行帮助 ==========
void print_help(const char* prog_name) {
  std::cout << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  -i, --input <path>     Input .bin file (required)\n"
            << "  -o, --output <path>    Output .bin file (required)\n"
            << "  -n, --num <N>          Target point count after downsampling\n"
            << "  --stride <3|4>         Floats per point (default: 4, KITTI format)\n"
            << "  --seed <S>             Random seed for reproducibility (default: 42)\n"
            << "  -h, --help             Show this help message\n"
            << "\nExamples:\n"
            << "  " << prog_name << " -i src.bin -o dst.bin -n 10000\n"
            << "  " << prog_name << " -i cloud.bin -o small.bin -n 5000 --stride 3 --seed 123\n";
}

// ========== 主函数 ==========
int main(int argc, char** argv) {
  std::string input_file, output_file;
  size_t target_num = 0;      // 0 = 不下采样
  int stride = 4;
  unsigned seed = 42;

  // 解析参数
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
      input_file = argv[++i];
    } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
      output_file = argv[++i];
    } else if ((arg == "-n" || arg == "--num") && i + 1 < argc) {
      target_num = std::stoul(argv[++i]);
    } else if ((arg == "--stride") && i + 1 < argc) {
      stride = std::stoi(argv[++i]);
      if (stride != 3 && stride != 4) {
        std::cerr << "Error: stride must be 3 or 4" << std::endl;
        return 1;
      }
    } else if ((arg == "--seed") && i + 1 < argc) {
      seed = std::stoul(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      print_help(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_help(argv[0]);
      return 1;
    }
  }

  // 检查必填参数
  if (input_file.empty() || output_file.empty()) {
    std::cerr << "Error: --input and --output are required" << std::endl;
    print_help(argv[0]);
    return 1;
  }

  // 执行下采样
  std::cout << "========================================" << std::endl;
  std::cout << "BIN Point Cloud Downsampling Tool" << std::endl;
  std::cout << "========================================" << std::endl;
  
  bool success = downsample_bin(input_file, output_file, stride, target_num, seed);
  
  std::cout << "========================================" << std::endl;
  if (success) {
    std::cout << "✓ Done!" << std::endl;
    return 0;
  } else {
    std::cerr << "✗ Failed!" << std::endl;
    return 1;
  }
}