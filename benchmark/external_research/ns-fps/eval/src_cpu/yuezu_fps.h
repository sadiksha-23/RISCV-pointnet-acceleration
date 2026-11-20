#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// =============================================================================
// 配置宏定义 - 可调参数
// =============================================================================

// 每个MortonBlock存储的点数（必须是2的幂，影响缓存行对齐）
#ifndef MORTON_BLOCK_SIZE
#define MORTON_BLOCK_SIZE 32
#endif

// 每个CacheBlock的子节点数（16叉树）
#ifndef CACHE_BLOCK_SIZE
#define CACHE_BLOCK_SIZE 16
#endif

// 默认各维度分块数（2的幂）
#ifndef DEFAULT_X_BLOCKS
#define DEFAULT_X_BLOCKS 16
#endif

#ifndef DEFAULT_Y_BLOCKS
#define DEFAULT_Y_BLOCKS 16
#endif

#ifndef DEFAULT_Z_BLOCKS
#define DEFAULT_Z_BLOCKS 16
#endif

// 边界 epsilon，防止浮点精度问题
#ifndef BOUNDARY_EPS
#define BOUNDARY_EPS 1e-6f
#endif

// 初始距离值（无穷大）
#ifndef INF_DISTANCE
#define INF_DISTANCE 1e30f
#endif

// =============================================================================
// 数据结构声明
// =============================================================================

struct SpaceRange {
  float min_x, max_x;
  float min_y, max_y;
  float min_z, max_z;
  uint32_t x_blocks, y_blocks, z_blocks;
  float inv_bs_x;
  float inv_bs_y;
  float inv_bs_z;
  uint32_t x_bits_;
  uint32_t y_bits_;
  uint32_t z_bits_;
  uint32_t z_shift_;

  // 预计算的查找表：lut[val] = 位扩展后的值
  std::vector<uint32_t> x_lut_;
  std::vector<uint32_t> y_lut_;
  std::vector<uint32_t> z_lut_;

  float block_size_x() const;
  float block_size_y() const;
  float block_size_z() const;
  
  uint32_t x_bits() const;
  uint32_t y_bits() const;
  uint32_t z_bits() const;
  uint32_t total_bits() const;
  size_t total_blocks() const;
};

struct Point3D {
  float x, y, z;
  size_t original_index;

  Point3D(float x_ = 0, float y_ = 0, float z_ = 0, size_t idx = 0);

  std::tuple<uint32_t, uint32_t, uint32_t>
  compute_block_indices(const SpaceRange &range) const;

  uint32_t compute_morton_code(const SpaceRange &range) const;
};

// =============================================================================
// 工具函数声明
// =============================================================================

static inline float squared_distance(float x1, float y1, float z1, float x2, float y2,
                       float z2);

SpaceRange make_uniform_range(float min_x, float max_x, float min_y,
                              float max_y, float min_z, float max_z,
                              uint32_t blocks_per_dim);

SpaceRange make_range(float min_x, float max_x, float min_y, float max_y,
                      float min_z, float max_z, uint32_t x_blocks,
                      uint32_t y_blocks, uint32_t z_blocks);

SpaceRange compute_bbox(const std::vector<Point3D> &points,
                        uint32_t blocks_per_dim = DEFAULT_X_BLOCKS);

SpaceRange compute_bbox(const std::vector<Point3D> &points, uint32_t x_blocks,
                        uint32_t y_blocks, uint32_t z_blocks);

// =============================================================================
// 核心类声明
// =============================================================================

class MortonBlock {
public:
  static constexpr size_t kBlockSize = MORTON_BLOCK_SIZE;

  MortonBlock();
  bool add_point(float x, float y, float z, size_t original_index);
  size_t get_count() const;
  bool is_full() const;
  void update_distance(float qx, float qy, float qz);
  float get_max_distance() const;
  Point3D get_farthest_point() const;

private:
  std::array<Point3D, kBlockSize> points_;
  std::array<float, kBlockSize> distance_cache_;
  size_t count_;
  float max_dist_;
  size_t max_index_;
};

struct PageTableEntry {
  std::vector<std::unique_ptr<MortonBlock>> blocks;
  size_t point_count = 0;
  void allocate(float x, float y, float z, size_t idx);
};

struct OccupiedTableEntry {
  std::unique_ptr<PageTableEntry> page_table_ptr;
  bool occupied = false;
  void allocate(float x, float y, float z, size_t idx);
  bool is_occupied() const;
};

class CacheBlock {
public:
  static constexpr size_t kBlockSize = CACHE_BLOCK_SIZE;

  CacheBlock();
  float get_max_distance() const;
  Point3D get_farthest_point() const;
  std::pair<float, Point3D> get_max() const;
  void rebuild_from_leaves(const std::vector<MortonBlock *> &blocks);
  void rebuild_from_internal(const std::vector<CacheBlock *> &blocks);
  void set_internal_child(size_t index, CacheBlock *child);
  std::array<CacheBlock *, kBlockSize> &get_internal_children();
  std::array<MortonBlock *, kBlockSize> &get_leaf_children();
  CacheBlock *get_parent() const;

private:
  friend class MortonStructure;

  bool is_leaf_level_;
  CacheBlock *parent_;
  size_t parent_slot_index_;
  std::array<float, kBlockSize> max_distances_;
  std::array<Point3D, kBlockSize> farthest_points_;
  float global_max_distance_;
  size_t global_max_index_;
  std::array<CacheBlock *, kBlockSize> internal_children_;
  std::array<MortonBlock *, kBlockSize> leaf_children_;
};

class MortonStructure {
public:
  explicit MortonStructure(const SpaceRange &range);
  ~MortonStructure();

  void allocate(const Point3D &point);
  void build_multilevel_cache(float qx, float qy, float qz);
  std::vector<MortonBlock *> update_morton_code_blocks(uint32_t id, float qx,
                                                       float qy, float qz);
  void
  update_cache_levels_from_blocks(const std::vector<MortonBlock *> &blocks);
  std::pair<float, Point3D> get_global_max_from_cache() const;
  std::vector<uint32_t> get_morton_codes_in_range(float qx, float qy, float qz,
                                                  float distance_sq) const;

private:
  //uint32_t encode_block(uint32_t x, uint32_t y, uint32_t z) const;
  void propagate_updates_upward(std::vector<CacheBlock *> &current);

  SpaceRange range_;
  std::vector<OccupiedTableEntry> occupied_table_;
  std::vector<std::vector<CacheBlock *>> cache_levels_;
  std::vector<std::vector<MortonBlock *>> leaf_blocks_per_l1_;
  std::unordered_map<MortonBlock *, size_t> leaf_block_to_l1_index_;
  bool cache_built_;
};

std::vector<size_t> yuezu_fps(const std::vector<Point3D> &point_cloud,
                              size_t sample_count, const SpaceRange &range);