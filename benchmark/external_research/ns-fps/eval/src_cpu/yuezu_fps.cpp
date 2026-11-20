#include "yuezu_fps.h"

// =============================================================================
// SpaceRange 实现
// =============================================================================

float SpaceRange::block_size_x() const { return (max_x - min_x) / x_blocks; }

float SpaceRange::block_size_y() const { return (max_y - min_y) / y_blocks; }

float SpaceRange::block_size_z() const { return (max_z - min_z) / z_blocks; }



uint32_t SpaceRange::x_bits() const {
  uint32_t bits = 0;
  while ((1u << bits) < x_blocks)
    ++bits;
  return bits ? bits : 1;
}

uint32_t SpaceRange::y_bits() const {
  uint32_t bits = 0;
  while ((1u << bits) < y_blocks)
    ++bits;
  return bits ? bits : 1;
}

uint32_t SpaceRange::z_bits() const {
  uint32_t bits = 0;
  while ((1u << bits) < z_blocks)
    ++bits;
  return bits ? bits : 1;
}


uint32_t SpaceRange::total_bits() const {
  return x_bits() + y_bits() + z_bits();
}

size_t SpaceRange::total_blocks() const {
  return static_cast<size_t>(x_blocks) * y_blocks * z_blocks;
}

// =============================================================================
// 便捷构造函数
// =============================================================================
static inline uint32_t _calc_bits(uint32_t blocks) {
  if (blocks <= 1) return 1;
#ifdef _MSC_VER
  unsigned long index;
  _BitScanReverse(&index, blocks - 1); // MSVC 等效指令
  return index + 1;
#else
  return 32 - __builtin_clz(blocks - 1); // GCC/Clang 快速位运算
#endif
}
// 内部工具：根据 bits 生成 LUT
static void _generate_morton_lut(uint32_t bits, const std::vector<uint32_t>& pos_list, std::vector<uint32_t>& lut) {
  uint32_t count = 1u << bits;
  lut.resize(count);
  for (uint32_t v = 0; v < count; ++v) {
      uint32_t code = 0;
      for (uint32_t k = 0; k < bits; ++k) {
          if (v & (1u << k)) {
              code |= 1u << pos_list[k];
          }
      }
      lut[v] = code;
  }
}

// 内部工具：生成 x/y/z 每一位的目标位置（完全对齐你原来的莫顿码顺序）
static void _compute_bit_positions(uint32_t xb, uint32_t yb, uint32_t zb,
                                std::vector<uint32_t>& x_pos,
                                std::vector<uint32_t>& y_pos,
                                std::vector<uint32_t>& z_pos) {
  uint32_t bit = 0;
  uint32_t common = std::min({xb, yb, zb});
  uint32_t maxb = std::max({xb, yb, zb});

  x_pos.clear();
  y_pos.clear();
  z_pos.clear();

  // 阶段1：公共位段 x→y→z
  for (uint32_t i = 0; i < common; ++i) {
      x_pos.push_back(bit++);
      y_pos.push_back(bit++);
      z_pos.push_back(bit++);
  }

  // 阶段2：剩余位（保持你原来的顺序）
  for (uint32_t i = common; i < maxb; ++i) {
      if (i < xb) x_pos.push_back(bit++);
      if (i < yb) y_pos.push_back(bit++);
      if (i < zb) z_pos.push_back(bit++);
  }
}
SpaceRange make_uniform_range(float min_x, float max_x, float min_y,
  float max_y, float min_z, float max_z,
  uint32_t blocks_per_dim) {
    const float inv_bs_x = 1.0f / ((max_x - min_x) / (float)blocks_per_dim);
    const float inv_bs_y = 1.0f / ((max_y - min_y) / (float)blocks_per_dim);
    const float inv_bs_z = 1.0f / ((max_z - min_z) / (float)blocks_per_dim);

    const uint32_t xb = _calc_bits(blocks_per_dim);
    const uint32_t yb = _calc_bits(blocks_per_dim);
    const uint32_t zb = _calc_bits(blocks_per_dim);

    // 构建基础结构体
    SpaceRange range = {
      min_x, max_x,
      min_y, max_y,
      min_z, max_z,
      blocks_per_dim, blocks_per_dim, blocks_per_dim,
      inv_bs_x, inv_bs_y, inv_bs_z,
      xb, yb, zb
  };

  // 自动生成莫顿码 LUT
  std::vector<uint32_t> x_pos, y_pos, z_pos;
  _compute_bit_positions(xb, yb, zb, x_pos, y_pos, z_pos);
  _generate_morton_lut(xb, x_pos, range.x_lut_);
  _generate_morton_lut(yb, y_pos, range.y_lut_);
  _generate_morton_lut(zb, z_pos, range.z_lut_);

  return range;
}

SpaceRange make_range(float min_x, float max_x, float min_y, float max_y,
                      float min_z, float max_z, uint32_t x_blocks,
                      uint32_t y_blocks, uint32_t z_blocks) {
  
    const float inv_bs_x = 1.0f / ((max_x - min_x) / (float)x_blocks);
    const float inv_bs_y = 1.0f / ((max_y - min_y) / (float)y_blocks);
    const float inv_bs_z = 1.0f / ((max_z - min_z) / (float)z_blocks);

    const uint32_t xb = _calc_bits(x_blocks);
    const uint32_t yb = _calc_bits(y_blocks);
    const uint32_t zb = _calc_bits(z_blocks);

  // 构建基础结构体
  SpaceRange range = {
    min_x, max_x,
    min_y, max_y,
    min_z, max_z,
    x_blocks, y_blocks, z_blocks,
    inv_bs_x, inv_bs_y, inv_bs_z,
    xb, yb, zb
  };

  // 自动生成莫顿码 LUT
  std::vector<uint32_t> x_pos, y_pos, z_pos;
  _compute_bit_positions(xb, yb, zb, x_pos, y_pos, z_pos);
  _generate_morton_lut(xb, x_pos, range.x_lut_);
  _generate_morton_lut(yb, y_pos, range.y_lut_);
  _generate_morton_lut(zb, z_pos, range.z_lut_);

  return range;
}

// =============================================================================
// Point3D 实现
// =============================================================================

Point3D::Point3D(float x_, float y_, float z_, size_t idx)
    : x(x_), y(y_), z(z_), original_index(idx) {}

std::tuple<uint32_t, uint32_t, uint32_t>
Point3D::compute_block_indices(const SpaceRange &range) const {
  float cx = std::max(range.min_x, std::min(x, range.max_x - BOUNDARY_EPS));
  float cy = std::max(range.min_y, std::min(y, range.max_y - BOUNDARY_EPS));
  float cz = std::max(range.min_z, std::min(z, range.max_z - BOUNDARY_EPS));

  // 建议 SpaceRange 预存逆元，将除法转乘法
  uint32_t ix = static_cast<uint32_t>((cx - range.min_x) * range.inv_bs_x);
  uint32_t iy = static_cast<uint32_t>((cy - range.min_y) * range.inv_bs_y);
  uint32_t iz = static_cast<uint32_t>((cz - range.min_z) * range.inv_bs_z);

  // uint32_t ix =
  //     static_cast<uint32_t>((cx - range.min_x) / range.block_size_x());
  // uint32_t iy =
  //     static_cast<uint32_t>((cy - range.min_y) / range.block_size_y());
  // uint32_t iz =
  //     static_cast<uint32_t>((cz - range.min_z) / range.block_size_z());

  ix = std::min(ix, range.x_blocks - 1);
  iy = std::min(iy, range.y_blocks - 1);
  iz = std::min(iz, range.z_blocks - 1);

  return {ix, iy, iz};
}


uint32_t Point3D::compute_morton_code(const SpaceRange &range) const {
  auto [ix, iy, iz] = compute_block_indices(range);

  return range.x_lut_[ix] | range.y_lut_[iy] | range.z_lut_[iz];
}



// =============================================================================
// 工具函数实现
// =============================================================================

static inline float squared_distance(float x1, float y1, float z1, float x2, float y2,
                       float z2) {
  float dx = x1 - x2, dy = y1 - y2, dz = z1 - z2;
  return dx * dx + dy * dy + dz * dz;
}

// =============================================================================
// MortonBlock 实现
// =============================================================================

MortonBlock::MortonBlock() : count_(0), max_dist_(0.0f), max_index_(0) {
  std::fill(distance_cache_.begin(), distance_cache_.end(),
            std::numeric_limits<float>::max());
}

bool MortonBlock::add_point(float x, float y, float z, size_t original_index) {
  if (count_ < kBlockSize) {
    points_[count_] = Point3D(x, y, z, original_index);
    distance_cache_[count_] = std::numeric_limits<float>::max();
    ++count_;
    return true;
  }
  return false;
}

size_t MortonBlock::get_count() const { return count_; }

bool MortonBlock::is_full() const { return count_ == kBlockSize; }

void MortonBlock::update_distance(float qx, float qy, float qz) {
  const size_t cnt = count_;
  if (cnt == 0) {
    max_dist_ = 0.0f;
    max_index_ = 0;
    return;
  }

  Point3D* __restrict__ pts = points_.data();        // 提示无内存重叠
  float *dist_cache = distance_cache_.data();

  float local_max = -1.0f;
  size_t local_max_idx = 0;

  for (size_t i = 0; i < cnt; ++i) {
    const Point3D &p = pts[i];
    float d = squared_distance(p.x, p.y, p.z, qx, qy, qz);

    float old_d = dist_cache[i];
    if (d < old_d)
      old_d = d;
    dist_cache[i] = old_d;

    if (old_d > local_max) {
      local_max = old_d;
      local_max_idx = i;
    }
  }

  max_dist_ = local_max;
  max_index_ = local_max_idx;
}

float MortonBlock::get_max_distance() const { return max_dist_; }

Point3D MortonBlock::get_farthest_point() const {
  return (count_ == 0) ? Point3D() : points_[max_index_];
}

// =============================================================================
// PageTableEntry 实现
// =============================================================================

void PageTableEntry::allocate(float x, float y, float z, size_t idx) {
  if (blocks.empty() || blocks.back()->is_full()) {
    blocks.push_back(std::make_unique<MortonBlock>());
  }
  blocks.back()->add_point(x, y, z, idx);
  ++point_count;
}

// =============================================================================
// OccupiedTableEntry 实现
// =============================================================================

void OccupiedTableEntry::allocate(float x, float y, float z, size_t idx) {
  if (!occupied) {
    page_table_ptr = std::make_unique<PageTableEntry>();
    occupied = true;
  }
  page_table_ptr->allocate(x, y, z, idx);
}

bool OccupiedTableEntry::is_occupied() const { return occupied; }

// =============================================================================
// CacheBlock 实现
// =============================================================================

CacheBlock::CacheBlock()
    : is_leaf_level_(false), parent_(nullptr), parent_slot_index_(0),
      global_max_distance_(0.0f), global_max_index_(0) {
  std::fill(max_distances_.begin(), max_distances_.end(), 0.0f);
  std::fill(farthest_points_.begin(), farthest_points_.end(), Point3D{});
  std::fill(internal_children_.begin(), internal_children_.end(), nullptr);
  std::fill(leaf_children_.begin(), leaf_children_.end(), nullptr);
}

float CacheBlock::get_max_distance() const { return global_max_distance_; }

Point3D CacheBlock::get_farthest_point() const {
  return farthest_points_[global_max_index_];
}

std::pair<float, Point3D> CacheBlock::get_max() const {
  return {global_max_distance_, farthest_points_[global_max_index_]};
}

void CacheBlock::rebuild_from_leaves(const std::vector<MortonBlock *> &blocks) {
  is_leaf_level_ = true;
  float gmax = -1.0f;
  size_t gidx = 0;

  for (size_t i = 0; i < kBlockSize; ++i) {
    if (i < blocks.size() && blocks[i]) {
      MortonBlock *b = blocks[i];
      leaf_children_[i] = b;
      float md = b->get_max_distance();
      max_distances_[i] = md;
      farthest_points_[i] = b->get_farthest_point();
    } else {
      leaf_children_[i] = nullptr;
      max_distances_[i] = 0.0f;
      farthest_points_[i] = Point3D{};
    }
    if (max_distances_[i] > gmax) {
      gmax = max_distances_[i];
      gidx = i;
    }
  }
  global_max_distance_ = (gmax < 0.0f) ? 0.0f : gmax;
  global_max_index_ = gidx;
}

void CacheBlock::rebuild_from_internal(
    const std::vector<CacheBlock *> &blocks) {
  is_leaf_level_ = false;
  float gmax = -1.0f;
  size_t gidx = 0;

  for (size_t i = 0; i < kBlockSize; ++i) {
    if (i < blocks.size() && blocks[i]) {
      CacheBlock *b = blocks[i];
      internal_children_[i] = b;
      max_distances_[i] = b->get_max_distance();
      farthest_points_[i] = b->get_farthest_point();
    } else {
      internal_children_[i] = nullptr;
      max_distances_[i] = 0.0f;
      farthest_points_[i] = Point3D{};
    }
    if (max_distances_[i] > gmax) {
      gmax = max_distances_[i];
      gidx = i;
    }
  }
  global_max_distance_ = (gmax < 0.0f) ? 0.0f : gmax;
  global_max_index_ = gidx;
}

void CacheBlock::set_internal_child(size_t index, CacheBlock *child) {
  if (child) {
    child->parent_ = this;
    child->parent_slot_index_ = index;
    internal_children_[index] = child;
  } else {
    internal_children_[index] = nullptr;
  }
}

std::array<CacheBlock *, CacheBlock::kBlockSize> &
CacheBlock::get_internal_children() {
  return internal_children_;
}

std::array<MortonBlock *, CacheBlock::kBlockSize> &
CacheBlock::get_leaf_children() {
  return leaf_children_;
}

CacheBlock *CacheBlock::get_parent() const { return parent_; }

// =============================================================================
// MortonStructure 实现
// =============================================================================

MortonStructure::MortonStructure(const SpaceRange &range)
    : range_(range), occupied_table_(range.total_blocks()),
      cache_built_(false) {}

MortonStructure::~MortonStructure() {
  for (auto &level : cache_levels_) {
    for (CacheBlock *block : level)
      delete block;
  }
}

void MortonStructure::allocate(const Point3D &point) {
  uint32_t morton_id = point.compute_morton_code(range_);
  size_t hash_idx = morton_id % occupied_table_.size();
  occupied_table_[hash_idx].allocate(point.x, point.y, point.z,
                                     point.original_index);
}

void MortonStructure::build_multilevel_cache(float qx, float qy, float qz) {
  for (auto &level : cache_levels_) {
    for (CacheBlock *block : level)
      delete block;
  }
  cache_levels_.clear();
  leaf_blocks_per_l1_.clear();
  leaf_block_to_l1_index_.clear();
  cache_built_ = false;

  std::vector<MortonBlock *> all_blocks;
  all_blocks.reserve(1024);

  for (auto &entry : occupied_table_) {
    if (entry.is_occupied()) {
      for (auto &block_ptr : entry.page_table_ptr->blocks) {
        block_ptr->update_distance(qx, qy, qz);
        all_blocks.push_back(block_ptr.get());
      }
    }
  }

  if (all_blocks.empty())
    return;

  size_t total = all_blocks.size();
  size_t l1_count = (total + (CACHE_BLOCK_SIZE - 1)) / CACHE_BLOCK_SIZE;
  leaf_blocks_per_l1_.resize(l1_count);
  leaf_block_to_l1_index_.reserve(total);

  std::vector<CacheBlock *> current_level;
  current_level.reserve(l1_count);

  for (size_t i = 0; i < l1_count; ++i) {
    auto *cache_block = new CacheBlock();
    size_t start = i * CACHE_BLOCK_SIZE;
    size_t end = std::min(start + CACHE_BLOCK_SIZE, total);

    leaf_blocks_per_l1_[i].reserve(end - start);
    for (size_t idx = start; idx < end; ++idx) {
      MortonBlock *blk = all_blocks[idx];
      leaf_blocks_per_l1_[i].push_back(blk);
      leaf_block_to_l1_index_[blk] = i;
    }

    cache_block->rebuild_from_leaves(leaf_blocks_per_l1_[i]);
    current_level.push_back(cache_block);
  }
  cache_levels_.push_back(std::move(current_level));

  while (cache_levels_.back().size() > 1) {
    const auto &prev = cache_levels_.back();
    size_t prev_size = prev.size();
    size_t next_count = (prev_size + (CACHE_BLOCK_SIZE - 1)) / CACHE_BLOCK_SIZE;
    std::vector<CacheBlock *> next_level;
    next_level.reserve(next_count);

    for (size_t i = 0; i < next_count; ++i) {
      auto *cb = new CacheBlock();
      std::vector<CacheBlock *> children;
      children.reserve(CACHE_BLOCK_SIZE);
      size_t start = i * CACHE_BLOCK_SIZE;
      size_t end = std::min(start + CACHE_BLOCK_SIZE, prev_size);

      for (size_t idx = start; idx < end; ++idx) {
        children.push_back(prev[idx]);
        cb->set_internal_child(idx - start, prev[idx]);
      }
      cb->rebuild_from_internal(children);
      next_level.push_back(cb);
    }
    cache_levels_.push_back(std::move(next_level));
  }

  cache_built_ = true;
}

std::vector<MortonBlock *>
MortonStructure::update_morton_code_blocks(uint32_t id, float qx, float qy,
                                           float qz) {

  std::vector<MortonBlock *> affected;
  if (id >= occupied_table_.size())
    return affected;

  auto &entry = occupied_table_[id];
  if (!entry.is_occupied())
    return affected;

  affected.reserve(entry.page_table_ptr->blocks.size());
  for (auto &block_ptr : entry.page_table_ptr->blocks) {
    block_ptr->update_distance(qx, qy, qz);
    affected.push_back(block_ptr.get());
  }
  return affected;
}

void MortonStructure::update_cache_levels_from_blocks(
    const std::vector<MortonBlock *> &blocks) {

  if (!cache_built_ || cache_levels_.empty() || blocks.empty())
    return;

  size_t l1_size = cache_levels_[0].size();
  if (l1_size == 0)
    return;

  std::vector<char> touched(l1_size, 0);
  std::vector<CacheBlock *> affected;
  affected.reserve(CACHE_BLOCK_SIZE);

  for (MortonBlock *blk : blocks) {
    auto it = leaf_block_to_l1_index_.find(blk);
    if (it != leaf_block_to_l1_index_.end()) {
      size_t idx = it->second;
      if (!touched[idx]) {
        touched[idx] = 1;
        if (idx < cache_levels_[0].size()) {
          CacheBlock *cb = cache_levels_[0][idx];
          cb->rebuild_from_leaves(leaf_blocks_per_l1_[idx]);
          affected.push_back(cb);
        }
      }
    }
  }

  if (!affected.empty()) {
    propagate_updates_upward(affected);
  }
}

std::pair<float, Point3D> MortonStructure::get_global_max_from_cache() const {
  if (!cache_built_ || cache_levels_.empty()) {
    return {0.0f, Point3D{}};
  }
  const auto &top = cache_levels_.back();
  if (!top.empty()) {
    return top[0]->get_max();
  }
  return {0.0f, Point3D{}};
}

std::vector<uint32_t>
MortonStructure::get_morton_codes_in_range(float qx, float qy, float qz,
                                           float distance_sq) const {

  if (distance_sq < 0.0f)
    distance_sq = 0.0f;
  float radius = std::sqrt(distance_sq);

  int nx = static_cast<int>(std::ceil(radius / range_.block_size_x()));
  int ny = static_cast<int>(std::ceil(radius / range_.block_size_y()));
  int nz = static_cast<int>(std::ceil(radius / range_.block_size_z()));

  Point3D query(qx, qy, qz);
  auto [cx, cy, cz] = query.compute_block_indices(range_);

  std::vector<uint32_t> codes;
  codes.reserve((2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1));

  for (int dz = -nz; dz <= nz; ++dz) {
    for (int dy = -ny; dy <= ny; ++dy) {
      for (int dx = -nx; dx <= nx; ++dx) {
        int bx = static_cast<int>(cx) + dx;
        int by = static_cast<int>(cy) + dy;
        int bz = static_cast<int>(cz) + dz;

        if (bx < 0 || bx >= static_cast<int>(range_.x_blocks) || by < 0 ||
            by >= static_cast<int>(range_.y_blocks) || bz < 0 ||
            bz >= static_cast<int>(range_.z_blocks)) {
          continue;
        }

        uint32_t code = range_.x_lut_[bx] | range_.y_lut_[by] | range_.z_lut_[bz];
        codes.push_back(code % occupied_table_.size());
      }
    }
  }
  return codes;
}

// uint32_t MortonStructure::encode_block(uint32_t x, uint32_t y,
//                                        uint32_t z) const {
//   return range_.x_lut_[x] | range_.y_lut_[y] | range_.z_lut_[z];

// }

void MortonStructure::propagate_updates_upward(
    std::vector<CacheBlock *> &current) {
  while (!current.empty() && current[0]->get_parent() != nullptr) {
    std::unordered_set<CacheBlock *> parents;
    parents.reserve(current.size());

    for (CacheBlock *child : current) {
      if (CacheBlock *p = child->get_parent())
        parents.insert(p);
    }
    if (parents.empty())
      break;

    std::vector<CacheBlock *> next;
    next.reserve(parents.size());

    for (CacheBlock *p : parents) {
      std::vector<CacheBlock *> children;
      children.reserve(CACHE_BLOCK_SIZE);
      auto &arr = p->get_internal_children();
      for (size_t i = 0; i < CacheBlock::kBlockSize; ++i) {
        if (arr[i])
          children.push_back(arr[i]);
      }
      p->rebuild_from_internal(children);
      next.push_back(p);
    }
    current = std::move(next);
  }
}

// =============================================================================
// bbox计算函数
// =============================================================================

SpaceRange compute_bbox(const std::vector<Point3D> &points,
                        uint32_t blocks_per_dim) {
  return compute_bbox(points, blocks_per_dim, blocks_per_dim, blocks_per_dim);
}

SpaceRange compute_bbox(const std::vector<Point3D> &points, uint32_t x_blocks,
                        uint32_t y_blocks, uint32_t z_blocks) {
  if (points.empty()) {
    return make_range(0, 1, 0, 1, 0, 1, 1, 1, 1);
  }

  float min_x = points[0].x, max_x = points[0].x;
  float min_y = points[0].y, max_y = points[0].y;
  float min_z = points[0].z, max_z = points[0].z;

  for (const auto &p : points) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
    min_z = std::min(min_z, p.z);
    max_z = std::max(max_z, p.z);
  }

  float eps = BOUNDARY_EPS;
  return make_range(min_x - eps, max_x + eps, min_y - eps, max_y + eps,
                    min_z - eps, max_z + eps, x_blocks, y_blocks, z_blocks);
}

// =============================================================================
// 主入口函数实现
// =============================================================================

std::vector<size_t> yuezu_fps(const std::vector<Point3D> &point_cloud,
                              size_t sample_count, const SpaceRange &range) {
  if (point_cloud.empty() || sample_count == 0) {
    return {};
  }
  if (sample_count >= point_cloud.size()) {
    std::vector<size_t> result;
    result.reserve(point_cloud.size());
    for (const auto &p : point_cloud) {
      result.push_back(p.original_index);
    }
    return result;
  }

  MortonStructure structure(range);

  for (const auto &point : point_cloud) {
    structure.allocate(point);
  }

  Point3D first = point_cloud[0];
  structure.build_multilevel_cache(first.x, first.y, first.z);

  std::vector<Point3D> sampled_points;
  sampled_points.reserve(sample_count);
  sampled_points.push_back(first);

  for (size_t i = 1; i < sample_count; ++i) {
    auto [max_dist_sq, farthest] = structure.get_global_max_from_cache();
    if (max_dist_sq <= 0.0f)
      break;

    sampled_points.push_back(farthest);

    auto morton_ids = structure.get_morton_codes_in_range(
        farthest.x, farthest.y, farthest.z, max_dist_sq);

    std::vector<MortonBlock *> all_affected;
    for (uint32_t id : morton_ids) {
      auto blocks = structure.update_morton_code_blocks(id, farthest.x,
                                                        farthest.y, farthest.z);
      if (!blocks.empty()) {
        all_affected.insert(all_affected.end(), blocks.begin(), blocks.end());
      }
    }
    structure.update_cache_levels_from_blocks(all_affected);
  }

  std::vector<size_t> indices;
  indices.reserve(sampled_points.size());
  for (const auto &p : sampled_points) {
    indices.push_back(p.original_index);
  }
  return indices;
}