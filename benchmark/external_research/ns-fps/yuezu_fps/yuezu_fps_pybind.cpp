#include "yuezu_fps.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// 兼容旧版本 pybind11 的 dict get 方法
template <typename T> T dict_get(py::dict d, const char *key, T default_val) {
  if (d.contains(key)) {
    return d[key].cast<T>();
  }
  return default_val;
}

// NumPy 数组转换
std::vector<Point3D> numpy_to_points(py::array_t<float> points_array) {
  py::buffer_info buf = points_array.request();

  if (buf.ndim != 2 || buf.shape[1] != 3) {
    throw std::runtime_error("Expected 2D array with shape (N, 3)");
  }

  size_t N = buf.shape[0];
  float *ptr = static_cast<float *>(buf.ptr);

  std::vector<Point3D> points;
  points.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    points.emplace_back(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2], i);
  }
  return points;
}

// 从 Python 对象解析 SpaceRange（支持 dict 或 SpaceRange 对象）
SpaceRange parse_range(py::object range_obj) {
  SpaceRange range;

  // 尝试转换为 SpaceRange 对象
  try {
    range = range_obj.cast<SpaceRange>();
    return range;
  } catch (const py::cast_error &) {
    // 不是 SpaceRange，尝试 dict
  }

  // 尝试转换为 dict
  try {
    py::dict d = range_obj.cast<py::dict>();
    range.min_x = d["min_x"].cast<float>();
    range.max_x = d["max_x"].cast<float>();
    range.min_y = d["min_y"].cast<float>();
    range.max_y = d["max_y"].cast<float>();
    range.min_z = d["min_z"].cast<float>();
    range.max_z = d["max_z"].cast<float>();
    range.x_blocks = dict_get(d, "x_blocks", DEFAULT_X_BLOCKS);
    range.y_blocks = dict_get(d, "y_blocks", DEFAULT_Y_BLOCKS);
    range.z_blocks = dict_get(d, "z_blocks", DEFAULT_Z_BLOCKS);
    return range;
  } catch (const py::cast_error &e) {
    throw std::runtime_error("range must be a SpaceRange object or a dict with "
                             "keys: min_x, max_x, min_y, max_y, min_z, max_z");
  }
}

// Python 接口
py::array_t<size_t> fps_numpy(py::array_t<float> points_array,
                              size_t sample_count,
                              py::object range_obj = py::none()) {

  auto points = numpy_to_points(points_array);

  SpaceRange range;
  if (range_obj.is_none()) {
    range = compute_bbox(points, DEFAULT_X_BLOCKS, DEFAULT_Y_BLOCKS,
                         DEFAULT_Z_BLOCKS);
  } else {
    range = parse_range(range_obj);
  }

  auto result = yuezu_fps(points, sample_count, range);

  py::array_t<size_t> arr(result.size());
  std::copy(result.begin(), result.end(), arr.mutable_data());
  return arr;
}

py::array_t<size_t> fps_with_bbox(py::array_t<float> points_array,
                                  size_t sample_count,
                                  uint32_t x_blocks = DEFAULT_X_BLOCKS,
                                  uint32_t y_blocks = DEFAULT_Y_BLOCKS,
                                  uint32_t z_blocks = DEFAULT_Z_BLOCKS) {

  auto points = numpy_to_points(points_array);
  auto range = compute_bbox(points, x_blocks, y_blocks, z_blocks);
  auto result = yuezu_fps(points, sample_count, range);

  py::array_t<size_t> arr(result.size());
  std::copy(result.begin(), result.end(), arr.mutable_data());
  return arr;
}

PYBIND11_MODULE(yuezu_fps_module, m) {
  m.doc() = "Fast Poisson Disk Sampling with Morton coding";

  // SpaceRange
  py::class_<SpaceRange>(m, "SpaceRange")
      .def(py::init<>()) // 添加默认构造函数
      .def_readwrite("min_x", &SpaceRange::min_x)
      .def_readwrite("max_x", &SpaceRange::max_x)
      .def_readwrite("min_y", &SpaceRange::min_y)
      .def_readwrite("max_y", &SpaceRange::max_y)
      .def_readwrite("min_z", &SpaceRange::min_z)
      .def_readwrite("max_z", &SpaceRange::max_z)
      .def_readwrite("x_blocks", &SpaceRange::x_blocks)
      .def_readwrite("y_blocks", &SpaceRange::y_blocks)
      .def_readwrite("z_blocks", &SpaceRange::z_blocks)
      .def("block_size_x", &SpaceRange::block_size_x)
      .def("block_size_y", &SpaceRange::block_size_y)
      .def("block_size_z", &SpaceRange::block_size_z)
      .def("x_bits", &SpaceRange::x_bits)
      .def("y_bits", &SpaceRange::y_bits)
      .def("z_bits", &SpaceRange::z_bits)
      .def("total_bits", &SpaceRange::total_bits)
      .def("total_blocks", &SpaceRange::total_blocks);

  // 构造函数
  m.def("make_range", &make_range, py::arg("min_x"), py::arg("max_x"),
        py::arg("min_y"), py::arg("max_y"), py::arg("min_z"), py::arg("max_z"),
        py::arg("x_blocks"), py::arg("y_blocks"), py::arg("z_blocks"));

  m.def("make_uniform_range", &make_uniform_range, py::arg("min_x"),
        py::arg("max_x"), py::arg("min_y"), py::arg("max_y"), py::arg("min_z"),
        py::arg("max_z"), py::arg("blocks_per_dim"));

  // FPS 函数
  m.def("fps", &fps_numpy, py::arg("points"), py::arg("sample_count"),
        py::arg("range") = py::none());

  m.def("fps_with_bbox", &fps_with_bbox, py::arg("points"),
        py::arg("sample_count"), py::arg("x_blocks") = DEFAULT_X_BLOCKS,
        py::arg("y_blocks") = DEFAULT_Y_BLOCKS,
        py::arg("z_blocks") = DEFAULT_Z_BLOCKS);

  // 常量
  m.attr("DEFAULT_X_BLOCKS") = DEFAULT_X_BLOCKS;
  m.attr("DEFAULT_Y_BLOCKS") = DEFAULT_Y_BLOCKS;
  m.attr("DEFAULT_Z_BLOCKS") = DEFAULT_Z_BLOCKS;
  m.attr("MORTON_BLOCK_SIZE") = MORTON_BLOCK_SIZE;
  m.attr("CACHE_BLOCK_SIZE") = CACHE_BLOCK_SIZE;
}