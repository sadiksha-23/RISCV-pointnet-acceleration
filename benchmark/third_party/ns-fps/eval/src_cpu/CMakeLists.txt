cmake_minimum_required(VERSION 3.10)
project(yuezu_fps_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==================== 原始CPU版本 ====================
add_library(yuezu_fps_cpu STATIC
    yuezu_fps.cpp
    yuezu_fps.h
)

target_include_directories(yuezu_fps_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(yuezu_fps_cpu PRIVATE
    -O3
    -march=native
    -ffast-math
)

# ==================== 测试程序 ====================

# 原始版本简单测试
add_executable(test_cpu_fps
    test_cpu.cpp
)

target_link_libraries(test_cpu_fps PRIVATE
    yuezu_fps_cpu
)



# 原始版本对比测试
add_executable(compare_cpu_test
    compare_cpu_test.cpp
)

target_link_libraries(compare_cpu_test PRIVATE
    yuezu_fps_cpu
)


# 对比测试程序
add_executable(compare_cpu_test_kitti
    compare_cpu_test_kitti.cpp
)

target_link_libraries(compare_cpu_test_kitti PRIVATE
    yuezu_fps_cpu
)

