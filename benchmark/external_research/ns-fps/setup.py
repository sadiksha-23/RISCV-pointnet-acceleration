from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "yuezu_fps.yuezu_fps_module",  # 模块名
        sources=[
            "yuezu_fps/yuezu_fps_pybind.cpp",  # pybind11 绑定代码
            "src/yuezu_fps.cpp",             # 核心实现
        ],
        include_dirs=["src"],               # 头文件路径
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="yuezu_fps",
    version="0.1.0",
    author="Your Name",
    description="Fast Farthest Point Sampling with Morton-based Neighbor Search",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["numpy"],
)