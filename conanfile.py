from conan import ConanFile
from conan.tools.cmake import CMake

class MatxConan(ConanFile):
    name = "matx"
    version = "1.0.0"
    license = "BSD 3-Clause License"
    homepage = "https://github.com/NVIDIA/MatX/"
    topics = ("hpc", "gpu", "cuda", "gpgpu", "gpu-computing")
    
    package_type = "header-library"
    description = (
        "MatX is a modern C++ library for numerical computing on NVIDIA GPUs and CPUs. "
        "Near-native performance can be achieved while using a simple syntax common in higher-level languages such as Python or MATLAB."
    )

    settings = "os", "compiler", "build_type", "arch"
    exports_sources = "CMakeLists.txt", "include/*", "cmake/*", "public/*", "LICENSE"
    
    generators = "CMakeToolchain"

    def package(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_target_name", "matx::matx")
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = ["lib/cmake/matx"]
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
