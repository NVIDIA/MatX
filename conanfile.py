from conan import ConanFile
from conan.tools.cmake import CMake

class MatxConan(ConanFile):
    """
    MatX Conan recipe for GPU-accelerated numerical computing.
    
    NOTE: Network Dependency During Package Creation
    ==================================================
    During 'conan create .', the package() method runs CMake's configure phase, which 
    automatically fetches CCCL (CUDA C++ Core Libraries) from GitHub via CPM. This means
    internet access is required for packaging, even in offline scenarios.
    
    For offline/air-gapped deployments:
    1. On an internet-enabled system, export CPM_SOURCE_CACHE and run 'conan create .'
    2. Transfer the populated CPM cache to the offline system
    3. Set CPM_SOURCE_CACHE before using the offline package
    
    See the build documentation for detailed offline deployment instructions.
    """
    name = "matx"
    version = "1.0.0"
    license = "BSD 3-Clause License"
    homepage = "https://github.com/NVIDIA/MatX/"
    topics = ("hpc", "gpu", "cuda", "gpgpu", "gpu-computing")
    
    package_type = "header-library"
    no_copy_source = True
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
