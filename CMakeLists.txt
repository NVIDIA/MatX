cmake_minimum_required(VERSION 3.23.1)

set(MATX_VERSION 0.9.2)

# Used for config file generation
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
else()
  set(NOT_SUBPROJECT OFF)
endif()

# CMake 3.24 can auto-detect GPUs, but it's not standard on any distrobution. For now, rapids-cmake has a utility
# function to do it, so we grab that as a dependency. The user can optionally override GPU_ARCH to specify
# their own. We check if rapids-cmake exists for projects that already include it so we don't have conflicting
# directories
if(NOT DEFINED rapids-cmake-dir)
  include(FetchContent)
  # Tell FetchContent to just use the local copy of rapids-cmake:
  FetchContent_Declare(rapids-cmake SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rapids-cmake")

  # Tell FetchContent to download remote copy of rapids-cmake:
  #FetchContent_Declare(rapids-cmake URL https://github.com/rapidsai/rapids-cmake/archive/refs/heads/branch-24.12.zip)
  FetchContent_MakeAvailable(rapids-cmake)
else()
  # The include() commands below search the module path for the corresponding .cmake files
  list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
endif()

include(${rapids-cmake-dir}/cpm/cccl.cmake)
include(${rapids-cmake-dir}/cpm/nvbench.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init(OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(rapids-cuda)
    set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
    message(STATUS "Auto-detecting GPU architectures since CMAKE_CUDA_ARCHITECTURES not defined")
    rapids_cuda_init_architectures(MATX)
endif()

# This needs to go after rapids initialization otherwise we get a rapids_export_parse_version error
project(MATX
        LANGUAGES CUDA CXX
        DESCRIPTION "A modern and efficient header-only C++ library for numerical computing on GPU"
        VERSION ${MATX_VERSION}
        HOMEPAGE_URL "https://github.com/NVIDIA/MatX")

if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80")
endif()
message(STATUS "Using GPU architectures ${CMAKE_CUDA_ARCHITECTURES}")

rapids_cmake_write_version_file(include/version_config.h)

# Command line options
option(MATX_BUILD_EXAMPLES "Build examples" OFF)
option(MATX_BUILD_TESTS "Build unit tests" OFF)
option(MATX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(MATX_NVTX_FLAGS "Enable NVTX Macros" OFF)
option(MATX_BUILD_DOCS "Build documentation" OFF)
option(MATX_BUILD_32_BIT "Build with 32-bit indexing support" OFF)
option(MATX_MULTI_GPU "Multi-GPU support" OFF)
option(MATX_EN_VISUALIZATION "Enable visualization support" OFF)
#option(MATX_EN_CUTLASS OFF)
option(MATX_EN_CUTENSOR OFF)
option(MATX_EN_CUDSS OFF)
option(MATX_EN_FILEIO OFF)
option(MATX_EN_X86_FFTW OFF "Enable x86 FFTW support")
option(MATX_EN_NVPL OFF, "Enable NVIDIA Performance Libraries for optimized ARM CPU support")
option(MATX_EN_BLIS OFF "Enable BLIS support")
option(MATX_EN_OPENBLAS OFF "Enable OpenBLAS (BLAS + LAPACK) support")
option(MATX_DISABLE_CUB_CACHE "Disable caching for CUB allocations" ON)
option(MATX_EN_COVERAGE OFF "Enable code coverage reporting")
option(MATX_EN_COMPLEX_OP_NAN_CHECKS "Enable full NaN/Inf handling for complex multiplication and division" OFF)
option(MATX_EN_CUDA_LINEINFO "Enable line information for CUDA kernels via -lineinfo nvcc flag" OFF)

set(MATX_EN_PYBIND11 OFF CACHE BOOL "Enable pybind11 support")

set(cudss_DIR "" CACHE PATH "Directory where cuDSS is installed.")
set(cutensor_DIR "" CACHE PATH "Directory where cuTENSOR is installed.")
set(cutensornet_DIR "" CACHE PATH "Directory where cuTensorNet is installed.")
set(eigen_DIR "" CACHE PATH "Directory where Eigen is installed")
set(blas_DIR "" CACHE PATH "Directory where a BLAS library (NVPL/OpenBLAS/BLIS) is installed (install prefix)")

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MATX_BUILD_DOCS)
    project(MATX_DOCS VERSION ${MATX_VERSION})
    add_subdirectory(docs_input)
endif()

# MatX requires C++17 to build. Enforce on all libraries pulled in as well
set(CMAKE_CXX_STANDARD 17)
set(CUDA_CXX_STANDARD 17)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (NOT GCC_VERSION VERSION_GREATER 8)
        message(FATAL_ERROR "${PROJECT_NAME} requires g++ 9 or higher if using g++ host compiler")
    endif()
endif()

# CPM is required for all package management
include(public/cpm-cmake/cmake/CPM.cmake)
# Helper for selecting build type
include(cmake/BuildType.cmake)

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET matx-exports
  INSTALL_EXPORT_SET matx-exports)

# Create our transitive target to pass build properties to external users and our own build environment
add_library(matx INTERFACE)
add_library(matx::matx ALIAS matx)
target_include_directories(matx INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                         "$<INSTALL_INTERFACE:include>")
target_include_directories(matx INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/kernels>"
"$<INSTALL_INTERFACE:include/matx/kernels>")
target_compile_features(matx INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)

# 11.2 and above required for async allocation
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.5)
    message(FATAL_ERROR "MatX requires CUDA 11.5 or higher. Please update before using.")
endif()

message(STATUS "Finding CCCL...")
rapids_cpm_cccl(
    BUILD_EXPORT_SET matx-exports
    INSTALL_EXPORT_SET matx-exports
)

target_link_libraries(matx INTERFACE CCCL::CCCL)

# Set flags for compiling tests faster (only for nvcc)
if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    set(MATX_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --threads 0 -ftemplate-backtrace-limit=0)
endif()

# Hack because CMake doesn't have short circult evaluation
if (NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR MATX_EN_CUDA_LINEINFO)
    set(MATX_CUDA_FLAGS ${MATX_CUDA_FLAGS} -lineinfo)
endif()

# Set preferred compiler warning flags. nvc++ doesn't support most warnings
string(FIND "${CMAKE_CUDA_HOST_COMPILER}" "nvc++" IS_NVCPP)
if (NOT ${IS_NVCPP} GREATER -1)
    set(WARN_FLAGS
        -Wall
        -Wextra
        -Wcast-align
        -Wunused
        -Wshadow
        -Wno-unknown-pragmas
        -Wnon-virtual-dtor)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(WARN_FLAGS ${WARN_FLAGS}
            -Wconversion
            -Wmisleading-indentation
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wnull-dereference)
    endif()
endif()

if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
message((STATUS "Using Clang compiler"))
    # Workaround for clang bug: https://github.com/llvm/llvm-project/issues/58491
    set(WARN_FLAGS ${WARN_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-command-line-argument>)
else()
    set(WARN_FLAGS ${WARN_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-Werror all-warnings>)
endif()
set(WARN_FLAGS ${WARN_FLAGS} $<$<COMPILE_LANGUAGE:CXX>:-Werror>)

# CUTLASS slows down compile times when used, so leave it as optional for now
# if (MATX_EN_CUTLASS)
#     include(cmake/GetCUTLASS.cmake)
#     set (CUTLASS_INC ${cutlass_SOURCE_DIR}/include/ ${cutlass_SOURCE_DIR}/tools/util/include/)
#     target_compile_definitions(matx INTERFACE MATX_ENABLE_CUTLASS=1)
# else()
#     set (CUTLASS_INC "")
#     target_compile_definitions(matx INTERFACE MATX_ENABLE_CUTLASS=0)
# endif()

# CUTLASS support is not maintained. Remove the option to avoid confusion

if (MATX_NVTX_FLAGS)
    add_definitions(-DMATX_NVTX_FLAGS)
    target_compile_definitions(matx INTERFACE MATX_NVTX_FLAGS)
endif()
if (MATX_BUILD_32_BIT)
    set(MATX_NVPL_INT_TYPE "lp64")
    target_compile_definitions(matx INTERFACE MATX_INDEX_32_BIT)
else()
    set(MATX_NVPL_INT_TYPE "ilp64")
endif()

# Host support
if (MATX_EN_NVPL OR MATX_EN_X86_FFTW OR MATX_EN_BLIS OR MATX_EN_OPENBLAS)
    message(STATUS "Enabling OpenMP support")
    find_package(OpenMP REQUIRED)
    target_link_libraries(matx INTERFACE OpenMP::OpenMP_CXX)
    target_compile_options(matx INTERFACE ${OpenMP_CXX_FLAGS})
    target_compile_definitions(matx INTERFACE MATX_EN_OMP)

    set(BLAS_FLAGS MATX_EN_NVPL MATX_EN_BLIS MATX_EN_OPENBLAS)
    set(ENABLED_BLAS_COUNT 0)
    foreach(BLAS_FLAG IN LISTS BLAS_FLAGS)
        if(${BLAS_FLAG})
            math(EXPR ENABLED_BLAS_COUNT "${ENABLED_BLAS_COUNT} + 1")
        endif()
    endforeach()
    if(ENABLED_BLAS_COUNT GREATER 1)
        message(AUTHOR_WARNING "Multiple Host BLAS libraries (${ENABLED_BLAS_COUNT}) are enabled. Only 1 will be used.")
    endif()

    if (MATX_EN_NVPL)
        message(STATUS "Enabling NVPL library support for ARM CPUs with ${MATX_NVPL_INT_TYPE} interface")
        find_package(nvpl REQUIRED COMPONENTS fft blas lapack HINTS ${blas_DIR})
        if (NOT MATX_BUILD_32_BIT)
            target_compile_definitions(matx INTERFACE NVPL_ILP64)
        endif()
        target_compile_definitions(matx INTERFACE NVPL_LAPACK_COMPLEX_CUSTOM)
        target_link_libraries(matx INTERFACE nvpl::fftw nvpl::blas_${MATX_NVPL_INT_TYPE}_omp nvpl::lapack_${MATX_NVPL_INT_TYPE}_omp)
        target_compile_definitions(matx INTERFACE MATX_EN_NVPL)
    else()
        # FFTW
        if (MATX_EN_X86_FFTW)
            message(STATUS "Enabling x86 FFTW")
            find_library(FFTW_LIB fftw3 REQUIRED)
            find_library(FFTWF_LIB fftw3f REQUIRED)
            find_library(FFTW_OMP_LIB fftw3_omp REQUIRED)
            find_library(FFTWF_OMP_LIB fftw3f_omp REQUIRED)
            target_link_libraries(matx INTERFACE ${FFTW_LIB} ${FFTWF_LIB} ${FFTW_OMP_LIB} ${FFTWF_OMP_LIB})
            target_compile_definitions(matx INTERFACE MATX_EN_X86_FFTW)
        endif()

        # BLAS
        if (MATX_EN_BLIS)
            message(STATUS "Enabling BLIS")
            include(cmake/FindBLIS.cmake)
            target_link_libraries(matx INTERFACE BLIS::BLIS)
            target_compile_definitions(matx INTERFACE MATX_EN_BLIS)
        elseif(MATX_EN_OPENBLAS)
            message(STATUS "Enabling OpenBLAS")
            include(cmake/FindOpenBLAS.cmake)
            target_link_libraries(matx INTERFACE OpenBLAS::OpenBLAS)
            target_compile_definitions(matx INTERFACE MATX_EN_OPENBLAS)
        endif()
    endif()
endif()

if (MATX_DISABLE_CUB_CACHE)
    target_compile_definitions(matx INTERFACE MATX_DISABLE_CUB_CACHE)
endif()

if (MATX_EN_COMPLEX_OP_NAN_CHECKS)
    target_compile_definitions(matx INTERFACE MATX_EN_COMPLEX_OP_NAN_CHECKS)
endif()

if (MATX_EN_COVERAGE)
    target_compile_options(matx INTERFACE -fprofile-arcs -ftest-coverage)
    target_link_options(matx INTERFACE -lgcov --coverage)
endif()

# Add nvtiff support if installed
find_library(NVTIFF_LIBRARY nvtiff)
if (NOT NVTIFF_LIBRARY)
    message(STATUS "Cannot find nvtiff library.  Disabling MatX nvtiff features.")
else()
    message(STATUS "Found nvtiff library at ${NVTIFF_LIBRARY}.  Enabling MatX nvtiff features.")
    target_compile_definitions(matx INTERFACE MATX_ENABLE_NVTIFF)
endif()


# Get the tensor libraries if we need them
if (MATX_EN_CUTENSOR)
    set(CUTENSORNET_VERSION 24.03.0.4)
    set(CUTENSOR_VERSION 2.0.1.2)

    include(cmake/FindcuTENSOR.cmake)
    include(cmake/FindcuTensorNet.cmake)
    target_compile_definitions(matx INTERFACE MATX_EN_CUTENSOR)

    target_link_libraries(matx INTERFACE cuTENSOR::cuTENSOR)
    target_link_libraries(matx INTERFACE cuTensorNet::cuTensorNet)

    # CUDA toolkit and most accompanying libraries like cuTENSOR use the old rpath instead of RUNPATH.
    # We switch to that format here for compatibility
    target_link_libraries(matx INTERFACE "-Wl,--disable-new-dtags")
endif()

if (MATX_EN_CUDSS)
    set(cuDSS_VERSION 0.4.0.2)
    include(cmake/FindcuDSS.cmake)
    target_compile_definitions(matx INTERFACE MATX_EN_CUDSS)
    target_link_libraries(matx INTERFACE cuDSS::cuDSS)
endif()

if (MATX_MULTI_GPU)
    include(cmake/FindNvshmem.cmake)
    find_package(Nvshmem REQUIRED)
endif()

# Find python3 and pybind11 for generating unit tests and benchmarks
if (MATX_EN_FILEIO OR MATX_EN_VISUALIZATION OR MATX_EN_PYBIND11 OR MATX_BUILD_EXAMPLES OR MATX_BUILD_TESTS OR MATX_BUILD_BENCHMARKS)
    message(STATUS "Enabling pybind11 support")
    set(MATX_EN_PYBIND11 ON)
    target_compile_definitions(matx INTERFACE MATX_ENABLE_PYBIND11)
    target_compile_definitions(matx INTERFACE MATX_ENABLE_FILEIO)

    include(cmake/GetPyBind11.cmake)
    find_package(Python3  REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED)

    # Check for python libs
    include(cmake/CheckPythonLibs.cmake)
    check_python_libs("numpy")
    check_optional_python_libs("cupy")

    # Required by pybind
    # https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-
    # visibility-than-the-type-of-its-field-someclass-member-wattributes
    target_compile_options(matx INTERFACE -fvisibility=hidden)
    target_link_libraries(matx INTERFACE pybind11::embed)

    # Visualization requires Python libraries
    if (MATX_EN_VISUALIZATION)
        target_compile_definitions(matx INTERFACE MATX_ENABLE_VIZ)
        check_python_libs("plotly.express")
    endif()
endif()

# Add in all CUDA linker dependencies
target_link_libraries(matx INTERFACE    CUDA::cudart
                                        CUDA::cublas
                                        CUDA::cublasLt
                                        CUDA::cufft
                                        CUDA::cusolver
                                        CUDA::cuda_driver
                                        CUDA::curand)

if (CMAKE_VERSION VERSION_LESS 3.25.0)
    target_link_libraries(matx INTERFACE CUDA::nvToolsExt)
else()
    target_link_libraries(matx INTERFACE CUDA::nvtx3)
endif()

# Build config files if the user isn't adding this as a subdirectory. At this point our transitive target
# should have all build properties needed based on the options passed in
if (NOT_SUBPROJECT)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS matx EXPORT matx-exports)
    install(DIRECTORY include/ DESTINATION include)
    install(FILES ${CMAKE_BINARY_DIR}/include/version_config.h DESTINATION include)

    set(doc_string
    [=[
    Provide targets for MatX.

    [MatX](https://github.com/NVIDIA/MatX) provides a Python-like syntax for near-native speed
    numerical computing on NVIDIA GPUs.
    ]=])

    rapids_export(
        INSTALL matx
        EXPORT_SET matx-exports
        GLOBAL_TARGETS matx
        NAMESPACE matx::
        DOCUMENTATION doc_string)

      # build export targets
      rapids_export(
        BUILD matx
        EXPORT_SET matx-exports
        GLOBAL_TARGETS matx
        NAMESPACE matx::
        DOCUMENTATION doc_string)
endif()


if (MATX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (MATX_BUILD_BENCHMARKS)
    rapids_cpm_nvbench()
    add_subdirectory(bench)
endif()

if (MATX_BUILD_TESTS)
    include(cmake/GetGTest.cmake)
    add_subdirectory(test)
endif()
