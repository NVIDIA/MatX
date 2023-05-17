cmake_minimum_required(VERSION 3.18)

# Used for config file generation
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
else()
  set(NOT_SUBPROJECT OFF)
endif()

# Command line options
option(MATX_BUILD_EXAMPLES "Build examples" OFF)
option(MATX_BUILD_TESTS "Build unit tests" OFF)
option(MATX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(MATX_NVTX_FLAGS "Enable NVTX Macros" OFF)
option(MATX_BUILD_DOCS "Build documentation. Mutually exclusive with all other options" OFF)
option(MATX_BUILD_32_BIT "Build with 32-bit indexing support" OFF)
option(MATX_MULTI_GPU "Multi-GPU support" OFF)
option(MATX_EN_VISUALIZATION "Enable visualization support" OFF)
option(MATX_EN_CUTLASS OFF)
option(MATX_EN_CUTENSOR OFF)
option(MATX_EN_FILEIO OFF)
option(MATX_DISABLE_CUB_CACHE "Disable caching for CUB allocations" ON)

set(MATX_EN_PYBIND11 OFF CACHE BOOL "Enable pybind11 support")

set(cutensor_DIR "" CACHE PATH "Directory where cuTENSOR is installed.")
set(cutensornet_DIR "" CACHE PATH "Directory where cuTensorNet is installed.")

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Building documentation is mutually exclusive with everything else, and doesn't require CUDA
if (MATX_BUILD_DOCS)
    project(MATX_DOCS)
    add_subdirectory(docs_input)
    return()
endif()

# CMake 3.24 can auto-detect GPUs, but it's not standard on any distrobution. For now, rapids-cmake has a utility
# function to do it, so we grab that as a dependency. The user can optionally override GPU_ARCH to specify
# their own
add_subdirectory(cmake/rapids-cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(rapids-cuda)
    set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
    message(STATUS "Auto-detecting GPU architectures since CMAKE_CUDA_ARCHITECTURES not defined")
    rapids_cuda_init_architectures(MATX)
endif()

# This needs to go after MATX_BUILD_DOCS check so it doesn't look for CUDA if we're just building docs
project(MATX
        LANGUAGES CUDA CXX
        DESCRIPTION "A modern and efficient header-only C++ library for numerical computing on GPU"
        VERSION 0.2.3
        HOMEPAGE_URL "https://github.com/NVIDIA/MatX")

if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80") 
endif()        
message(STATUS "Using GPU architectures ${CMAKE_CUDA_ARCHITECTURES}")        

rapids_cmake_write_version_file(include/version_config.h)

# MatX requires C++17 to build. Enforce on all libraries pulled in as well
set(CMAKE_CXX_STANDARD 17)
set(CUDA_CXX_STANDARD 17)

# CPM is required for all package management
include(public/cpm-cmake/cmake/CPM.cmake)
# Helper for selecting build type
include(cmake/BuildType.cmake)

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET matx-exports
  INSTALL_EXPORT_SET matx-exports)

rapids_cpm_init()

# Create our transitive target to pass build properties to external users and our own build environment
add_library(matx INTERFACE)
add_library(matx::matx ALIAS matx)
target_include_directories(matx INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                         "$<INSTALL_INTERFACE:include>")
target_include_directories(matx INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/kernels>"
"$<INSTALL_INTERFACE:include/matx/kernels>")                                         
target_compile_features(matx INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)
target_compile_options(matx INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# 11.2 and above required for async allocation
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.4)
    message(FATAL_ERROR "MatX requires CUDA 11.4 or higher. Please update before using.")
endif()

# We typically need newer versions libcudacxx than availabled in the toolkit.  pull down specific version here
message(STATUS "Need libcuda++. Finding...")
set(LIBCUDACXX_VERSION "2.1.0" CACHE STRING "Version of libcudacxx to use")
include(cmake/FindLibcudacxx.cmake)
target_include_directories(matx INTERFACE "$<BUILD_INTERFACE:${LIBCUDACXX_INCLUDE_DIR}>")    

# Set flags for compiling tests faster
set(MATX_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --threads 0 -ftemplate-backtrace-limit=0)
if (NOT CMAKE_BUILD_TYPE OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(MATX_CUDA_FLAGS ${MATX_CUDA_FLAGS} -g -lineinfo)
endif()

# Set preferred compiler warning flags
set(WARN_FLAGS  -Wall 
                -Wextra 
                -Wcast-align
                -Wunused
                -Wconversion
                -Wno-unknown-pragmas 
                -Wnon-virtual-dtor 
                -Wshadow)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(WARN_FLAGS ${WARN_FLAGS} 
        -Wmisleading-indentation
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wnull-dereference)
endif()

set(WARN_FLAGS ${WARN_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:-Werror all-warnings>)
set(WARN_FLAGS ${WARN_FLAGS} $<$<COMPILE_LANGUAGE:CXX>:-Werror>)

# CUTLASS slows down compile times when used, so leave it as optional for now
if (MATX_EN_CUTLASS)
    include(cmake/GetCUTLASS.cmake)
    set (CUTLASS_INC ${cutlass_SOURCE_DIR}/include/ ${cutlass_SOURCE_DIR}/tools/util/include/)
    target_compile_definitions(matx INTERFACE MATX_ENABLE_CUTLASS=1)
else()
    set (CUTLASS_INC "")
    target_compile_definitions(matx INTERFACE MATX_ENABLE_CUTLASS=0)
endif()

message(status ${MATX_DISABLE_CUB_CACHE})
if (MATX_DISABLE_CUB_CACHE)
    target_compile_definitions(matx INTERFACE MATX_DISABLE_CUB_CACHE=1)
endif()

# Get the tensor libraries if we need them
if (MATX_EN_CUTENSOR)
    include(cmake/FindcuTENSOR.cmake)
    include(cmake/FindcuTensorNet.cmake)
    target_compile_definitions(matx INTERFACE MATX_ENABLE_CUTENSOR=1)
    
    target_link_libraries(matx INTERFACE cuTENSOR::cuTENSOR)
    target_link_libraries(matx INTERFACE cuTensorNet::cuTensorNet)

    # CUDA toolkit and most accompanying libraries like cuTENSOR use the old rpath instead of RUNPATH.
    # We switch to that format here for compatibility
    target_link_libraries(matx INTERFACE "-Wl,--disable-new-dtags")
else()
    target_compile_definitions(matx INTERFACE MATX_ENABLE_CUTENSOR=0)
endif()

if (MATX_MULTI_GPU)
    include(cmake/FindNvshmem.cmake)
    find_package(Nvshmem REQUIRED)
endif()

# NVHPC has a bug where it doesn't create the CUDA includes as system includes. This causes errors with
# the warnings we enable. Forcefully add it here.
target_compile_options(matx INTERFACE -isystem=${CUDAToolkit_INCLUDE_DIRS})

# Find python3 and pybind11 for generating unit tests and benchmarks
if (MATX_EN_FILEIO OR MATX_EN_VISUALIZATION OR MATX_EN_PYBIND11 OR MATX_BUILD_EXAMPLES OR MATX_BUILD_TESTS OR MATX_BUILD_BENCHMARKS)
    message(STATUS "Enabling pybind11 support")
    set(MATX_EN_PYBIND11 ON)
    target_compile_definitions(matx INTERFACE MATX_ENABLE_PYBIND11=1)
    target_compile_definitions(matx INTERFACE MATX_ENABLE_FILEIO=1)
    target_compile_options(matx INTERFACE -DMATX_ROOT="${PROJECT_SOURCE_DIR}")

    include(cmake/GetPyBind11.cmake)
    find_package(Python3  REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED)

    # Check for python libs
    include(cmake/CheckPythonLibs.cmake)
    check_python_libs("numpy")

    # Required by pybind
    # https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-
    # visibility-than-the-type-of-its-field-someclass-member-wattributes
    target_compile_options(matx INTERFACE -fvisibility=hidden)
    target_link_libraries(matx INTERFACE pybind11::embed)

    # Visualization requires Python libraries
    if (MATX_EN_VISUALIZATION) 
        target_compile_definitions(matx INTERFACE MATX_ENABLE_VIZ=1)
        check_python_libs("plotly.express")
    else()
        target_compile_definitions(matx INTERFACE MATX_ENABLE_VIZ=0)
    endif()    
else()
    message(WARNING "pybind11 support disabled. Visualizations and file IO will be disabled")
    target_compile_definitions(matx INTERFACE MATX_ENABLE_PYBIND11=0)
    target_compile_definitions(matx INTERFACE MATX_ENABLE_FILEIO=0)
endif()

# Add in all CUDA linker dependencies
target_link_libraries(matx INTERFACE    CUDA::cudart
                                        CUDA::nvToolsExt 
                                        CUDA::cublas 
                                        CUDA::cublasLt 
                                        CUDA::cufft 
                                        CUDA::cusolver
                                        CUDA::cuda_driver)

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


if (MATX_NVTX_FLAGS)
    add_definitions(-DMATX_NVTX_FLAGS)
    target_compile_definitions(matx INTERFACE MATX_NVTX_FLAGS)
endif()
if (MATX_BUILD_32_BIT)
    add_definitions(-DINDEX_32_BIT)
    target_compile_definitions(matx INTERFACE INDEX_32_BIT)
else()
    add_definitions(-DINDEX_64_BIT)
    target_compile_definitions(matx INTERFACE INDEX_64_BIT)
endif()

if (MATX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (MATX_BUILD_BENCHMARKS)
    include(cmake/GetNVBench.cmake)
    add_subdirectory(bench)
endif()

if (MATX_BUILD_TESTS)
    include(cmake/GetGTest.cmake)
    add_subdirectory(test)
endif()

