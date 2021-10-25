cmake_minimum_required(VERSION 3.18)

# Used for config file generation
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
else()
  set(NOT_SUBPROJECT OFF)
endif()

# Command line options
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_DOCS "Build documentation. Mutually exclusive with all other options" OFF)
option(BUILD_32_BIT "Build with 32-bit indexing support" OFF)
option(MULTI_GPU "Multi-GPU support" OFF)
option(EN_VISUALIZATION "Enable visualization support" OFF)
option(EN_CUTLASS OFF)
option(GPU_ARCH "List of GPU architectures to build for, separated by semicolon" OFF)

# Building documentation is mutually exclusive with everything else, and doesn't require CUDA
if (BUILD_DOCS)
    project(MATX_DOCS)
    add_subdirectory(docs)
    return()
endif()

# This needs to go after BUILD_DOCS check so it doesn't look for CUDA if we're just building docs
project(MATX
        LANGUAGES CUDA CXX
        DESCRIPTION "A modern and efficient header-only C++ library for numerical computing on GPU"
        VERSION 0.0.13
        HOMEPAGE_URL "https://github.com/NVIDIA/MatX")

# In an upcoming CMake it will have the capability to auto-detect GPU architectures. For now, rapids-cmake has a utility
# function to do it, so we grab that as a dependency. The user can optionally override GPU_ARCH to specify
# their own
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-21.12/RAPIDS.cmake
     ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

rapids_cmake_write_version_file(include/version_config.h)

# Figure out what GPU arch their on if it's not specified. If we can't figure it out and it's not specified, fall back to 70;80
if(NOT GPU_ARCH)
    include(rapids-cuda)
    set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
    rapids_cuda_init_architectures(MATX)
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "Tried to detect GPU architecture of current machine, but failed. Falling back to 70;80")
        set(CMAKE_CUDA_ARCHITECTURES "70;80")
    else()
        message(STATUS "GPU_ARCH not specified. Using GPU architectures of this machine for building (${CMAKE_CUDA_ARCHITECTURES})")    
    endif()
else()
    message(STATUS "Using GPU architectures ${GPU_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
endif()


# MatX requires C++17 to build. Enforce on all libraries pulled in as well
set (CMAKE_CXX_STANDARD 17)
set (CUDA_CXX_STANDARD 17)

# CPM is required for all package management
include(cmake/GetCPM.cmake)
# Helper for selecting build type
include(cmake/BuildType.cmake)

rapids_find_package(
  CUDAToolkit
  BUILD_EXPORT_SET matx-exports
  INSTALL_EXPORT_SET matx-exports)

rapids_cpm_init()

# Create our transitive target to pass build properties to external users and our own build environment
add_library(matx INTERFACE)
add_library(matx::matx ALIAS matx)
target_include_directories(matx INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                         "$<INSTALL_INTERFACE:include>")
target_include_directories(matx INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/kernels>"
"$<INSTALL_INTERFACE:include/kernels>")                                         
target_compile_features(matx INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)
target_compile_options(matx INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# 11.2 and above required for async allocation
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.2)
    message(FATAL_ERROR "MatX requires CUDA 11.2 or higher. Please update before using.")
endif()

# If we're on CUDA 11.4 or lower we need a newer version of libcudacxx
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.5)
    message(STATUS "Need libcuda++ 1.6.0 or higher (included in CTK 11.5+). Finding...")
    set(LIBCUDACXX_VERSION "1.6.0" CACHE STRING "Version of libcudacxx to use")
    include(cmake/FindLibcudacxx.cmake)
    target_include_directories(matx INTERFACE "$<BUILD_INTERFACE:${LIBCUDACXX_INCLUDE_DIR}>")    
else()
    message(STATUS "Recent libcuda++ found in CUDA toolkit. Skipping finding...")
endif()

# Set flags for compiling tests faster
set(MATX_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --threads 0)
if (NOT CMAKE_BUILD_TYPE OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(MATX_CUDA_FLAGS ${MATX_CUDA_FLAGS} -g -lineinfo)
endif()

# Set preferred compiler warning flags
set(WARN_FLAGS  -Wall 
                -Wextra 
                -Werror all-warnings
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
        -Wnull-dereference
        -Wuseless-cast)
endif()

# CUTLASS slows down compile times when used, so leave it as optional for now
if (EN_CUTLASS)
    include(cmake/GetCUTLASS.cmake)
    set (CUTLASS_INC ${cutlass_SOURCE_DIR}/include/ ${cutlass_SOURCE_DIR}/tools/util/include/)
    target_compile_definitions(matx INTERFACE ENABLE_CUTLASS=1)
else()
    set (CUTLASS_INC "")
    target_compile_definitions(matx INTERFACE ENABLE_CUTLASS=0)
endif()

if (MULTI_GPU)
    include(cmake/FindNvshmem.cmake)
    find_package(Nvshmem REQUIRED)
endif()

# Find python3 and pybind11 for generating unit tests and benchmarks
if (BUILD_BENCHMARKS OR BUILD_TESTS OR EN_VISUALIZATION OR BUILD_EXAMPLES)
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
    if (EN_VISUALIZATION) 
        check_python_libs("plotly" "pandas")
    endif()    
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



if (BUILD_32_BIT)
    add_definitions(-DINDEX_32_BIT)
    target_compile_definitions(matx INTERFACE INDEX_32_BIT)
else()
    add_definitions(-DINDEX_64_BIT)
    target_compile_definitions(matx INTERFACE INDEX_64_BIT)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (BUILD_BENCHMARKS)
    include(cmake/GetNVBench.cmake)
    add_subdirectory(bench)
endif()

if (BUILD_TESTS)
    include(cmake/GetGTest.cmake)
    add_subdirectory(test)
endif()

