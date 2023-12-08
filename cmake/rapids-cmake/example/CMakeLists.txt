#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.10/RAPIDS.cmake
       ${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/EXAMPLE_RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(integration)

project(integration VERSION 21.12 LANGUAGES C CXX CUDA)
message(STATUS "Example: CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

option(CUDA_STATIC_RUNTIME "Use CUDA static runtime" OFF)
# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Set the CUDA runtime CUDA targets will use
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})


# conda environment
rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )

# Find the CUDAToolkit
rapids_find_package(CUDAToolkit REQUIRED
                    BUILD_EXPORT_SET example-targets
                    INSTALL_EXPORT_SET example-targets
                    )

# Generate a `FindZLIBS` module that will searches for the zlib.h
# header and `libz` or `libzlib` library.
# In practise this function should be used to search for packages
# that don't provide a -config.cmake file or CMake doesn't
# already package a find module for ( includes ZLIB ).
rapids_find_generate_module(ZLIBS
    HEADER_NAMES        zlib.h
    LIBRARY_NAMES       z zlib
    BUILD_EXPORT_SET    example-targets
    INSTALL_EXPORT_SET  example-targets
    )

# Since `rapids_find_generate_module` doesn't interact with the export-set
# like `rapids_find` does, we manually establish this connection.
# The vast majority of calls should use `rapids_find_package`, and
# rapids_export_package should only be used in edge cases. The most common
# edge case being when a package doesn't provide a built directory -config.cmake
# module
find_package(ZLIBS REQUIRED)
rapids_export_package(INSTALL ZLIBS example-targets GLOBAL_TARGETS ZLIBS::ZLIBS)
rapids_export_package(BUILD ZLIBS example-targets)

# find Threads (needed by cudftestutil)
rapids_find_package(Threads REQUIRED
                    BUILD_EXPORT_SET example-targets
                    INSTALL_EXPORT_SET example-targets
                    )

rapids_cpm_init()

# Use one of rapids-cmake pre-configured packages
# And mark rmm as part of our export interface
include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_rmm(BUILD_EXPORT_SET example-targets
               INSTALL_EXPORT_SET example-targets)

# Bring in any arbitrary package
rapids_cpm_find(fmt 7.1.3
  GLOBAL_TARGETS fmt::fmt
  CPM_ARGS
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 7.1.3
    GIT_SHALLOW TRUE
)

add_library(example SHARED source.cu)

set_target_properties(example
    PROPERTIES BUILD_RPATH                         "$ORIGIN"
               INSTALL_RPATH                       "$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               )

target_link_libraries(example
    PUBLIC
    rmm::rmm
    PRIVATE
    fmt::fmt)

# Add Conda env exist s, use it for link and include dirs
if(TARGET conda_env)
  target_link_libraries(example PUBLIC $<BUILD_INTERFACE:conda_env> )
endif()

# - install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir( lib_dir )
install(TARGETS example
        DESTINATION ${lib_dir}
        EXPORT example-targets
        )

################################################################################################
# - install export -------------------------------------------------------------------------------
set(doc_string
[=[
Provide targets for the example library.

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure
dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt
mollit anim id est laborum.

]=])

 rapids_export(INSTALL example
    EXPORT_SET example-targets
    GLOBAL_TARGETS example # since we can't hook into EXPORT SETS
    NAMESPACE example::
    DOCUMENTATION doc_string
    )

################################################################################################
# - build export -------------------------------------------------------------------------------
set(code_string
[=[
  message(STATUS "hi from example-config")
]=])
rapids_export(BUILD example
    EXPORT_SET example-targets
    GLOBAL_TARGETS example # since we can't hook into EXPORT SETS
    LANGUAGES CUDA
    DOCUMENTATION doc_string
    FINAL_CODE_BLOCK code_string
    )
