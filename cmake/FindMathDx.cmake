#=============================================================================
# Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

set(MathDx_VERSION_FULL ${MathDx_VERSION}.${MathDx_NANO})

if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  message(FATAL_ERROR "MATX_EN_MATHDX requires CUDA 13.0 or newer for MathDx ${MathDx_VERSION_FULL}")
endif()

set(MATHDX_CUDA_VERSION "cuda13")
set(MATHDX_CUDA_SUFFIX "cuda13.0")

message(STATUS "Using MathDx ${MathDx_VERSION_FULL} (${MATHDX_CUDA_VERSION})")
message(STATUS "Using libmathdx ${LIBMATHDX_VERSION} (${MATHDX_CUDA_VERSION})")

CPMAddPackage(
  NAME MathDx
  VERSION ${MathDx_VERSION_FULL}
  URL https://developer.nvidia.com/downloads/compute/cuSOLVERDx/redist/cuSOLVERDx/${MATHDX_CUDA_VERSION}/nvidia-mathdx-${MathDx_VERSION_FULL}-${MATHDX_CUDA_VERSION}.tar.gz
  DOWNLOAD_ONLY YES
)

if(WIN32)
  set(LIBMATHDX_PLATFORM "win32-x86_64")
  set(LIBMATHDX_EXT "zip")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(LIBMATHDX_PLATFORM "Linux-aarch64")
  else()
    set(LIBMATHDX_PLATFORM "Linux-x86_64")
  endif()
  set(LIBMATHDX_EXT "tar.gz")
else()
  message(FATAL_ERROR "Unsupported platform for libmathdx download")
endif()

set(LIBMATHDX_URL "https://developer.nvidia.com/downloads/compute/cublasdx/redist/cublasdx/${MATHDX_CUDA_VERSION}/libmathdx-${LIBMATHDX_PLATFORM}-${LIBMATHDX_VERSION}-${MATHDX_CUDA_SUFFIX}.${LIBMATHDX_EXT}")

message(STATUS "libmathdx URL: ${LIBMATHDX_URL}")

CPMAddPackage(
  NAME libmathdx
  VERSION ${LIBMATHDX_VERSION}
  URL ${LIBMATHDX_URL}
  DOWNLOAD_ONLY YES
)

set(MATX_MATHDX_ROOT "${PROJECT_BINARY_DIR}/_deps/mathdx-src/nvidia/mathdx/${MathDx_VERSION}")
set(MATHDX_INCLUDE_DIR "${MATX_MATHDX_ROOT}/include")
set(MATHDX_CUTLASS_INCLUDE_DIR "${MATX_MATHDX_ROOT}/external/cutlass/include")
set(LIBMATHDX_ROOT "${PROJECT_BINARY_DIR}/_deps/libmathdx-src")
set(LIBMATHDX_INCLUDE_DIR "${LIBMATHDX_ROOT}/include")

find_library(LIBMATHDX_LIBRARY
  NAMES mathdx libmathdx
  PATHS "${LIBMATHDX_ROOT}/lib"
  NO_DEFAULT_PATH
)

if(NOT LIBMATHDX_LIBRARY OR NOT EXISTS "${LIBMATHDX_INCLUDE_DIR}/libmathdx.h")
  message(FATAL_ERROR "Could not find libmathdx ${LIBMATHDX_VERSION} library or headers after download")
endif()

if(NOT TARGET libmathdx::libmathdx)
  add_library(libmathdx::libmathdx INTERFACE IMPORTED)
  set_target_properties(libmathdx::libmathdx PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LIBMATHDX_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${LIBMATHDX_LIBRARY}"
  )
endif()

set(cublasdx_CUTLASS_ROOT "${MATX_MATHDX_ROOT}/external/cutlass")
set(cusolverdx_CUTLASS_ROOT "${MATX_MATHDX_ROOT}/external/cutlass")

find_package(mathdx REQUIRED COMPONENTS cufftdx cublasdx cusolverdx curanddx CONFIG
  PATHS
    "${MATX_MATHDX_ROOT}/lib/cmake/mathdx"
    "/opt/nvidia/mathdx/${MathDx_VERSION}"
)

set(MATHDX_ROOT "${MATX_MATHDX_ROOT}")
