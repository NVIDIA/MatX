#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

#[=======================================================================[.rst:
FindMathDx
--------

Find MathDx

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target(s):

``MathDx::MathDx``
  The MathDx library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``MathDx_FOUND``
  True if MathDx is found.
``MathDx_INCLUDE_DIRS``
  The include directories needed to use MathDx.
``MathDx_VERSION_STRING``
  The version of the MathDx library found. [OPTIONAL]

#]=======================================================================]
set(MathDx_VERSION_FULL ${MathDx_VERSION}.${MathDx_NANO})

# Prefer using a Config module if it exists for this project
set(MathDx_NO_CONFIG FALSE)
if(NOT MathDx_NO_CONFIG)
  find_package(MathDx CONFIG QUIET HINTS ${MathDx_DIR})
  if(MathDx_FOUND)
    find_package_handle_standard_args(MathDx DEFAULT_MSG MathDx_CONFIG)
    return()
  endif()
endif()

find_path(MathDx_INCLUDE_DIR NAMES MathDx.h )

set(MathDx_IS_HEADER_ONLY TRUE)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)


find_package_handle_standard_args(MathDx
                                  REQUIRED_VARS MathDx_LIBRARY MathDx_INCLUDE_DIR
                                  VERSION_VAR )

if(NOT MathDx_FOUND)
  set(MathDx_FILENAME libMathDx-linux-x86_64-${MathDx_VERSION}-archive)

  message(STATUS "MathDx not found. Downloading library. By continuing this download you accept to the license terms of MathDx")

  CPMAddPackage(
    NAME MathDx
    VERSION ${MathDx_VERSION}
    URL https://developer.download.nvidia.com/compute/cuFFTDx/redist/cuFFTDx/nvidia-mathdx-${MathDx_VERSION_FULL}.tar.gz
    DOWNLOAD_ONLY YES 
  )
endif()

# Download libmathdx based on CUDA version and platform
# Detect CUDA version (12 or 13)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  set(LIBMATHDX_CUDA_VERSION "cuda13")
  set(LIBMATHDX_CUDA_SUFFIX "cuda13.0")
else()
  set(LIBMATHDX_CUDA_VERSION "cuda12")
  set(LIBMATHDX_CUDA_SUFFIX "cuda12.0")
endif()

# Detect platform
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
  message(WARNING "Unsupported platform for libmathdx download")
endif()

# Set libmathdx version
set(LIBMATHDX_VERSION "0.2.3")

# Download libmathdx if platform is supported
if(DEFINED LIBMATHDX_PLATFORM)
  set(LIBMATHDX_URL "https://developer.nvidia.com/downloads/compute/cublasdx/redist/cublasdx/${LIBMATHDX_CUDA_VERSION}/libmathdx-${LIBMATHDX_PLATFORM}-${LIBMATHDX_VERSION}-${LIBMATHDX_CUDA_SUFFIX}.${LIBMATHDX_EXT}")
  
  message(STATUS "Downloading libmathdx for ${LIBMATHDX_PLATFORM} with ${LIBMATHDX_CUDA_VERSION}")
  message(STATUS "libmathdx URL: ${LIBMATHDX_URL}")
  
  CPMAddPackage(
    NAME libmathdx
    VERSION ${LIBMATHDX_VERSION}
    URL ${LIBMATHDX_URL}
    DOWNLOAD_ONLY YES
  )
  
  # Add libmathdx to the search paths
  set(LIBMATHDX_ROOT "${PROJECT_BINARY_DIR}/_deps/libmathdx-src")
  list(APPEND CMAKE_PREFIX_PATH "${LIBMATHDX_ROOT}")
  
  # Find libmathdx library file
  find_library(LIBMATHDX_LIBRARY
    NAMES mathdx libmathdx
    PATHS "${LIBMATHDX_ROOT}/lib"
    NO_DEFAULT_PATH
  )
  
  # Set include directories
  set(LIBMATHDX_INCLUDE_DIR "${LIBMATHDX_ROOT}/include")
  set(LIBMATHDX_INCLUDE_DIR "${LIBMATHDX_ROOT}/include" PARENT_SCOPE)
  
  if(LIBMATHDX_LIBRARY AND EXISTS ${LIBMATHDX_INCLUDE_DIR})
    message(STATUS "Found libmathdx library: ${LIBMATHDX_LIBRARY}")
    message(STATUS "Found libmathdx include dir: ${LIBMATHDX_INCLUDE_DIR}")
    
    # Create libmathdx target
    if(NOT TARGET libmathdx::libmathdx)
      add_library(libmathdx::libmathdx INTERFACE IMPORTED)
      set_target_properties(libmathdx::libmathdx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBMATHDX_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${LIBMATHDX_LIBRARY}"
      )
    endif()
  else()
    message(WARNING "Could not find libmathdx library or include directory after download")
  endif()
endif()

find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG
PATHS
    "${PROJECT_BINARY_DIR}/_deps/mathdx-src/nvidia/mathdx/${MathDx_VERSION}/lib/cmake/mathdx/"
    "${PROJECT_BINARY_DIR}/_deps/libmathdx-src/lib/cmake/libmathdx/"
    "${PROJECT_BINARY_DIR}/_deps/libmathdx-src"
    "/opt/nvidia/mathdx/${MathDx_VERSION_FULL}"
)

