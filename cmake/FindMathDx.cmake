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

find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG
PATHS
    "${PROJECT_BINARY_DIR}/_deps/mathdx-src/nvidia/mathdx/${MathDx_VERSION}/lib/cmake/mathdx/"
    "/opt/nvidia/mathdx/${MathDx_VERSION_FULL}"
)

