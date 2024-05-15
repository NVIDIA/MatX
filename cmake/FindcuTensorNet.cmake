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
FindcuTensorNet
--------

Find cuTensorNet

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target(s):

``cuTensorNet::cuTensorNet``
  The cuTensorNet library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``cuTensorNet_FOUND``
  True if cuTensorNet is found.
``cuTensorNet_INCLUDE_DIRS``
  The include directories needed to use cuTensorNet.
``cuTensorNet_LIBRARIES``
  The libraries needed to usecuTensorNet.
``cuTensorNet_VERSION_STRING``
  The version of the cuTensorNet library found. [OPTIONAL]

#]=======================================================================]

# Prefer using a Config module if it exists for this project
set(cuTensorNet_NO_CONFIG FALSE)
if(NOT cuTensorNet_NO_CONFIG)
  find_package(cuTensorNet CONFIG QUIET HINTS ${cutensornet_DIR})
  if(cuTensorNet_FOUND)
    find_package_handle_standard_args(cuTensorNet DEFAULT_MSG cuTensorNet_CONFIG)
    return()
  endif()
endif()

find_path(cuTensorNet_INCLUDE_DIR NAMES cutensornet.h )

set(cuTensorNet_IS_HEADER_ONLY FALSE)
if(NOT cuTensorNet_LIBRARY AND NOT cuTensorNet_IS_HEADER_ONLY)
  find_library(cuTensorNet_LIBRARY_RELEASE NAMES libcutensornet.so NAMES_PER_DIR )
  find_library(cuTensorNet_LIBRARY_DEBUG   NAMES libcutensornet.sod   NAMES_PER_DIR )

  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(cuTensorNet)
  unset(cuTensorNet_FOUND) #incorrectly set by select_library_configurations
endif()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(cuTensorNet_IS_HEADER_ONLY)
  find_package_handle_standard_args(cuTensorNet
                                    REQUIRED_VARS cuTensorNet_INCLUDE_DIR
                                    VERSION_VAR )
else()
  find_package_handle_standard_args(cuTensorNet
                                    REQUIRED_VARS cuTensorNet_LIBRARY cuTensorNet_INCLUDE_DIR
                                    VERSION_VAR )
endif()

if(NOT cuTensorNet_FOUND)
  message(STATUS "cuTensorNet not found. Downloading library. By continuing this download you accept to the license terms of cuQuantum SDK")

  set(CUTENSORNET_FILENAME cuquantum-linux-x86_64-${CUTENSORNET_VERSION}_cuda${CUDAToolkit_VERSION_MAJOR}-archive)
  
  CPMAddPackage(
               NAME cutensornet
               VERSION ${CUTENSORNET_VERSION}
               URL https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/${CUTENSORNET_FILENAME}.tar.xz
               # Eigen's CMakelists are not intended for library use
               DOWNLOAD_ONLY YES 
               )
      
  set(cuTensorNet_LIBRARY ${cutensornet_SOURCE_DIR}/lib/libcutensornet.so) 
  set(cuTensorNet_INCLUDE_DIR ${cutensornet_SOURCE_DIR}/include) 

  set(cuTensorNet_FOUND TRUE)
endif()

if(cuTensorNet_FOUND)
  set(cuTensorNet_INCLUDE_DIRS ${cuTensorNet_INCLUDE_DIR})

  if(NOT cuTensorNet_LIBRARIES)
    set(cuTensorNet_LIBRARIES ${cuTensorNet_LIBRARY})
  endif()

  if(NOT TARGET cuTensorNet::cuTensorNet)
    add_library(cuTensorNet::cuTensorNet UNKNOWN IMPORTED)
    set_target_properties(cuTensorNet::cuTensorNet PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${cuTensorNet_INCLUDE_DIRS}")

    if(cuTensorNet_LIBRARY_RELEASE)
      set_property(TARGET cuTensorNet::cuTensorNet APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(cuTensorNet::cuTensorNet PROPERTIES
        IMPORTED_LOCATION_RELEASE "${cuTensorNet_LIBRARY_RELEASE}")
    endif()

    if(cuTensorNet_LIBRARY_DEBUG)
      set_property(TARGET cuTensorNet::cuTensorNet APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(cuTensorNet::cuTensorNet PROPERTIES
        IMPORTED_LOCATION_DEBUG "${cuTensorNet_LIBRARY_DEBUG}")
    endif()

    if(NOT cuTensorNet_LIBRARY_RELEASE AND NOT cuTensorNet_LIBRARY_DEBUG)
      set_property(TARGET cuTensorNet::cuTensorNet APPEND PROPERTY
        IMPORTED_LOCATION "${cuTensorNet_LIBRARY}")
    endif()
  endif()
endif()

unset(cuTensorNet_NO_CONFIG)
unset(cuTensorNet_IS_HEADER_ONLY)
