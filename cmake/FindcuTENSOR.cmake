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
FindcuTENSOR
--------

Find cuTENSOR

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target(s):

``cuTENSOR::cuTENSOR``
  The cuTENSOR library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``cuTENSOR_FOUND``
  True if cuTENSOR is found.
``cuTENSOR_INCLUDE_DIRS``
  The include directories needed to use cuTENSOR.
``cuTENSOR_LIBRARIES``
  The libraries needed to usecuTENSOR.
``cuTENSOR_VERSION_STRING``
  The version of the cuTENSOR library found. [OPTIONAL]

#]=======================================================================]

# Prefer using a Config module if it exists for this project
set(cuTENSOR_NO_CONFIG FALSE)
if(NOT cuTENSOR_NO_CONFIG)
  find_package(cuTENSOR CONFIG QUIET HINTS ${cutensor_DIR})
  if(cuTENSOR_FOUND)
    find_package_handle_standard_args(cuTENSOR DEFAULT_MSG cuTENSOR_CONFIG)
    return()
  endif()
endif()

find_path(cuTENSOR_INCLUDE_DIR NAMES cutensor.h )

set(cuTENSOR_IS_HEADER_ONLY FALSE)
if(NOT cuTENSOR_LIBRARY AND NOT cuTENSOR_IS_HEADER_ONLY)
  find_library(cuTENSOR_LIBRARY_RELEASE NAMES libcutensor.so NAMES_PER_DIR )
  find_library(cuTENSOR_LIBRARY_DEBUG   NAMES libcutensor.sod   NAMES_PER_DIR )

  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(cuTENSOR)
  unset(cuTENSOR_FOUND) #incorrectly set by select_library_configurations
endif()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(cuTENSOR_IS_HEADER_ONLY)
  find_package_handle_standard_args(cuTENSOR
                                    REQUIRED_VARS cuTENSOR_INCLUDE_DIR
                                    VERSION_VAR )
else()
  find_package_handle_standard_args(cuTENSOR
                                    REQUIRED_VARS cuTENSOR_LIBRARY cuTENSOR_INCLUDE_DIR
                                    VERSION_VAR )
endif()

if(NOT cuTENSOR_FOUND)
  set(CUTENSOR_FILENAME libcutensor-linux-x86_64-${CUTENSOR_VERSION}-archive)

  message(STATUS "cuTENSOR not found. Downloading library. By continuing this download you accept to the license terms of cuTENSOR")

  CPMAddPackage(
    NAME cutensor
    VERSION ${CUTENSOR_VERSION}
    URL https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-${CUTENSOR_VERSION}-archive.tar.xz
    # Eigen's CMakelists are not intended for library use
    DOWNLOAD_ONLY YES 
  )
      
  set(cuTENSOR_LIBRARY ${cutensor_SOURCE_DIR}/lib/${CUDAToolkit_VERSION_MAJOR}/libcutensor.so)
  set(cuTENSOR_INCLUDE_DIR ${cutensor_SOURCE_DIR}/include) 


  set(cuTENSOR_FOUND TRUE)
endif()

if(cuTENSOR_FOUND)
  set(cuTENSOR_INCLUDE_DIRS ${cuTENSOR_INCLUDE_DIR})

  if(NOT cuTENSOR_LIBRARIES)
    set(cuTENSOR_LIBRARIES ${cuTENSOR_LIBRARY})
  endif()

  if(NOT TARGET cuTENSOR::cuTENSOR)
    add_library(cuTENSOR::cuTENSOR UNKNOWN IMPORTED)
    set_target_properties(cuTENSOR::cuTENSOR PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${cuTENSOR_INCLUDE_DIRS}")

    if(cuTENSOR_LIBRARY_RELEASE)
      set_property(TARGET cuTENSOR::cuTENSOR APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(cuTENSOR::cuTENSOR PROPERTIES
        IMPORTED_LOCATION_RELEASE "${cuTENSOR_LIBRARY_RELEASE}")
    endif()

    if(cuTENSOR_LIBRARY_DEBUG)
      set_property(TARGET cuTENSOR::cuTENSOR APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(cuTENSOR::cuTENSOR PROPERTIES
        IMPORTED_LOCATION_DEBUG "${cuTENSOR_LIBRARY_DEBUG}")
    endif()

    if(NOT cuTENSOR_LIBRARY_RELEASE AND NOT cuTENSOR_LIBRARY_DEBUG)
      set_property(TARGET cuTENSOR::cuTENSOR APPEND PROPERTY
        IMPORTED_LOCATION "${cuTENSOR_LIBRARY}")
    endif()
  endif()
endif()

unset(cuTENSOR_NO_CONFIG)
unset(cuTENSOR_IS_HEADER_ONLY)
