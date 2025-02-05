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
FindcuDSS
--------

Find cuDSS

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target(s):

``cuDSS::cuDSS``
  The cuDSS library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``cuDSS_FOUND``
  True if cuDSS is found.
``cuDSS_INCLUDE_DIRS``
  The include directories needed to use cuDSS.
``cuDSS_LIBRARIES``
  The libraries needed to usecuDSS.
``cuDSS_VERSION_STRING``
  The version of the cuDSS library found. [OPTIONAL]

#]=======================================================================]

# Prefer using a Config module if it exists for this project
set(cuDSS_NO_CONFIG FALSE)
if(NOT cuDSS_NO_CONFIG)
  find_package(cuDSS CONFIG QUIET HINTS ${cuDSS_DIR})
  if(cuDSS_FOUND)
    find_package_handle_standard_args(cuDSS DEFAULT_MSG cuDSS_CONFIG)
    return()
  endif()
endif()

find_path(cuDSS_INCLUDE_DIR NAMES cuDSS.h )

set(cuDSS_IS_HEADER_ONLY FALSE)
if(NOT cuDSS_LIBRARY AND NOT cuDSS_IS_HEADER_ONLY)
  find_library(cuDSS_LIBRARY_RELEASE NAMES libcuDSS.so NAMES_PER_DIR )
  find_library(cuDSS_LIBRARY_DEBUG   NAMES libcuDSS.sod   NAMES_PER_DIR )

  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(cuDSS)
  unset(cuDSS_FOUND) #incorrectly set by select_library_configurations
endif()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(cuDSS_IS_HEADER_ONLY)
  find_package_handle_standard_args(cuDSS
                                    REQUIRED_VARS cuDSS_INCLUDE_DIR
                                    VERSION_VAR )
else()
  find_package_handle_standard_args(cuDSS
                                    REQUIRED_VARS cuDSS_LIBRARY cuDSS_INCLUDE_DIR
                                    VERSION_VAR )
endif()

if(NOT cuDSS_FOUND)
  set(cuDSS_FILENAME libcuDSS-linux-x86_64-${cuDSS_VERSION}-archive)

  message(STATUS "cuDSS not found. Downloading library. By continuing this download you accept to the license terms of cuDSS")

  CPMAddPackage(
    NAME cuDSS
    VERSION ${cuDSS_VERSION}
    URL https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-${cuDSS_VERSION}_cuda12-archive.tar.xz
    DOWNLOAD_ONLY YES 
  )
      
  set(cuDSS_LIBRARY ${cuDSS_SOURCE_DIR}/lib/libcudss.so)
  set(cuDSS_INCLUDE_DIR ${cuDSS_SOURCE_DIR}/include) 


  set(cuDSS_FOUND TRUE)
endif()

if(cuDSS_FOUND)
  set(cuDSS_INCLUDE_DIRS ${cuDSS_INCLUDE_DIR})

  if(NOT cuDSS_LIBRARIES)
    set(cuDSS_LIBRARIES ${cuDSS_LIBRARY})
  endif()

  if(NOT TARGET cuDSS::cuDSS)
    add_library(cuDSS::cuDSS UNKNOWN IMPORTED)
    set_target_properties(cuDSS::cuDSS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${cuDSS_INCLUDE_DIRS}")

    if(cuDSS_LIBRARY_RELEASE)
      set_property(TARGET cuDSS::cuDSS APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(cuDSS::cuDSS PROPERTIES
        IMPORTED_LOCATION_RELEASE "${cuDSS_LIBRARY_RELEASE}")
    endif()

    if(cuDSS_LIBRARY_DEBUG)
      set_property(TARGET cuDSS::cuDSS APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(cuDSS::cuDSS PROPERTIES
        IMPORTED_LOCATION_DEBUG "${cuDSS_LIBRARY_DEBUG}")
    endif()

    if(NOT cuDSS_LIBRARY_RELEASE AND NOT cuDSS_LIBRARY_DEBUG)
      set_property(TARGET cuDSS::cuDSS APPEND PROPERTY
        IMPORTED_LOCATION "${cuDSS_LIBRARY}")
    endif()
  endif()
endif()

unset(cuDSS_NO_CONFIG)
unset(cuDSS_IS_HEADER_ONLY)
