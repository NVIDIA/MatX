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
include(${rapids-cmake-dir}/export/package.cmake)


rapids_export_package(install ZLIB test_export_set
                      GLOBAL_TARGETS ZLIB::ZLIB
                      )
if(NOT TARGET rapids_export_install_test_export_set)
  message(FATAL_ERROR "rapids_export_package failed to generate target for install")
endif()

rapids_export_package(INSTALL PNG test_export_set
                      GLOBAL_TARGETS PNG::PNG_V2
                      )


# Verify that we encoded both packages for exporting
get_target_property(packages rapids_export_install_test_export_set PACKAGE_NAMES)
if(NOT ZLIB IN_LIST packages)
  message(FATAL_ERROR "rapids_export_package failed to record ZLIB needs to be exported")
endif()
if(NOT PNG IN_LIST packages)
  message(FATAL_ERROR "rapids_export_package failed to record PNG needs to be exported")
endif()

# Verify that we encoded what `targets` are marked as global export
get_target_property( global_targets rapids_export_install_test_export_set GLOBAL_TARGETS)
if( NOT "ZLIB::ZLIB" IN_LIST global_targets)
  message(FATAL_ERROR "rapids_export_package failed to record ZLIB::ZLIB needs to be global")
endif()
if( NOT "PNG::PNG_V2" IN_LIST global_targets)
  message(FATAL_ERROR "rapids_export_package failed to record PNG::PNG_V2 needs to be global")
endif()

# Verify that temp install package configuration files exist
if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/install/package_ZLIB.cmake" OR
   NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/install/package_PNG.cmake")
  message(FATAL_ERROR "rapids_export_package failed to generate a find_package configuration")
endif()

# Verify that temp build package configuration files don't exist
if(EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_ZLIB.cmake" OR
   EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_PNG.cmake")
  message(FATAL_ERROR "rapids_export_package(INSTALL) generated temp files in the wrong directory")
endif()
