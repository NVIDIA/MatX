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
include(${rapids-cmake-dir}/find/package.cmake)

rapids_find_package(ZLIB INSTALL_EXPORT_SET test_export_set GLOBAL_TARGETS ZLIB::ZLIB)
rapids_find_package(PNG INSTALL_EXPORT_SET test_export_set )

if(ZLIB_FOUND)
  get_target_property(is_imported ZLIB::ZLIB IMPORTED)
  get_target_property(is_global ZLIB::ZLIB IMPORTED_GLOBAL)
  if(NOT is_imported OR NOT is_global)
  	message(FATAL_ERROR "rapids_find_package failed to make ZLIB::ZLIB GLOBAL")
  endif()
endif()

# Check the export information was invoked
if(NOT TARGET rapids_export_install_test_export_set)
  message(FATAL_ERROR "rapids_find_package failed to generate target for build")
endif()

# Verify that we encoded both packages for exporting
get_target_property(packages rapids_export_install_test_export_set PACKAGE_NAMES)
if(ZLIB_FOUND AND NOT ZLIB IN_LIST packages)
  message(FATAL_ERROR "rapids_find_package failed to record ZLIB needs to be exported")
endif()
if(PNG_FOUND AND NOT PNG IN_LIST packages)
  message(FATAL_ERROR "rapids_find_package failed to record PNG needs to be exported")
endif()

# Verify that we encoded what `targets` are marked as global export
get_target_property( global_targets rapids_export_install_test_export_set GLOBAL_TARGETS)
if(ZLIB_FOUND AND NOT "ZLIB::ZLIB" IN_LIST global_targets)
  message(FATAL_ERROR "rapids_find_package failed to record ZLIB::ZLIB needs to be global")
endif()
