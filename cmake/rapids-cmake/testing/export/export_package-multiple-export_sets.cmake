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
include(${rapids-cmake-dir}/export/package.cmake)


rapids_export_package(BUILD DifferingExportSets export1 GLOBAL_TARGETS EDT::EDT)
rapids_export_package(BUILD DifferingExportSets export2 GLOBAL_TARGETS EDT::EDT)


# Verify that we have the package and targets listed in both export sets
get_target_property(packages1 rapids_export_build_export1 PACKAGE_NAMES)
get_target_property(packages2 rapids_export_build_export2 PACKAGE_NAMES)

get_target_property(global_targets1 rapids_export_build_export1 GLOBAL_TARGETS)
get_target_property(global_targets2 rapids_export_build_export2 GLOBAL_TARGETS)

if(NOT packages1 STREQUAL packages2)
  message(FATAL_ERROR "rapids_export_package failed to record same package is in multiple export sets")
endif()

if(NOT global_targets1 STREQUAL global_targets2)
  message(FATAL_ERROR "rapids_export_package failed to record same target is in multiple export sets")
endif()
