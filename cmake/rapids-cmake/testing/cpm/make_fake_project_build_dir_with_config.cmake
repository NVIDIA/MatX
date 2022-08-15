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

function(make_fake_project_build_dir_with_config name version config_file version_file)
  include(CMakePackageConfigHelpers)

  # Generate a fake config module for RapidsTestFind
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}-build/")
  file(MAKE_DIRECTORY ${build_dir})

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${config_file}"
    "${build_dir}/${config_file}"
    INSTALL_DESTINATION "${build_dir}/${config_file}"
  )

  write_basic_package_version_file("${build_dir}/${version_file}"
    VERSION ${version}
    COMPATIBILITY SameMajorVersion)
endfunction()
