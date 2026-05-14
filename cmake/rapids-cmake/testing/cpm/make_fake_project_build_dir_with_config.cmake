# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

function(make_fake_project_build_dir_with_config name version config_file version_file)
  include(CMakePackageConfigHelpers)

  # Generate a fake config module for RapidsTestFind
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}-build/")
  file(MAKE_DIRECTORY ${build_dir})

  configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/${config_file}"
                                "${build_dir}/${config_file}"
                                INSTALL_DESTINATION "${build_dir}/${config_file}")

  write_basic_package_version_file("${build_dir}/${version_file}" VERSION ${version}
                                   COMPATIBILITY SameMajorVersion)
endfunction()
