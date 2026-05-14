# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/cccl.cmake)

rapids_cpm_init()

rapids_cpm_cccl(INSTALL_EXPORT_SET example_export)

include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
rapids_cpm_package_info(cccl VERSION_VAR cccl_version)

set(cccl_path "${CMAKE_BINARY_DIR}/rapids-cmake/example_export/install/package_CCCL.cmake")

file(READ "${cccl_path}" contents)
message(STATUS "contents: ${contents}")
string(FIND "${contents}" "${cccl_version}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_cpm_cccl failed to generate a find_package configuration with version"
  )
endif()
