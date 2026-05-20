# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/package_override.cmake)

rapids_cpm_init()

# Write an invalid override file with url_hash but no url
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/override.json
     [=[
{
  "packages": {
    "invalid_pkg": {
      "version": "1.0.0",
      "url_hash": "SHA256=deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678"
    }
  }
}
  ]=])

rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/override.json)

# This should fail because url_hash is provided without url
include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
rapids_cpm_package_info(invalid_pkg VERSION_VAR version CPM_VAR cpm_args)
