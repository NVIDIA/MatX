# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/package.cmake)

rapids_find_package(CUDAToolkit REQUIRED INSTALL_EXPORT_SET test_export_set
                    BUILD_EXPORT_SET test_export_set)

# no version specified at find time, verify none recorded If we record the found version we break
# things like CUDA backwards runtime compat
set(to_match_string "find_dependency(CUDAToolkit)")

set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_CUDAToolkit.cmake")
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_find_package(BUILD) failed to preserve version information in exported file"
  )
endif()

set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/install/package_CUDAToolkit.cmake")
file(READ "${path}" contents)
string(FIND "${contents}" "${to_match_string}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_find_package(INSTALL) failed to preserve version information in exported file"
  )
endif()
