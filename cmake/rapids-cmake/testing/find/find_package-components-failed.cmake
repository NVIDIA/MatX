# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/package.cmake)

set(CMAKE_PREFIX_PATH "${rapids-cmake-testing-dir}/find/find_package-components/")

rapids_find_package(FakeDependency 11 COMPONENTS AAAAA BUILD_EXPORT_SET test_export_set)

if(FakeDependency_FOUND)
  message(FATAL_ERROR "rapids_find_package recorded incorrect FOUND state for a failed find_package request"
  )
endif()

set(path "${CMAKE_BINARY_DIR}/rapids-cmake/test_export_set/build/package_FakeDependency.cmake")
if(EXISTS "${path}")
  message(FATAL_ERROR "rapids_find_package(BUILD) recorded a failed find_package request")
endif()
