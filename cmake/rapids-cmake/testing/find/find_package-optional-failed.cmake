# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/package.cmake)

rapids_find_package(BUILD_FIND_OPT BUILD_EXPORT_SET test_export_set GLOBAL_TARGETS ZLIB::ZLIB)
rapids_find_package(INSTALL_FIND_OPT INSTALL_EXPORT_SET test_export_set GLOBAL_TARGETS ZLIB::ZLIB)

if(BUILD_FIND_OPT_FOUND OR INSTALL_FIND_OPT_FOUND)
  message(FATAL_ERROR "rapids_find_package should have reported a failed find ")
endif()

if(TARGET ZLIB::ZLIB)
  message(FATAL_ERROR "rapids_find_package shouldn't create targets when the find failed")
endif()

# Check the export information was invoked
if(TARGET rapids_export_build_test_export_set)
  message(FATAL_ERROR "rapids_find_package shouldn't have constructed a export set")
endif()
if(TARGET rapids_export_install_test_export_set)
  message(FATAL_ERROR "rapids_find_package shouldn't have constructed a export set")
endif()
