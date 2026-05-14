# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/generate_module.cmake)
include(${rapids-cmake-testing-dir}/utils/check_copyright_header.cmake)

rapids_find_generate_module(RapidsTest HEADER_NAMES rapids-cmake-test-header_only.hpp
                            INSTALL_EXPORT_SET test_set)

check_copyright_header("${CMAKE_BINARY_DIR}/cmake/find_modules/FindRapidsTest.cmake")
