# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/package.cmake)

rapids_find_package(ZLIB 99999999999 EXACT REQUIRED INSTALL_EXPORT_SET test_export_set
                    GLOBAL_TARGETS ZLIB::ZLIB)
