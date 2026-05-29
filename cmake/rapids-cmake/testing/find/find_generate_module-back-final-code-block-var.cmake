# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/generate_module.cmake)

rapids_find_generate_module(RapidsTest HEADER_NAMES rapids-cmake-test-header_only.hpp
                                                    FINAL_CODE_BLOCK var_doesn't_exist
                            INSTALL_EXPORT_SET test_set)
