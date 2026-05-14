# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cython/create_modules.cmake)

# Test that a invocation without calling rapids_cython_init fails.
rapids_cython_create_modules(SOURCE_FILES test.pyx)
