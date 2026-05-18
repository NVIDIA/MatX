# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)
project(rapids-test-a LANGUAGES CXX)

include(${rapids-cmake-dir}/cpm/init.cmake)
rapids_cpm_init()

include(${rapids-cmake-dir}/cpm/gbench.cmake)
rapids_cpm_gbench()
