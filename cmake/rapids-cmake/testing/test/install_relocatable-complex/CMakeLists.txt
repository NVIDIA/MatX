# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.20)
set(CMAKE_CONFIGURATION_TYPES Debug)
project(rapids-test-project LANGUAGES CUDA)

include(${rapids-cmake-dir}/rapids-test.cmake)

add_library(preload SHARED usage.cu)

# Setup a VERSION and SOVERSION so we generate a multi-line file(INSTALL command
set_target_properties(preload PROPERTIES VERSION 1.1.1 SOVERSION 1)
add_subdirectory(tests)

install(TARGETS preload DESTINATION lib)
