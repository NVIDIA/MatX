# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/export.cmake)

cmake_minimum_required(VERSION 3.20)
project(FakEProJecT LANGUAGES CXX VERSION 3.1.4)

add_library(fakeLib INTERFACE)
install(TARGETS fakeLib EXPORT fake_set)

add_library(fakeLib_c1 INTERFACE)
install(TARGETS fakeLib_c1 EXPORT fake_set_c1)

rapids_export(BUILD FakEProJecT EXPORT_SET fake_set COMPONENTS c1 NAMESPACE test::)
