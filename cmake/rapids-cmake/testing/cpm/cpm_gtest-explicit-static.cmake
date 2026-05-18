# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gtest.cmake)

rapids_cpm_init()
rapids_cpm_gtest(BUILD_STATIC)

get_target_property(type gtest TYPE)
if(NOT type STREQUAL STATIC_LIBRARY)
  message(FATAL_ERROR "rapids_cpm_gtest failed to get a static version of gtest")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/use_gtest.cpp"
     [=[
#include <gtest/gtest.h>

// The fixture for testing class Foo.
class FooTest : public testing::Test {

  FooTest() {}
  ~FooTest() override { }

  void SetUp() override {}
  void TearDown() override {}
};
]=])
add_library(uses_gtest SHARED ${CMAKE_CURRENT_BINARY_DIR}/use_gtest.cpp)
target_link_libraries(uses_gtest PRIVATE GTest::gtest)
