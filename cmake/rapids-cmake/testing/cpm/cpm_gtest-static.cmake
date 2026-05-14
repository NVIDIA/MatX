# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/gtest.cmake)

rapids_cpm_init()

if(TARGET GTest::gtest)
  message(FATAL_ERROR "Expected GTest::gtest not to exist")
endif()

set(BUILD_SHARED_LIBS OFF)
rapids_cpm_gtest()

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
