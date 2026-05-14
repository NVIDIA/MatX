# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/test/init.cmake)
include(${rapids-cmake-dir}/test/add.cmake)

enable_language(CUDA)

rapids_test_init()

file(WRITE "${CMAKE_BINARY_DIR}/main.cu" "int main(){return 0;}")
add_executable(verify_alloc "${CMAKE_BINARY_DIR}/main.cu")

enable_testing()
rapids_test_add(NAME simple_test COMMAND verify_alloc INSTALL_COMPONENT_SET testing)

get_test_property(simple_test RESOURCE_GROUPS value)
if(value)
  message(FATAL_ERROR "Unexpected RESOURCE_GROUPS test property value(${value}) found, should not be set"
  )
endif()
