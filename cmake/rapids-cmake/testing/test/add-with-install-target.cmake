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
rapids_test_add(NAME simple_test COMMAND ${CMAKE_COMMAND} -E env verify_alloc GPUS 1
                INSTALL_COMPONENT_SET testing INSTALL_TARGET verify_alloc)

# Verify that we have recorded `simple_test` as part of the `testing` component
get_target_property(names rapids_test_install_testing TESTS_TO_RUN)
if(NOT "simple_test" IN_LIST names)
  message(FATAL_ERROR "Failed to record `simple_test` as part of the testing component")
endif()

# Verify that `verify_alloc` is marked as to be installed
get_target_property(names rapids_test_install_testing TARGETS_TO_INSTALL)
if(NOT "verify_alloc" IN_LIST names)
  message(FATAL_ERROR "Failed to record `verify_alloc` as a target to be installed in the testing component"
  )
endif()
