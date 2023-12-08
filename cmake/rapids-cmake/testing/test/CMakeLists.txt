#=============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
add_cmake_config_test( rapids-test.cmake )

add_cmake_config_test(generate_resource_spec-no-gpus.cmake)
add_cmake_config_test(generate_resource_spec-simple.cmake)

set(percent_error_message "rapids_test_gpu_requirements requires a numeric PERCENT value [1-100].")
set(gpu_error_message "rapids_test_gpu_requirements GPUS requires a numeric value")
set(gpu_missing_message "rapids_test_gpu_requirements requires the GPUS option to be provided")
add_cmake_config_test(gpu_requirements-char-percent.cmake SHOULD_FAIL "${percent_error_message}")
add_cmake_config_test(gpu_requirements-char-gpu.cmake SHOULD_FAIL "${gpu_error_message}")
add_cmake_config_test(gpu_requirements-zero-gpu.cmake)
add_cmake_config_test(gpu_requirements-max-percent.cmake)
add_cmake_config_test(gpu_requirements-min-percent.cmake)
add_cmake_config_test(gpu_requirements-multi.cmake)
add_cmake_config_test(gpu_requirements-negative-percent.cmake SHOULD_FAIL "${percent_error_message}")
add_cmake_config_test(gpu_requirements-negative-gpu.cmake SHOULD_FAIL "${gpu_error_message}")
add_cmake_config_test(gpu_requirements-no-gpus.cmake SHOULD_FAIL "${gpu_missing_message}")
add_cmake_config_test(gpu_requirements-no-percent.cmake)
add_cmake_config_test(gpu_requirements-over-max-percent.cmake SHOULD_FAIL "${percent_error_message}")
add_cmake_config_test(gpu_requirements-single.cmake)
add_cmake_config_test(gpu_requirements-zero-percent.cmake SHOULD_FAIL "${percent_error_message}")

add_cmake_config_test(init-existing-specfile.cmake)
add_cmake_config_test(init-simple.cmake)

set(wrong_component_message "No install component set [wrong_component] can be found")
add_cmake_build_test(install_relocatable-include-in-all.cmake)
add_cmake_build_test(install_relocatable-labels.cmake)
add_cmake_build_test(install_relocatable-no-tests.cmake)
add_cmake_build_test(install_relocatable-simple.cmake)
add_cmake_build_test(install_relocatable-complex)
add_cmake_config_test(install_relocatable-wrong-component.cmake SHOULD_FAIL "${wrong_component_message}")

add_cmake_ctest_test(add-impossible-allocation SHOULD_FAIL "Insufficient resources for test")
add_cmake_config_test(add-with-install-component.cmake)
add_cmake_config_test(add-with-no-gpus.cmake)

add_cmake_ctest_test(add-no-cuda-toolkit)
if(RAPIDS_CMAKE_TESTING_GPU_COUNT GREATER 0)
  add_cmake_ctest_test(add-allocation-simple)
  add_cmake_ctest_test(add-allocation-throws-runtime-error)
  add_cmake_ctest_test(add-multi-allocations-same-gpu)
  add_cmake_ctest_test(add-multi-jobs-single-gpu SERIAL)
  add_cmake_ctest_test(no_add-with-cpp-handler-from-docs)
  add_cmake_ctest_test(no_add-with-no-resource-spec)
endif()
if(RAPIDS_CMAKE_TESTING_GPU_COUNT GREATER 1)
  add_cmake_ctest_test(add-single-job-multi-gpu)
endif()
