# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
add_cmake_config_test(rapids-cuda.cmake)

add_cmake_config_test(enable_fatbin_compression-invalid-tuning-flag.cmake SHOULD_FAIL
                      "rapids_cuda_enable_fatbin_compression `TUNE_FOR`")
add_cmake_config_test(enable_fatbin_compression-no-arguments.cmake SHOULD_FAIL
                      "rapids_cuda_enable_fatbin_compression requires")
add_cmake_config_test(enable_fatbin_compression-target.cmake)
add_cmake_config_test(enable_fatbin_compression-var.cmake)

add_cmake_config_test(init_runtime-existing.cmake)
add_cmake_config_test(init_runtime-multiple.cmake)
add_cmake_config_test(init_runtime-shared.cmake)
add_cmake_config_test(init_runtime-static.cmake)

add_cmake_config_test(set_runtime-shared.cmake)
add_cmake_config_test(set_runtime-static.cmake)
add_cmake_config_test(set_runtime-on-interface-target.cmake)

add_cmake_config_test(init_arch-all-via-undef)
add_cmake_config_test(init_arch-existing-project-flags.cmake)
add_cmake_config_test(init_arch-native.cmake)
add_cmake_config_test(init_arch-native-via-env.cmake)
add_cmake_config_test(init_arch-native-no-gpu.cmake)
add_cmake_config_test(init_arch-rapids.cmake)
add_cmake_config_test(init_arch-rapids-via-env.cmake)
add_cmake_config_test(init_arch-user.cmake)
add_cmake_config_test(init_arch-user-via-env.cmake)
add_cmake_config_test(init_arch-user-over-env.cmake)
add_cmake_config_test(init_arch-var-over-env.cmake)

add_cmake_config_test(set_arch-existing.cmake)
add_cmake_config_test(set_arch-invalid-mode.cmake)
add_cmake_config_test(set_arch-native.cmake)
add_cmake_config_test(set_arch-rapids.cmake)
