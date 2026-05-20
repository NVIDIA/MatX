# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
add_cmake_config_test(rapids-find.cmake)

add_cmake_config_test(find_package-no-variable-leak.cmake)

add_cmake_config_test(find_package-components)
add_cmake_config_test(find_package-components-failed.cmake)

add_cmake_config_test(find_package-build.cmake)
add_cmake_config_test(find_package-install.cmake)

add_cmake_config_test(find_package-cudatoolkit-patching.cmake)

# The inverse of `find_package-optional-failed` is the above two tests. The inverse of
# `find_package-required-found` is always a hard error and therefore has no test
add_cmake_config_test(find_package-optional-failed.cmake)
add_cmake_config_test(find_package-required-found.cmake)
add_cmake_config_test(find_package-version-explicit-failed.cmake)

add_cmake_config_test(find_package-version-explicit.cmake)
add_cmake_config_test(find_package-version-none.cmake)
add_cmake_config_test(find_package-version-required-explicit-failed.cmake SHOULD_FAIL
                      "but required is exact version \"99999999999\"")

add_cmake_config_test(find_generate_module-build)
add_cmake_config_test(find_generate_module-install)
add_cmake_config_test(find_generate_module-header-only)

add_cmake_config_test(find_generate_module-code-blocks)
add_cmake_config_test(find_generate_module-back-initial-code-block-var.cmake SHOULD_FAIL
                      "INITIAL_CODE_BLOCK variable `var_doesn't_exist` doesn't exist")
add_cmake_config_test(find_generate_module-back-final-code-block-var.cmake SHOULD_FAIL
                      "FINAL_CODE_BLOCK variable `var_doesn't_exist` doesn't exist")

add_cmake_config_test(find_generate_module-verify-copyright-header.cmake)
