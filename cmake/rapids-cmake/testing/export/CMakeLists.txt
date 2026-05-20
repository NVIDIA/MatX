# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
add_cmake_config_test(rapids-export.cmake)

add_cmake_config_test(export_cpm-build-possible-dir.cmake)
add_cmake_config_test(export_cpm-build.cmake)
add_cmake_config_test(export_cpm-install.cmake)
add_cmake_config_test(export_cpm-options-escaped.cmake)

add_cmake_build_test(export-verify-build-components-dependencies)
add_cmake_build_test(export-verify-build-components-targets)
add_cmake_build_test(export-verify-build-multiple-export-sets-to-single-component)
add_cmake_build_test(export-verify-build-namespaces)
add_cmake_build_test(export-verify-code-block)
add_cmake_build_test(export-verify-doc-string)
add_cmake_build_test(export-verify-global-targets)
add_cmake_build_test(export-verify-install-components)
add_cmake_build_test(export-verify-no-namespace)

add_cmake_config_test(export-verify-bad-code-block-var.cmake SHOULD_FAIL
                      "FINAL_CODE_BLOCK variable `var_doesn't_exist` doesn't exist")
add_cmake_config_test(export-verify-missing-components.cmake SHOULD_FAIL
                      "is missing COMPONENTS as COMPONENTS_EXPORT_SET was provided")
add_cmake_config_test(export-verify-missing-components-export.cmake SHOULD_FAIL
                      "is missing COMPONENTS_EXPOR22T_SET as COMPONENTS was provided")
add_cmake_config_test(export-verify-bad-doc-var.cmake SHOULD_FAIL
                      "DOCUMENTATION variable `var_doesn't_exist` doesn't exist")
add_cmake_config_test(export-verify-calendar-version-matching.cmake)
add_cmake_config_test(export-verify-explicit-disabled-version.cmake)
add_cmake_config_test(export-verify-explicit-major-version-only-matching.cmake)
add_cmake_config_test(export-verify-explicit-patch-version-value-of-zero.cmake)
add_cmake_config_test(export-verify-explicit-version-all-zeros.cmake)
add_cmake_config_test(export-verify-explicit-version-value.cmake)
add_cmake_config_test(export-verify-file-names.cmake)
add_cmake_config_test(export-verify-implicit-disabled-version.cmake)
add_cmake_config_test(export-verify-implicit-major-version-only-matching.cmake)
add_cmake_config_test(export-verify-version.cmake)
add_cmake_config_test(export-verify-copyright-header.cmake)

add_cmake_config_test(export_component-build)

add_cmake_config_test(export_package-build-possible-dir.cmake)
add_cmake_config_test(export_package-build-post-find-code.cmake)
add_cmake_config_test(export_package-build-with-components-and-version.cmake)
add_cmake_config_test(export_package-build-with-components.cmake)
add_cmake_config_test(export_package-build-with-empty-components.cmake)
add_cmake_config_test(export_package-build-with-version.cmake)
add_cmake_config_test(export_package-build.cmake)
add_cmake_config_test(export_package-install-possible-dir.cmake)
add_cmake_config_test(export_package-install-post-find-code.cmake)
add_cmake_config_test(export_package-install-with-components-and-version.cmake)
add_cmake_config_test(export_package-install-with-components.cmake)
add_cmake_config_test(export_package-install-with-version.cmake)
add_cmake_config_test(export_package-install.cmake)
add_cmake_config_test(export_package-multiple-export_sets.cmake)

add_cmake_config_test(write_dependencies-cpm-preserve-options.cmake)
add_cmake_config_test(write_dependencies-duplicate-cpm.cmake)
add_cmake_config_test(write_dependencies-duplicate-packages.cmake)
add_cmake_config_test(write_dependencies-duplicate-cpm-and-package.cmake)
add_cmake_config_test(write_dependencies-multiple-directories)
add_cmake_config_test(write_dependencies-root-dirs.cmake)

add_cmake_build_test(write_language-multiple-nested-enables)
add_cmake_build_test(write_language-nested-dirs)
