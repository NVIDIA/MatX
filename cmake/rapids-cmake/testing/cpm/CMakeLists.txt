# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
add_cmake_config_test(rapids-cpm.cmake)

# Do to https://github.com/rapidsai/rapids-cmake/issues/680 we need to suppress dev warning as
# errors for all the rapids_cpm tests
set(RAPIDS_TEST_DISABLE_DEV_ERRORS ON)

add_cmake_config_test(cpm_find-add-pkg-source)
add_cmake_config_test(cpm_find-and-find_package)
add_cmake_config_test(cpm_find-build-patch)
add_cmake_config_test(cpm_find-components)
add_cmake_config_test(cpm_find-existing-build-dir)
add_cmake_config_test(cpm_find-existing-target)
add_cmake_config_test(cpm_find-existing-target-to-export-sets)
add_cmake_config_test(cpm_find-gtest-no-gmock)
add_cmake_config_test(cpm_find-options-escaped)
add_cmake_config_test(cpm_find-patch-command NO_CPM_CACHE)
add_cmake_config_test(cpm_find-patch-command-embedded NO_CPM_CACHE)
add_cmake_config_test(cpm_find-patch-command-required NO_CPM_CACHE)
add_cmake_config_test(cpm_find-patch-command-required-fails NO_CPM_CACHE SHOULD_FAIL
                      "rapids-cmake [GTest]: failed to apply patch")
add_cmake_config_test(cpm_find-restore-cpm-vars)
add_cmake_config_test(cpm_find-version-explicit-install.cmake)

add_cmake_build_test(cpm_generate_pins-format-patches NO_CPM_CACHE)
add_cmake_build_test(cpm_generate_pins-nested)
add_cmake_build_test(cpm_generate_pins-no-src-dir)
add_cmake_build_test(cpm_generate_pins-custom-src-dir)
add_cmake_build_test(cpm_generate_pins-override NO_CPM_CACHE)
add_cmake_build_test(cpm_generate_pins-pure-cpm)
add_cmake_build_test(cpm_generate_pins-simple)
add_cmake_build_test(cpm_generate_pins-simple-via-variable)
add_cmake_build_test(cpm_generate_pins-var-and-arg)
add_cmake_build_test(cpm_generate_pins-url-mode)
add_cmake_config_test(cpm_convert_patch_json-special-chars.cmake)

add_cmake_config_test(cpm_init-bad-default-path.cmake SHOULD_FAIL "rapids_cpm_init can't load")
add_cmake_config_test(cpm_init-bad-default-cmake-var.cmake SHOULD_FAIL "rapids_cpm_init can't load")
add_cmake_config_test(cpm_init-bad-override-path.cmake SHOULD_FAIL
                      "rapids_cpm_package_override can't load")
add_cmake_config_test(cpm_init-bad-override-cmake-var.cmake SHOULD_FAIL
                      "rapids_cpm_package_override can't load")
add_cmake_config_test(cpm_init-custom-default-simple.cmake)
add_cmake_config_test(cpm_init-custom-default-multiple.cmake)
add_cmake_config_test(cpm_init-custom-default-via-cmake-var.cmake)
add_cmake_config_test(cpm_init-override-multiple.cmake)
add_cmake_config_test(cpm_init-override-simple.cmake)
add_cmake_config_test(cpm_init-override-via-cmake-var.cmake)

add_cmake_config_test(cpm_package_override-add-new-project.cmake)
add_cmake_config_test(cpm_package_override-bad-path.cmake SHOULD_FAIL
                      "rapids_cpm_package_override can't load")
add_cmake_config_test(cpm_package_override-before-init.cmake)
add_cmake_config_test(cpm_package_override-defaults-with-different-casing.cmake)
add_cmake_config_test(cpm_package_override-defaults-with-different-casing-warning.cmake)
add_cmake_config_test(cpm_package_override-empty-patches.cmake)
add_cmake_config_test(cpm_package_override-empty.cmake)
add_cmake_config_test(cpm_package_override-env-var-support.cmake)
add_cmake_config_test(cpm_package_override-multiple-cmake-var.cmake)
add_cmake_config_test(cpm_package_override-multiple.cmake)
add_cmake_config_test(cpm_package_override-no-version-value.cmake SHOULD_FAIL
                      "rapids_cmake can't parse")
add_cmake_config_test(cpm_package_override-obey-cpm-source-var.cmake)
add_cmake_config_test(cpm_package_override-patches.cmake)
add_cmake_config_test(cpm_package_override-source-subdir.cmake)
add_cmake_config_test(cpm_package_override-simple.cmake)
add_cmake_config_test(cpm_package_override-url-mode.cmake)
add_cmake_config_test(cpm_package_override-url-without-hash.cmake SHOULD_FAIL
                      "has 'url' but is missing 'url_hash'")
add_cmake_config_test(cpm_package_override-hash-without-url.cmake SHOULD_FAIL
                      "has 'url_hash' but is missing 'url'")
add_cmake_config_test(cpm_package_override-url-to-git.cmake)
add_cmake_config_test(cpm_package_override-git-to-url.cmake)

add_cmake_config_test(cpm_package_info-url-mode.cmake)

add_cmake_config_test(cpm_generate_patch_command-invalid.cmake)
add_cmake_config_test(cpm_generate_patch_command-override-case-insensitive.cmake)
add_cmake_config_test(cpm_generate_patch_command-override.cmake)
add_cmake_config_test(cpm_generate_patch_command-current_json_dir.cmake)
add_cmake_config_test(cpm_generate_patch_command-verify-copyright-header.cmake)
add_cmake_config_test(cpm_generate_patch_command-build.cmake)

add_cmake_config_test(cpm_bs_thread_pool-simple.cmake)
add_cmake_config_test(cpm_bs_thread_pool-export.cmake)
add_cmake_build_test(cpm_bs_thread_pool-build-config-works.cmake)
add_cmake_build_test(cpm_bs_thread_pool-install-config-works.cmake)

add_cmake_config_test(cpm_cccl-simple.cmake)
add_cmake_config_test(cpm_cccl-enable-unstable.cmake)
add_cmake_config_test(cpm_cccl-export.cmake)
add_cmake_build_test(cpm_cccl-version-2-8.cmake NO_CPM_CACHE)
add_cmake_build_test(cpm_cccl-version-3-0.cmake NO_CPM_CACHE)
add_cmake_build_test(cpm_cccl-preserve-custom-install-loc)

add_cmake_config_test(cpm_cuco-simple.cmake)
add_cmake_config_test(cpm_cuco-export.cmake)

add_cmake_config_test(cpm_gbench-export.cmake)
add_cmake_config_test(cpm_gbench-simple.cmake)
add_cmake_config_test(cpm_gbench-explicit-static.cmake)

add_cmake_config_test(cpm_gtest-export.cmake)
add_cmake_config_test(cpm_gtest-simple.cmake)
add_cmake_config_test(cpm_gtest-static.cmake)
add_cmake_config_test(cpm_gtest-explicit-static.cmake)

add_cmake_config_test(cpm_logger-simple.cmake)

add_cmake_config_test(cpm_nvbench-export.cmake SERIAL)
add_cmake_config_test(cpm_nvbench-simple.cmake SERIAL)
add_cmake_config_test(cpm_nvbench-already-found-fmt.cmake SERIAL)
add_cmake_build_test(cpm_nvbench-conda.cmake SERIAL)
add_cmake_config_test(cpm_nvbench-explicit-static.cmake SERIAL)

add_cmake_config_test(cpm_nvcomp-export.cmake)
add_cmake_config_test(cpm_nvcomp-proprietary-off.cmake)
add_cmake_config_test(cpm_nvcomp-proprietary-on.cmake)
add_cmake_config_test(cpm_nvcomp-simple.cmake)
add_cmake_config_test(cpm_nvcomp-invalid-arch.cmake NO_CPM_CACHE)
add_cmake_config_test(cpm_nvcomp-always-download-proprietary_binary.cmake SERIAL)
add_cmake_config_test(cpm_nvcomp-override-clears-proprietary_binary.cmake SERIAL)
add_cmake_build_test(cpm_nvcomp-proprietary_binary-lib-location.cmake SERIAL NO_CPM_CACHE)
add_cmake_build_test(cpm_nvcomp-proprietary_binary-lib-location-nested.cmake SERIAL NO_CPM_CACHE)

add_cmake_build_test(cpm_nvtx-build-config-works.cmake)
add_cmake_build_test(cpm_nvtx-install-config-works.cmake)
add_cmake_config_test(cpm_nvtx-export.cmake)
add_cmake_config_test(cpm_nvtx-simple.cmake)
add_cmake_build_test(cpm_nvtx-build-config-subdir)
add_cmake_build_test(cpm_nvtx-install-config-subdir)

add_cmake_config_test(cpm_proprietary-url-ctk-version-find-ctk.cmake)
add_cmake_config_test(cpm_proprietary-url-ctk-version.cmake)
add_cmake_config_test(cpm_proprietary-url-no-ctk-parsing.cmake)

add_cmake_config_test(cpm_rmm-export.cmake)
add_cmake_config_test(cpm_rmm-simple.cmake)
