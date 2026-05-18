# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake)

rapids_cpm_generate_patch_command(not_a_project 1 patch_command build_patch_only)
if(patch_command)
  message(FATAL_ERROR "not_a_project should not have a patch command")
endif()

rapids_cpm_init()
rapids_cpm_generate_patch_command(not_a_project 1 patch_command build_patch_only)
if(patch_command)
  message(FATAL_ERROR "not_a_project should not have a patch command")
endif()
