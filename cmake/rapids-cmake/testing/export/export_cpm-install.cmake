# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/export/cpm.cmake)

rapids_export_cpm(INSTALL RaFT fake_set CPM_ARGS FAKE_PACKAGE_ARGS TRUE)

rapids_export_cpm(install RMM fake_set CPM_ARGS VERSION 2.0 FAKE_PACKAGE_ARGS FALSE
                  GLOBAL_TARGETS RMM::RMM_POOL)

if(NOT TARGET rapids_export_install_fake_set)
  message(FATAL_ERROR "rapids_export_cpm failed to generate target for install")
endif()

# Verify that we encoded both packages for exporting
get_target_property(packages rapids_export_install_fake_set PACKAGE_NAMES)
if(NOT RaFT IN_LIST packages)
  message(FATAL_ERROR "rapids_export_cpm failed to record RaFT needs to be exported")
endif()
if(NOT RMM IN_LIST packages)
  message(FATAL_ERROR "rapids_export_cpm failed to record RMM needs to be exported")
endif()

# Verify that we encoded what `targets` are marked as global export
get_target_property(global_targets rapids_export_install_fake_set GLOBAL_TARGETS)
if(NOT "RMM::RMM_POOL" IN_LIST global_targets)
  message(FATAL_ERROR "rapids_export_cpm failed to record RMM::RMM_POOL needs to be global")
endif()

# Verify that CPM property is set
get_target_property(requires_cpm rapids_export_install_fake_set REQUIRES_CPM)
if(NOT requires_cpm)
  message(FATAL_ERROR "rapids_export_cpm failed to record that CPM is required by the export set")
endif()

# Verify that cpm configuration files exist
if(NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/fake_set/install/cpm_RaFT.cmake"
   OR NOT EXISTS "${CMAKE_BINARY_DIR}/rapids-cmake/fake_set/install/cpm_RMM.cmake")
  message(FATAL_ERROR "rapids_export_cpm failed to generate a CPM configuration")
endif()
