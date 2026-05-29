# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

# Make sure we always have CMake 3.23 policies when executing this file since we can be executing in
# directories of users of rapids-cmake which have a lower minimum cmake version and therefore
# different policies
#
cmake_policy(PUSH)
cmake_policy(VERSION 3.23)

# Include the needed functions that write out the the pinned versions file
include("${rapids-cmake-dir}/cpm/detail/pinning_write_file.cmake")

# Compute and write out the pinned versions file
rapids_cpm_pinning_write_file()

cmake_policy(POP)
