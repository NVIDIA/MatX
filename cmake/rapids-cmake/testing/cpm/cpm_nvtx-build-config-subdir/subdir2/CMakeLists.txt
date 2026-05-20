# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

rapids_cpm_nvtx3(BUILD_EXPORT_SET uses-nvtx3-exports)

add_library(uses-nvtx3-cpp INTERFACE)
target_link_libraries(uses-nvtx3-cpp INTERFACE nvtx3::nvtx3-cpp)
install(TARGETS uses-nvtx3-cpp EXPORT uses-nvtx3-exports)

rapids_export(BUILD uses-nvtx3 EXPORT_SET uses-nvtx3-exports GLOBAL_TARGETS uses-nvtx3-cpp
              NAMESPACE uses-nvtx3::)
