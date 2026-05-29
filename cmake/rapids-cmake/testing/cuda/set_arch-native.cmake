# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/set_architectures.cmake)

# Required by `NATIVE` as it does compiler detection
enable_language(CUDA)

rapids_cuda_set_architectures(NATIVE)

include("${rapids-cmake-testing-dir}/cuda/validate-cuda-native.cmake")
