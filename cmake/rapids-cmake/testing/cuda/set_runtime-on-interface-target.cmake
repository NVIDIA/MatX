# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/set_runtime.cmake)

add_library(uses_cuda INTERFACE)
rapids_cuda_set_runtime(uses_cuda USE_STATIC TRUE)

get_target_property(linked_libs uses_cuda INTERFACE_LINK_LIBRARIES)
if(NOT "$<TARGET_NAME_IF_EXISTS:CUDA::cudart_static>" IN_LIST linked_libs)
  message(FATAL_ERROR "rapids_cuda_set_runtime shouldn't set CUDA::cudart_static in target linked libraries correctly"
  )
endif()
