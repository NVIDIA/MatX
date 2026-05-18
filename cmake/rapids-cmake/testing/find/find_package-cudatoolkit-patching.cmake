# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/find/package.cmake)

function(verify_links_to target library)
  get_target_property(link_libs ${target} INTERFACE_LINK_LIBRARIES)
  if(NOT ${library} IN_LIST link_libs)
    message(FATAL_ERROR "${target} doesn't link to ${library}")
  endif()
endfunction()

rapids_find_package(CUDAToolkit)

if(TARGET CUDA::cublas_static)
  verify_links_to(CUDA::cublas CUDA::cublasLt)
  verify_links_to(CUDA::cublas_static CUDA::cublasLt_static)
  verify_links_to(CUDA::cusolver_static CUDA::cusolver_lapack_static)
endif()
