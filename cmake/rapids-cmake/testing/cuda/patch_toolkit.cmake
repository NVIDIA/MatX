#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
include(${rapids-cmake-dir}/cuda/patch_toolkit.cmake)

function(verify_links_to target library)
  get_target_property(link_libs ${target} INTERFACE_LINK_LIBRARIES)
  if(NOT ${library} IN_LIST link_libs)
    message(FATAL_ERROR "${target} doesn't link to ${library}")
  endif()
endfunction()

# Verify we can call before find_package
rapids_cuda_patch_toolkit()

find_package(CUDAToolkit)
rapids_cuda_patch_toolkit()

if(TARGET CUDA::cublas_static)
  verify_links_to(CUDA::cublas CUDA::cublasLt)
  verify_links_to(CUDA::cublas_static CUDA::cublasLt_static)
  verify_links_to(CUDA::cusolver_static CUDA::cusolver_lapack_static)
endif()
