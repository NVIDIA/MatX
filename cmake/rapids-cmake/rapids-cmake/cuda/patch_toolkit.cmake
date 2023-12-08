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
include_guard(GLOBAL)

#[=======================================================================[.rst:
rapids_cuda_patch_toolkit
---------------------------------

.. versionadded:: v22.10.00

Corrects missing dependencies in the CUDA toolkit

  .. code-block:: cmake

    rapids_cuda_patch_toolkit( )

For CMake versions 3.23.1-3, and 3.24.1 the dependencies
of cublas and cusolver targets are incorrect. This module must be called
from the same CMakeLists.txt as the first `find_project(CUDAToolkit)` to
patch the targets.

.. note::
  :cmake:command:`rapids_cpm_find` will automatically call this module
  when asked to find the CUDAToolkit.

#]=======================================================================]
function(rapids_cuda_patch_toolkit)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.patch_toolkit")

  get_directory_property(itargets IMPORTED_TARGETS)
  if(CMAKE_VERSION VERSION_LESS 3.24.2)
    if(CUDA::cublas IN_LIST itargets)
      target_link_libraries(CUDA::cublas INTERFACE CUDA::cublasLt)
    endif()

    if(CUDA::cublas_static IN_LIST itargets)
      target_link_libraries(CUDA::cublas_static INTERFACE CUDA::cublasLt_static)
    endif()

    if(CUDA::cusolver_static IN_LIST itargets)
      target_link_libraries(CUDA::cusolver_static INTERFACE CUDA::cusolver_lapack_static)
    endif()
  endif()
endfunction()
