#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
rapids_cuda_set_runtime
-------------------------------

.. versionadded:: v23.08.00

Establish what CUDA runtime library should be used by a single target

  .. code-block:: cmake

    rapids_cuda_set_runtime( target USE_STATIC (TRUE|FALSE) )

  Establishes what CUDA runtime will be used for a target, via
  the :cmake:prop_tgt:`CUDA_RUNTIME_LIBRARY <cmake:prop_tgt:CUDA_RUNTIME_LIBRARY>`
  and by linking to `CUDA::cudart` or `CUDA::cudart_static` if the :cmake:module:`find_package(CUDAToolkit)
  <cmake:module:FindCUDAToolkit>` has been called.

  The linking to the `CUDA::cudart` or `CUDA::cudart_static` will have the following
  usage behavior:

    - For `INTERFACE` targets the linking will be `INTERFACE`
    - For all other targets the linking will be `PRIVATE`

 .. note::
  If using the deprecated `FindCUDA.cmake` you must use the
  :cmake:command:`rapids_cuda_init_runtime` method to properly establish the default
  mode.

  When `USE_STATIC TRUE` is provided the target will link to a
    statically-linked CUDA runtime library.

  When `USE_STATIC FALSE` is provided the target will link to a
    shared-linked CUDA runtime library.


#]=======================================================================]
function(rapids_cuda_set_runtime target use_static value)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.set_runtime")

  get_target_property(type ${target} TYPE)
  if(type STREQUAL "INTERFACE_LIBRARY")
    set(mode INTERFACE)
  else()
    set(mode PRIVATE)
  endif()

  if(${value})
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY STATIC)
    target_link_libraries(${target} ${mode} $<TARGET_NAME_IF_EXISTS:CUDA::cudart_static>)
  else()
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY SHARED)
    target_link_libraries(${target} ${mode} $<TARGET_NAME_IF_EXISTS:CUDA::cudart>)
  endif()

endfunction()
