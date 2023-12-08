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
rapids_cuda_architectures_policy
--------------------------------

.. versionadded:: v23.02.00

Maps deprecated mode values to new supported values and outputs rapids-cmake
deprecation warnings.

.. versionchanged:: v23.06.00
Now errors on deprecated mode values and outputs guidance on how to upgrade

.. code-block:: cmake

  rapids_cuda_architectures_policy( (FROM_INIT|FROM_SET) mode_variable )

#]=======================================================================]
function(rapids_cuda_architectures_policy called_from mode_variable)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.architectures_policy")

  include("${rapids-cmake-dir}/cmake/detail/policy.cmake")

  set(value ${${mode_variable}})
  set(new_value ${value})
  if(value STREQUAL "ALL")
    set(new_value "RAPIDS")
    if(called_from STREQUAL "FROM_INIT")
      rapids_cmake_policy(DEPRECATED_IN 23.02
                          REMOVED_IN 23.06
                          MESSAGE [=[Usage of `ALL` as value for `CMAKE_CUDA_ARCHITECTURES` or the env variable `CUDAARCHS` has been deprecated, use `RAPIDS` instead.]=]
      )
    elseif(called_from STREQUAL "FROM_SET")
      rapids_cmake_policy(DEPRECATED_IN 23.02
                          REMOVED_IN 23.06
                          MESSAGE [=[Usage of `ALL` as value passed to `rapids_cuda_set_architectures` has been deprecated, use `RAPIDS` instead.]=]
      )
    endif()
  endif()
  if(value STREQUAL "EMPTY_STR")
    set(new_value "NATIVE")
    rapids_cmake_policy(DEPRECATED_IN 23.02
                        REMOVED_IN 23.06
                        MESSAGE [=[Usage of `""` as value for `CMAKE_CUDA_ARCHITECTURES` has been deprecated, use `NATIVE` instead.]=]
    )
  endif()

  set(${mode_variable} ${new_value} PARENT_SCOPE)
endfunction()
