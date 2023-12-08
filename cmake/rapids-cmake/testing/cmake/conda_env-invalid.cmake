#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cmake/support_conda_env.cmake)

unset(ENV{CONDA_BUILD})
unset(ENV{CONDA_PREFIX})
set(ENV{BUILD_PREFIX} "/usr/local/build_prefix")
set(ENV{PREFIX} "/opt/local/prefix")

rapids_cmake_support_conda_env(conda_env)

if(TARGET conda_env)
  message(FATAL_ERROR "Not expected for `conda_env` target to exist")
endif()


set(before_call_value "${CMAKE_PREFIX_PATH}" )
rapids_cmake_support_conda_env(conda_env2 MODIFY_PREFIX_PATH)
if(TARGET conda_env2)
  message(FATAL_ERROR "Not expected for `conda_env2` target to exist")
endif()

if(NOT "${before_call_value}" STREQUAL "${CMAKE_PREFIX_PATH}")
  message(FATAL_ERROR "Not expected for `rapids_cmake_support_conda_env` to modify CMAKE_PREFIX_PATH")
endif()
