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
include(${rapids-cmake-dir}/cmake/support_conda_env.cmake)


set(ENV{CONDA_BUILD} "1")
set(ENV{BUILD_PREFIX} "/usr/local/build_prefix")
set(ENV{target_platform} "linux-64")
set(ENV{PREFIX} "/opt/local/prefix")
set(ENV{CONDA_PREFIX} "/opt/conda/prefix")

rapids_cmake_support_conda_env(conda_env)
if(NOT TARGET conda_env)
  message(FATAL_ERROR "Expected target conda_env to exist")
endif()

get_target_property(include_dirs conda_env INTERFACE_INCLUDE_DIRECTORIES)
if( NOT "$ENV{BUILD_PREFIX}/include" IN_LIST include_dirs)
  message(FATAL_ERROR "Expected env{BUILD_PREFIX} to be in the include dirs of `conda_env`")
endif()
if( NOT "$ENV{PREFIX}/include" IN_LIST include_dirs)
  message(FATAL_ERROR "Expected env{PREFIX} to be in the include dirs of `conda_env`")
endif()
if("$ENV{CONDA_PREFIX}/include" IN_LIST include_dirs)
  message(FATAL_ERROR "Not expected for env{CONDA_PREFIX} to be in the include dirs of `conda_env`")
endif()

get_target_property(link_dirs conda_env INTERFACE_LINK_DIRECTORIES)
if( NOT "$ENV{BUILD_PREFIX}/lib" IN_LIST link_dirs)
  message(FATAL_ERROR "Expected env{BUILD_PREFIX} to be in the link dirs of `conda_env`")
endif()
if( NOT "$ENV{PREFIX}/lib" IN_LIST link_dirs)
  message(FATAL_ERROR "Expected env{PREFIX} to be in the link dirs of `conda_env`")
endif()
if("$ENV{CONDA_PREFIX}/lib" IN_LIST link_dirs)
  message(FATAL_ERROR "Not expected for env{CONDA_PREFIX} to be in the link dirs of `conda_env`")
endif()

get_target_property(link_options conda_env INTERFACE_LINK_OPTIONS)
message(STATUS "link_options: ${link_options}")
if( NOT "$<HOST_LINK:SHELL:LINKER:-rpath-link=$ENV{BUILD_PREFIX}/lib>" IN_LIST link_options)
  message(FATAL_ERROR "Expected rpath-link=env{BUILD_PREFIX} to be in the link options of `conda_env`")
endif()
if( NOT "$<HOST_LINK:SHELL:LINKER:-rpath-link=$ENV{PREFIX}/lib>" IN_LIST link_options)
  message(FATAL_ERROR "Expected rpath-link=env{PREFIX} to be in the link options of `conda_env`")
endif()
if( NOT "$<HOST_LINK:SHELL:LINKER:-rpath-link=$ENV{PREFIX}/targets/x86_64-linux/lib>" IN_LIST link_options)
  message(FATAL_ERROR "Expected rpath-link=env{PREFIX}/targets/x86_64-linux/ to be in the link options of `conda_env`")
endif()
if("$<HOST_LINK:SHELL:LINKER:-rpath-link=$ENV{CONDA_PREFIX}/lib>" IN_LIST link_options)
  message(FATAL_ERROR "Not expected for rpath-link=env{CONDA_PREFIX} to be in the link options of `conda_env`")
endif()

# No effect as the target already exists
set(before_call_value "${CMAKE_PREFIX_PATH}" )
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
if(NOT ("${before_call_value}" STREQUAL "${CMAKE_PREFIX_PATH}") )
  message(FATAL_ERROR "Expected rapids_cmake_support_conda_env not to change CMAKE_PREFIX_PATH")
endif()

# New target being used, so this should modify CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "placeholder" )
set(ENV{CMAKE_PREFIX_PATH} "env_1:env_2" )
rapids_cmake_support_conda_env(conda_env_modify MODIFY_PREFIX_PATH)
if(NOT TARGET conda_env_modify)
  message(FATAL_ERROR "Expected target conda_env_modify to exist")
endif()

list(LENGTH CMAKE_PREFIX_PATH len)
if( len GREATER 6)
  message(FATAL_ERROR "CMAKE_PREFIX_PATH length is wrong after MODIFY_PREFIX_PATH")
endif()

list(GET CMAKE_PREFIX_PATH 0 first_value)
list(GET CMAKE_PREFIX_PATH 1 second_value)
list(GET CMAKE_PREFIX_PATH 2 third_value)
list(GET CMAKE_PREFIX_PATH 3 fourth_value)
list(GET CMAKE_PREFIX_PATH 4 fifth_value)
list(GET CMAKE_PREFIX_PATH 5 sixth_value)
set(correct_list "placeholder" "env_1" "env_2" "$ENV{PREFIX}/targets/x86_64-linux" "$ENV{PREFIX}" "$ENV{BUILD_PREFIX}")
set(actual_list "${first_value}" "${second_value}" "${third_value}" "${fourth_value}" "${fifth_value}" "${sixth_value}")

foreach(correct actual IN ZIP_LISTS correct_list actual_list)
  if(NOT correct STREQUAL actual)
    message(STATUS "correct: ${correct}")
    message(STATUS "actual: ${actual}")
    message(FATAL_ERROR "MODIFY_PREFIX_PATH failed")
  endif()
endforeach()
