# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cmake/support_conda_env.cmake)

set(ENV{CONDA_BUILD} "2")
set(ENV{BUILD_PREFIX} "/usr/local/build_prefix")
set(ENV{PREFIX} "/opt/local/prefix")
set(ENV{CONDA_PREFIX} "/opt/conda/prefix")

rapids_cmake_support_conda_env(conda_env)
if(NOT TARGET conda_env)
  message(FATAL_ERROR "Expected target conda_env to exist")
endif()

get_target_property(compile_options conda_env INTERFACE_COMPILE_OPTIONS)
if(NOT "$<$<CONFIG:Debug>:-O0>" IN_LIST compile_options)
  message(FATAL_ERROR "Expected $<$<CONFIG:Debug>>:-O0> to be in the compile options of `conda_env`"
  )
endif()

get_target_property(include_dirs conda_env INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
if("$ENV{BUILD_PREFIX}/include" IN_LIST include_dirs)
  message(FATAL_ERROR "Not expected env{BUILD_PREFIX} to be in the include dirs of `conda_env`")
endif()
if("$ENV{PREFIX}/include" IN_LIST include_dirs)
  message(FATAL_ERROR "Not expected env{PREFIX} to be in the include dirs of `conda_env`")
endif()
if(NOT "$ENV{CONDA_PREFIX}/include" IN_LIST include_dirs)
  message(FATAL_ERROR "Expected for env{CONDA_PREFIX} to be in the include dirs of `conda_env`")
endif()

get_target_property(link_dirs conda_env INTERFACE_LINK_DIRECTORIES)
if("$ENV{BUILD_PREFIX}/lib" IN_LIST link_dirs)
  message(FATAL_ERROR "Not expected env{BUILD_PREFIX} to be in the link dirs of `conda_env`")
endif()
if("$ENV{PREFIX}/lib" IN_LIST link_dirs)
  message(FATAL_ERROR "Not expected env{PREFIX} to be in the link dirs of `conda_env`")
endif()
if(NOT "$ENV{CONDA_PREFIX}/lib" IN_LIST link_dirs)
  message(FATAL_ERROR "Expected for env{CONDA_PREFIX} to be in the link dirs of `conda_env`")
endif()

get_target_property(link_options conda_env INTERFACE_LINK_OPTIONS)
if("$<HOST_LINK:SHELL:LINKER:-rpath-link=$ENV{BUILD_PREFIX}/lib>" IN_LIST link_options)
  message(FATAL_ERROR "Not expected rpath-link=env{BUILD_PREFIX} to be in the link options of `conda_env`"
  )
endif()
if("$<HOST_LINK:SHELL:LINKER:-rpath-link=$ENV{PREFIX}/lib>" IN_LIST link_options)
  message(FATAL_ERROR "Not expected rpath-link=env{PREFIX} to be in the link options of `conda_env`"
  )
endif()
if(NOT "$<HOST_LINK:SHELL:LINKER:-rpath-link=$ENV{CONDA_PREFIX}/lib>" IN_LIST link_options)
  message(FATAL_ERROR "Expected for rpath-link=env{CONDA_PREFIX} to be in the link options of `conda_env`"
  )
endif()

set(CMAKE_PREFIX_PATH "placeholder")
set(ENV{CMAKE_PREFIX_PATH} "env_1:env_2")
rapids_cmake_support_conda_env(conda_env_modify MODIFY_PREFIX_PATH)
if(NOT TARGET conda_env_modify)
  message(FATAL_ERROR "Expected target conda_env_modify to exist")
endif()

list(LENGTH CMAKE_PREFIX_PATH len)
if(len GREATER 4)
  message(FATAL_ERROR "CMAKE_PREFIX_PATH length is wrong after MODIFY_PREFIX_PATH")
endif()

list(GET CMAKE_PREFIX_PATH 0 first_value)
list(GET CMAKE_PREFIX_PATH 1 second_value)
list(GET CMAKE_PREFIX_PATH 2 third_value)
list(GET CMAKE_PREFIX_PATH 3 fourth_value)
set(correct_list "placeholder" "env_1" "env_2" "$ENV{CONDA_PREFIX}")
set(actual_list "${first_value}" "${second_value}" "${third_value}" "${fourth_value}")

foreach(correct actual IN ZIP_LISTS correct_list actual_list)
  if(NOT correct STREQUAL actual)
    message(FATAL_ERROR "MODIFY_PREFIX_PATH failed")
  endif()
endforeach()
