# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)

enable_language(CUDA)

set(stub_file "${CMAKE_CURRENT_BINARY_DIR}/stub.cu")
file(WRITE "${stub_file}" [=[int stub(){return 42;}]=])

set(name_suffix b s r)
set(modes balanced size runtime_perf)
set(compile_options balance size speed)

foreach(suffix mode IN ZIP_LISTS name_suffix modes)
  rapids_cuda_enable_fatbin_compression(TARGET not_yet_${suffix} TUNE_FOR ${mode})

  add_library(exists_${suffix} SHARED ${stub_file})
  rapids_cuda_enable_fatbin_compression(TARGET exists_${suffix} TUNE_FOR ${mode})
endforeach()

# Validate all the targets now have the proper values
foreach(suffix mode option IN ZIP_LISTS name_suffix modes compile_options)
  set(properties INTERFACE_COMPILE_OPTIONS COMPILE_OPTIONS)
  set(targets not_yet_${suffix} exists_${suffix})
  foreach(target property IN ZIP_LISTS targets properties)
    get_target_property(compile_opts ${target} ${property})
    if(NOT compile_opts MATCHES "-Xfatbin=-compress-all")
      message(FATAL_ERROR "${target} missing the compress-all compile flag")
    endif()
    if(NOT compile_opts MATCHES "--compress-mode=${option}")
      message(FATAL_ERROR "${target} missing the proper compress-mode flag of ${option} instead has ${compile_opts}"
      )
    endif()
  endforeach()
endforeach()

# Handle checking all the tune types that map to `rapids`
add_library(exists SHARED ${stub_file})
add_library(exists_rapids SHARED ${stub_file})

set(properties INTERFACE_COMPILE_OPTIONS COMPILE_OPTIONS)
set(targets not_yet exists)
foreach(target property IN ZIP_LISTS targets properties)
  rapids_cuda_enable_fatbin_compression(TARGET ${target})
  rapids_cuda_enable_fatbin_compression(TARGET ${target}_rapids TUNE_FOR rapids)

  get_target_property(compile_opts_a ${target} ${property})
  get_target_property(compile_opts_b ${target}_rapids ${property})
  if(NOT compile_opts_a STREQUAL compile_opts_b)
    message(FATAL_ERROR "rapids_cuda_enable_fatbin_compression without any TUNE_FOR should match 'rapids'"
    )
  endif()
  if(NOT compile_opts_a MATCHES "-Xfatbin=-compress-all")
    message(FATAL_ERROR "${target} missing the compress-all compile flag")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0 AND NOT compile_opts_a MATCHES
                                                                  "--compress-mode")
    message(FATAL_ERROR "${target} missing the compress-mode compile flag")
  endif()
endforeach()
