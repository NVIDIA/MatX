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
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/spdlog.cmake)

enable_language(CXX)

rapids_cpm_init()
rapids_cpm_spdlog(FMT_OPTION "EXTERNAL_FMT_HO")


file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/use_external_fmt.cpp" [=[

#ifndef SPDLOG_FMT_EXTERNAL
#error "SPDLOG_FMT_EXTERNAL not defined"
#endif

]=])

add_library(spdlog_extern_fmt SHARED "${CMAKE_CURRENT_BINARY_DIR}/use_external_fmt.cpp")
target_link_libraries(spdlog_extern_fmt PRIVATE spdlog::spdlog)

add_library(spdlog-header-only_extern_fmt SHARED "${CMAKE_CURRENT_BINARY_DIR}/use_external_fmt.cpp")
target_link_libraries(spdlog-header-only_extern_fmt PRIVATE spdlog::spdlog_header_only)
