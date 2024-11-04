#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/bs_thread_pool.cmake)

rapids_cpm_init()

if(TARGET rapids_bs_thread_pool)
  message(FATAL_ERROR "Expected rapids_bs_thread_pool not to exist")
endif()

rapids_cpm_bs_thread_pool()

if(NOT TARGET rapids_bs_thread_pool)
  message(FATAL_ERROR "Expected rapids_bs_thread_pool target to exist")
endif()
if(NOT TARGET BS::thread_pool)
  message(FATAL_ERROR "Expected BS::thread_pool target to exist")
endif()

rapids_cpm_bs_thread_pool()

include(${rapids-cmake-dir}/cpm/generate_pinned_versions.cmake)
rapids_cpm_generate_pinned_versions(OUTPUT ${CMAKE_BINARY_DIR}/versions.json)
