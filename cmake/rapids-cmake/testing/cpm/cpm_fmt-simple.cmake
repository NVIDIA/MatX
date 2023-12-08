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
include(${rapids-cmake-dir}/cpm/fmt.cmake)

rapids_cpm_init()

if(TARGET fmt::fmt-header-only)
  message(FATAL_ERROR "Expected fmt::fmt-header-only expected to not exist")
endif()

if(TARGET fmt::fmt)
  message(FATAL_ERROR "Expected fmt::fmt expected to not exist")
endif()

rapids_cpm_fmt()

if(NOT TARGET fmt::fmt-header-only)
  message(FATAL_ERROR "Expected fmt::fmt-header-only target to exist")
endif()

if(NOT TARGET fmt::fmt)
  message(FATAL_ERROR "Expected fmt::fmt target to exist")
endif()

rapids_cpm_fmt()
