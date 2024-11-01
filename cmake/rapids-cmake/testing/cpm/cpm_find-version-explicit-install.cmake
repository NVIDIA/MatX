#=============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/cccl.cmake)

rapids_cpm_init()

rapids_cpm_cccl(INSTALL_EXPORT_SET example_export)

include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details(cccl cccl_version repository tag shallow exclude)

set(cccl_path "${CMAKE_BINARY_DIR}/rapids-cmake/example_export/install/package_CCCL.cmake")

file(READ "${cccl_path}" contents)
message(STATUS "contents: ${contents}")
string(FIND "${contents}" "${cccl_version}" is_found)
if(is_found EQUAL -1)
  message(FATAL_ERROR "rapids_cpm_cccl failed to generate a find_package configuration with version")
endif()
