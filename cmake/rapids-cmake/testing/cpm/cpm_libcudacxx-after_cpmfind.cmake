#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)

rapids_cpm_init()
include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
rapids_cpm_package_details(libcudacxx version repository tag shallow exclude)

include("${rapids-cmake-dir}/cpm/find.cmake")
rapids_cpm_find(libcudacxx ${version}
                CPM_ARGS
                GIT_REPOSITORY ${repository}
                GIT_TAG ${tag}
                GIT_SHALLOW ${shallow}
                EXCLUDE_FROM_ALL ${exclude}
                DOWNLOAD_ONLY TRUE)

                
rapids_cpm_libcudacxx()
if(NOT TARGET libcudacxx::libcudacxx)
  message(FATAL_ERROR "Expected libcudacxx::libcudacxx target to exist")
endif()

rapids_cpm_libcudacxx()
