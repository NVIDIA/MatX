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
include(${rapids-cmake-dir}/cpm/cccl.cmake)

rapids_cpm_init()

if(TARGET CCCL::CCCL)
  message(FATAL_ERROR "Expected CCCL::CUB not to exist")
endif()
if(TARGET CCCL::CUB)
  message(FATAL_ERROR "Expected CCCL::CUB not to exist")
endif()
if(TARGET CCCL::libcudacxx)
  message(FATAL_ERROR "Expected CCCL::libcudacxx not to exist")
endif()
if(TARGET CCCL::Thrust)
  message(FATAL_ERROR "Expected CCCL::Thrust not to exist")
endif()
if(TARGET libcudacxx::libcudacxx)
  message(FATAL_ERROR "Expected libcudacxx::libcudacxx not to exist")
endif()

rapids_cpm_cccl()
if(NOT TARGET CCCL::CCCL)
  message(FATAL_ERROR "Expected CCCL::CUB target to exist")
endif()
if(NOT TARGET CCCL::CUB)
  message(FATAL_ERROR "Expected CCCL::CUB target to exist")
endif()
if(NOT TARGET CCCL::libcudacxx)
  message(FATAL_ERROR "Expected CCCL::libcudacxx target to exist")
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24.0 AND NOT TARGET CCCL::Thrust)
  message(FATAL_ERROR "Expected CCCL::Thrust target to exist")
endif()
if(NOT TARGET libcudacxx::libcudacxx)
  message(FATAL_ERROR "Expected libcudacxx::libcudacxx target to exist")
endif()

rapids_cpm_cccl()
