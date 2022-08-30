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
include(${rapids-cmake-dir}/cmake/install_lib_dir.cmake)

unset(ENV{CONDA_BUILD})
unset(ENV{CONDA_PREFIX})

rapids_cmake_install_lib_dir( lib_dir )
if(DEFINED CMAKE_INSTALL_LIBDIR)
  message(FATAL_ERROR "rapids_cmake_install_lib_dir shouldn't have caused the CMAKE_INSTALL_LIBDIR variable to exist")
endif()

include(GNUInstallDirs)
if(NOT lib_dir STREQUAL CMAKE_INSTALL_LIBDIR)
  message(FATAL_ERROR "rapids_cmake_install_lib_dir computed '${lib_dir}', but we expected '${CMAKE_INSTALL_LIBDIR}' as it should match GNUInstallDirs")
endif()

# unset CMAKE_INSTALL_LIBDIR so it doesn't leak into our CMakeCache.txt and cause subsequent
# re-runs of the test to fail
unset(CMAKE_INSTALL_LIBDIR)
unset(CMAKE_INSTALL_LIBDIR CACHE)
