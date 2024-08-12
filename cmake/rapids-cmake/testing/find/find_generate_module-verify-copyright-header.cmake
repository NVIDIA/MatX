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
include(${rapids-cmake-dir}/find/generate_module.cmake)
include(${rapids-cmake-testing-dir}/utils/check_copyright_header.cmake)

rapids_find_generate_module( RapidsTest
  HEADER_NAMES rapids-cmake-test-header_only.hpp
  INSTALL_EXPORT_SET test_set
  )

check_copyright_header("${CMAKE_BINARY_DIR}/cmake/find_modules/FindRapidsTest.cmake")
