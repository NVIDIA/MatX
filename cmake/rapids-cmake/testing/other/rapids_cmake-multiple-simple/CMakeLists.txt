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
cmake_minimum_required(VERSION 3.23.1)
project(rapids-test-project LANGUAGES CXX)

cmake_path(GET rapids-cmake-dir PARENT_PATH rapids.cmake-location)
cmake_path(APPEND rapids.cmake-location "RAPIDS.cmake")

# Remove this cache variable so we can properly test
# fetching rapids-cmake work as intended
unset(rapids-cmake-dir CACHE)

# Remove this cache variable so we can properly test
# fetching rapids-cmake work as intended
unset(rapids-cmake-dir CACHE)

add_subdirectory(A)
add_subdirectory(B)
