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
include_guard(GLOBAL)

# Make sure we always have CMake 3.23 policies when executing this file since we can be executing in
# directories of users of rapids-cmake which have a lower minimum cmake version and therefore
# different policies
#
cmake_policy(PUSH)
cmake_policy(VERSION 3.23)

# Include the needed functions that write out the the pinned versions file
include("${rapids-cmake-dir}/cpm/detail/pinning_write_file.cmake")

# Compute and write out the pinned versions file
rapids_cpm_pinning_write_file()

cmake_policy(POP)
