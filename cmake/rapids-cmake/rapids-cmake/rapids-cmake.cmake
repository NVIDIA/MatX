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
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/build_type.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/install_lib_dir.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/parse_version.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/support_conda_env.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/write_git_revision_file.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/write_version_file.cmake)
