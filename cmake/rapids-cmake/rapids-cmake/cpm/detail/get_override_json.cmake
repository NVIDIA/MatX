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

#[=======================================================================[.rst:
get_override_json
--------------------------

. code-block:: cmake

  get_override_json(package_name output_variable)

#]=======================================================================]
function(get_override_json package_name output_variable)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cpm.get_override_json")
  get_property(json_data GLOBAL PROPERTY rapids_cpm_${package_name}_override_json)
  set(${output_variable} "${json_data}" PARENT_SCOPE)
endfunction()
