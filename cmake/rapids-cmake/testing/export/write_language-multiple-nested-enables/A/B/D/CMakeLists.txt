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

include(${rapids-cmake-dir}/export/write_language.cmake)

function(enable_nested_language file_path)
  #replicates CPM including the file
  include("${file_path}")
endfunction()

set(path "${CMAKE_CURRENT_BINARY_DIR}/enable_cuda_language.cmake")
rapids_export_write_language(BUILD CUDA "${path}")
enable_nested_language("${path}")

add_library(D STATIC static.cu)
