#=============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

function(determine_gpu_count )
  #run nvidia-smi and extract gpu details
  execute_process(COMMAND nvidia-smi --list-gpus OUTPUT_VARIABLE smi_output)

  string(REPLACE "\n" ";" smi_output "${smi_output}")
  list(POP_BACK smi_output) #remove the trailing `;` entry
  list(LENGTH smi_output gpu_count)
  set(RAPIDS_CMAKE_TESTING_GPU_COUNT ${gpu_count} PARENT_SCOPE)
endfunction()
