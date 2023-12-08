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
cmake_minimum_required(VERSION 3.23.1)

if(DEFINED ENV{CTEST_RESOURCE_GROUP_COUNT})
  # cmake-lint: disable=E1120
  foreach(index RANGE 0 ${CTEST_RESOURCE_GROUP_COUNT})
    set(allocation $ENV{CTEST_RESOURCE_GROUP_${index}_GPUS})
    if(DEFINED allocation)
      # strings look like "id:value,slots:value" so let's make a super lazy parser by deleting `id:`
      # and replacing `,slots:` with `;` so we have a list with two items.
      string(REPLACE "id:" "" allocation "${allocation}")
      string(REPLACE ",slots:" ";" allocation "${allocation}")
      list(GET allocation 0 device_ids)
      # slots are the cmake test requirements term for what we call percent. So we can ignore the
      # second item in the list
      set(ENV{CUDA_VISIBLE_DEVICES} ${device_ids})
    endif()
  endforeach()
endif()
execute_process(COMMAND ${command_to_run} ${command_args} COMMAND_ECHO STDOUT
                        COMMAND_ERROR_IS_FATAL ANY)
