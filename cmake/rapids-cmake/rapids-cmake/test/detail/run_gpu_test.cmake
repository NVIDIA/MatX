# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 3.30.4)

if(DEFINED ENV{CTEST_RESOURCE_GROUP_COUNT})
  math(EXPR max_index "$ENV{CTEST_RESOURCE_GROUP_COUNT}-1")
  # cmake-lint: disable=E1120
  foreach(index RANGE 0 "${max_index}")
    set(allocation $ENV{CTEST_RESOURCE_GROUP_${index}_GPUS})
    if(DEFINED allocation)
      # strings look like "id:value,slots:value" so let's make a super lazy parser by deleting `id:`
      # and replacing `,slots:` with `;` so we have a list with two items.
      string(REPLACE "id:" "" allocation "${allocation}")
      string(REPLACE ",slots:" ";" allocation "${allocation}")
      list(GET allocation 0 device_ids)
      # slots are the cmake test requirements term for what we call percent, so we can ignore the
      # second item in the list
      set(ENV{CUDA_VISIBLE_DEVICES} ${device_ids})
    endif()
  endforeach()
endif()
execute_process(COMMAND ${command_to_run} ${command_args} COMMAND_ECHO STDOUT
                        COMMAND_ERROR_IS_FATAL ANY)
