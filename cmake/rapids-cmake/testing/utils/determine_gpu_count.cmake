# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

function(determine_gpu_count)
  # run nvidia-smi and extract gpu details
  execute_process(COMMAND nvidia-smi --list-gpus OUTPUT_VARIABLE smi_output)

  string(REPLACE "\n" ";" smi_output "${smi_output}")
  list(POP_BACK smi_output) # remove the trailing `;` entry
  list(LENGTH smi_output gpu_count)
  set(RAPIDS_CMAKE_TESTING_GPU_COUNT ${gpu_count} PARENT_SCOPE)
endfunction()
