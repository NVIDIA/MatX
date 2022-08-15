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

# Function uses the CUDA runtime API to query the compute capability of the device, so if a user
# doesn't pass any architecture options to CMake we only build the current architecture
function(rapids_cuda_detect_architectures possible_archs_var gpu_archs)
  list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.cuda.detect_architectures")

  # Unset this first in case it's set to <empty_string> Which can happen inside rapids
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  set(possible_archs ${${possible_archs_var}})

  set(eval_file ${PROJECT_BINARY_DIR}/eval_gpu_archs.cu)
  set(eval_exe ${PROJECT_BINARY_DIR}/eval_gpu_archs)
  set(error_file ${PROJECT_BINARY_DIR}/eval_gpu_archs.stderr.log)
  file(WRITE ${eval_file}
       "
#include <cstdio>
#include <set>
#include <string>
using namespace std;
int main(int argc, char** argv) {
  set<string> archs;
  int nDevices;
  if((cudaGetDeviceCount(&nDevices) == cudaSuccess) && (nDevices > 0)) {
    for(int dev=0;dev<nDevices;++dev) {
      char buff[32];
      cudaDeviceProp prop;
      if(cudaGetDeviceProperties(&prop, dev) != cudaSuccess) continue;
      sprintf(buff, \"%d%d\", prop.major, prop.minor);
      archs.insert(buff);
    }
  }
  if(archs.empty()) {
    printf(\"${possible_archs}\");
  } else {
    bool first = true;
    for(const auto& arch : archs) {
      printf(first? \"%s\" : \";%s\", arch.c_str());
      first = false;
    }
  }
  printf(\"\\n\");
  return 0;
}
")

  set(__gpu_archs "${possible_archs}")
  if(DEFINED CMAKE_CUDA_COMPILER)
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -std=c++11 -o ${eval_exe} --run ${eval_file}
                    OUTPUT_VARIABLE __gpu_archs OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_FILE ${error_file})
    message(STATUS "Auto detection of gpu-archs: ${__gpu_archs}")
  endif()

  set(${gpu_archs} ${__gpu_archs} PARENT_SCOPE)

endfunction()
