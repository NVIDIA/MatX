#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
include(${rapids-cmake-dir}/cpm/init.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)
include(${rapids-cmake-dir}/cpm/nvbench.cmake)

enable_language(CUDA)
enable_language(CXX)

include(${rapids-cmake-dir}/cuda/set_architectures.cmake)
rapids_cuda_set_architectures(RAPIDS)

# Force shared libs so that nvbench doesn't have a chance to use a static fmt
set(BUILD_SHARED_LIBS ON)
rapids_cpm_init()
rapids_cpm_rmm()
rapids_cpm_nvbench()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/use_fmt.cpp" [=[
#include <spdlog/spdlog.h>
#include <nvbench/nvbench.cuh>

template <typename Type>
void nvbench_distinct(nvbench::state& state, nvbench::type_list<Type>)
{
}

using data_type = nvbench::type_list<bool, int8_t, int32_t, int64_t, float>;

NVBENCH_BENCH_TYPES(nvbench_distinct, NVBENCH_TYPE_AXES(data_type))
  .set_name("distinct")
  .set_type_axes_names({"Type"})
  .add_int64_axis("NumRows", {10'000, 100'000, 1'000'000, 10'000'000});

int main() { return 0; }
]=])


add_library(uses_fmt SHARED "${CMAKE_CURRENT_BINARY_DIR}/use_fmt.cpp")
target_link_libraries(uses_fmt PRIVATE rmm::rmm nvbench::nvbench)
target_compile_features(uses_fmt PRIVATE cxx_std_17)
