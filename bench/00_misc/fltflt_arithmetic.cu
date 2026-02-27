////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

// Benchmarks for fltflt (float-float) arithmetic operations.
// Compares performance across three precision modes: float, double, and fltflt.

#include "matx.h"
#include <nvbench/nvbench.cuh>

using namespace matx;

// Precision types to compare
using precision_types = nvbench::type_list<float, double, fltflt>;

//==============================================================================
// Custom kernels that perform operations iteratively to increase arithmetic intensity
//==============================================================================

// Instruction-level parallelism factor
static constexpr int ILP_FACTOR = 8;
// Unroll factor for the inner loop
static constexpr int ITER_UNROLL_FACTOR = 16;

template <typename T>
__global__ void iterative_add_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Initialize multiple independent accumulators in registers
    T acc[ILP_FACTOR] = {};
    const T val = static_cast<T>(0.123456789);

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 0; i < iterations; i++) {
      // Independent operations for ILP - fully unrollable
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        acc[ilp] = acc[ilp] + val;
      }
    }

    // Combine and write to prevent optimization
    T result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename T>
__global__ void iterative_sub_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T acc[ILP_FACTOR] = {};
    const T val = static_cast<T>(0.001234567);

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 0; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        acc[ilp] = acc[ilp] - val;
      }
    }

    T result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename T>
__global__ void iterative_mul_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T acc[ILP_FACTOR];
    const T val = static_cast<T>(1.0000001);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      acc[ilp] = val * val;
    }

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 2; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        acc[ilp] = acc[ilp] * val;
      }
    }

    T result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename T>
__global__ void iterative_div_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T acc[ILP_FACTOR];
    const T val = static_cast<T>(1.0000001);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      acc[ilp] = val / val;
    }

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        acc[ilp] = acc[ilp] / val;
      }
    }

    T result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename T>
__global__ void iterative_sqrt_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T val[ILP_FACTOR];
    const T init_val = static_cast<T>(2.718281828);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      val[ilp] = sqrt(init_val);
    }

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        val[ilp] = sqrt(val[ilp]);
      }
    }

    T result_val = val[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + val[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename T>
__global__ void iterative_abs_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T acc[ILP_FACTOR];
    const T val = static_cast<T>(-0.123456789);
    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      acc[ilp] = val;
    }

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 0; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        acc[ilp] = -abs(acc[ilp]);
      }
    }

    T result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename T>
__global__ void iterative_fma_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T acc[ILP_FACTOR];
    const T val_a = static_cast<T>(1.001);
    const T val_b = static_cast<T>(1.002);
    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      acc[ilp] = val_b;
    }

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 0; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        if constexpr (std::is_same_v<T, fltflt>) {
          acc[ilp] = fltflt_fma(val_a, acc[ilp], val_b);
        } else {
          acc[ilp] = val_a * acc[ilp] + val_b;
        }
      }
    }

    T result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

//==============================================================================
// Addition Benchmark
//==============================================================================
template <typename PrecisionType>
void fltflt_bench_add(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  // Create output tensor only
  auto result = make_tensor<PrecisionType>({size});

  // Add metrics
  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  // Benchmark execution
  state.exec([&](nvbench::launch &launch) {
    iterative_add_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_add, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Subtraction Benchmark
//==============================================================================
template <typename PrecisionType>
void fltflt_bench_sub(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_sub_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_sub, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Multiplication Benchmark
//==============================================================================
template <typename PrecisionType>
void fltflt_bench_mul(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_mul_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_mul, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Division Benchmark
//==============================================================================
template <typename PrecisionType>
void fltflt_bench_div(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_div_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_div, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Square Root Benchmark
//==============================================================================
template <typename PrecisionType>
void fltflt_bench_sqrt(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_sqrt_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_sqrt, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Absolute Value Benchmark
//==============================================================================
template <typename PrecisionType>
void fltflt_bench_abs(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_abs_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_abs, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Fused Multiply-Add Benchmark
//==============================================================================
template <typename PrecisionType>
void fltflt_bench_fma(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_fma_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_fma, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Multiply-Add (MADD) Benchmark - Separate Multiply and Add Operations
//==============================================================================
template <typename T>
__global__ void iterative_madd_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T acc[ILP_FACTOR];
    const T val_a = static_cast<T>(1.001);
    const T val_b = static_cast<T>(1.002);
    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      acc[ilp] = val_b;
    }

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 0; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        if constexpr (std::is_same_v<T, fltflt>) {
          // Explicitly separate multiply and add for fltflt
          acc[ilp] = fltflt_add(fltflt_mul(val_a, acc[ilp]), val_b);
        } else {
          // For float/double, use natural expression (may or may not fuse)
          acc[ilp] = val_a * acc[ilp] + val_b;
        }
      }
    }

    T result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename PrecisionType>
void fltflt_bench_madd(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_madd_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_madd, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Round to Nearest Benchmark
//==============================================================================
template <typename T>
__global__ void iterative_round_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T val[ILP_FACTOR];
    //const T init_val = static_cast<T>(std::numbers::e);
    const T init_val = static_cast<T>(33554432.5);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      if constexpr (std::is_same_v<T, fltflt>) {
        val[ilp] = fltflt_round_to_nearest(init_val);
      } else if constexpr (std::is_same_v<T, float>) {
        val[ilp] = nearbyintf(init_val);
      } else {
        val[ilp] = nearbyint(init_val);
      }
    }

    //#pragma unroll ITER_UNROLL_FACTOR
    #pragma unroll 1
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        if constexpr (std::is_same_v<T, fltflt>) {
          val[ilp] = fltflt_round_to_nearest(val[ilp]);
        } else if constexpr (std::is_same_v<T, float>) {
          val[ilp] = nearbyintf(val[ilp]);
        } else {
          val[ilp] = nearbyint(val[ilp]);
        }
      }
    }

    T result_val = val[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + val[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename PrecisionType>
void fltflt_bench_round(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_round_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_round, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Floating-Point Modulo Benchmark
//==============================================================================
template <typename T>
__global__ void iterative_fmod_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T val[ILP_FACTOR];
    T init_val = static_cast<T>(100000.0 * std::numbers::pi);
    const T divisor = static_cast<T>(std::numbers::e);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      if constexpr (std::is_same_v<T, fltflt>) {
        val[ilp] = fltflt_fmod(init_val, divisor);
      } else if constexpr (std::is_same_v<T, float>) {
        val[ilp] = fmodf(init_val, divisor);
      } else {
        val[ilp] = fmod(init_val, divisor);
      }
    }

    #pragma unroll 1
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        if constexpr (std::is_same_v<T, fltflt>) {
          val[ilp] = val[ilp].hi + fltflt_fmod(init_val, divisor);
          asm volatile("" : "+f"(val[ilp].hi), "+f"(val[ilp].lo));
        } else if constexpr (std::is_same_v<T, float>) {
          val[ilp] = val[ilp] + fmodf(init_val, divisor);
          asm volatile("" : "+f"(val[ilp]));
        } else {
          val[ilp] = val[ilp] + fmod(init_val, divisor);
          asm volatile("" : "+d"(val[ilp]));
        }
      }
      if constexpr (std::is_same_v<T, fltflt>) {
        init_val = init_val + 2048.0f;
      } else {
        init_val += static_cast<T>(2048.0f);
      }
    }

    T result_val = val[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + val[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename PrecisionType>
void fltflt_bench_fmod(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_fmod_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_fmod, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Truncate (Round Toward Zero) Benchmark
//==============================================================================
template <typename T>
__global__ void iterative_trunc_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T val[ILP_FACTOR];
    T init_val = static_cast<T>(6.55557238028172302e+09);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      if constexpr (std::is_same_v<T, fltflt>) {
        val[ilp] = fltflt_round_toward_zero(init_val);
      } else if constexpr (std::is_same_v<T, float>) {
        val[ilp] = truncf(init_val);
      } else {
        val[ilp] = trunc(init_val);
      }
    }

    #pragma unroll 1
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        if constexpr (std::is_same_v<T, fltflt>) {
          val[ilp] = val[ilp].hi + fltflt_round_toward_zero(init_val);
          asm volatile("" : "+f"(val[ilp].hi), "+f"(val[ilp].lo));
        } else if constexpr (std::is_same_v<T, float>) {
          val[ilp] = val[ilp] + truncf(init_val);
          asm volatile("" : "+f"(val[ilp]));
        } else {
          val[ilp] = val[ilp] + trunc(init_val);
          asm volatile("" : "+d"(val[ilp]));
        }
      }
      init_val = init_val + static_cast<T>(2048.0f);
    }

    T result_val = val[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + val[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename PrecisionType>
void fltflt_bench_trunc(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_trunc_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_trunc, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Floor Benchmark
//==============================================================================
template <typename T>
__global__ void iterative_floor_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T val[ILP_FACTOR];
    T init_val = static_cast<T>(6.55557238028172302e+09);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      if constexpr (std::is_same_v<T, fltflt>) {
        val[ilp] = fltflt_floor(init_val);
      } else if constexpr (std::is_same_v<T, float>) {
        val[ilp] = floorf(init_val);
      } else {
        val[ilp] = floor(init_val);
      }
    }

    #pragma unroll 1
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        if constexpr (std::is_same_v<T, fltflt>) {
          val[ilp] = val[ilp].hi + fltflt_floor(init_val);
          asm volatile("" : "+f"(val[ilp].hi), "+f"(val[ilp].lo));
        } else if constexpr (std::is_same_v<T, float>) {
          val[ilp] = val[ilp] + floorf(init_val);
          asm volatile("" : "+f"(val[ilp]));
        } else {
          val[ilp] = val[ilp] + floor(init_val);
          asm volatile("" : "+d"(val[ilp]));
        }
      }
      init_val = init_val + static_cast<T>(2048.0f);
    }

    T result_val = val[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + val[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename PrecisionType>
void fltflt_bench_floor(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_floor_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_floor, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Cast to Double Benchmark
//==============================================================================
template <typename T>
__global__ void iterative_cast2dbl_kernel(double* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    double acc[ILP_FACTOR] = {};
    T src_val = static_cast<T>(1.23456789012345);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      acc[ilp] = static_cast<double>(src_val);
    }

    #pragma unroll 1
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        acc[ilp] = static_cast<double>(src_val);
        asm volatile("" : "+d"(acc[ilp]));
      }
      src_val = src_val + static_cast<T>(0.0001);
    }

    double result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename PrecisionType>
void fltflt_bench_cast2dbl(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<double>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<double>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_cast2dbl_kernel<PrecisionType><<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_cast2dbl, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Cast to fltflt Benchmark
//==============================================================================
template <typename T>
__global__ void iterative_cast2fltflt_kernel(fltflt* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    fltflt acc[ILP_FACTOR] = {};
    T src_val = static_cast<T>(1.23456789012345);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      acc[ilp] = static_cast<fltflt>(src_val);
    }

    #pragma unroll 1
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        acc[ilp] = static_cast<fltflt>(src_val);
        asm volatile("" : "+f"(acc[ilp].hi), "+f"(acc[ilp].lo));
      }
      src_val = src_val + static_cast<T>(0.0001);
    }

    fltflt result_val = acc[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + acc[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename PrecisionType>
void fltflt_bench_cast2fltflt(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<fltflt>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<fltflt>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_cast2fltflt_kernel<PrecisionType><<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_cast2fltflt, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});
