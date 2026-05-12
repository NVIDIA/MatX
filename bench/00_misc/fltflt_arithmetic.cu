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

// Per-row GOPS/s summary so downstream tooling reads throughput straight off
// the nvbench table/CSV. Each kernel does ILP_FACTOR ops per inner iteration
// per element, with `iterations` outer iterations, on `size` elements:
//   total_ops_per_call = size * iterations * ILP_FACTOR * ops_per_op
// `ops_per_op` lets compound ops (FMA = 2, norm3d = 5, etc.) self-report.
static void add_gops_per_sec_summary(nvbench::state &state, double ops_per_op = 1.0)
{
  const double seconds = state.get_summary("Batch GPU").get_float64("value");
  const int64_t size = state.get_int64("Array Size");
  const int64_t iterations = state.get_int64("Iterations");
  const double total_ops = static_cast<double>(size) * static_cast<double>(iterations)
                         * static_cast<double>(ILP_FACTOR) * ops_per_op;

  auto &s = state.add_summary("matx/fltflt/gops_per_sec");
  s.set_string("name", "Gops/s");
  s.set_string("hint", "item_rate");
  s.set_string("description", "Giga-operations per second (per-row throughput)");
  s.set_float64("value", total_ops / seconds / 1e9);
}

// Bump a value to the next bit pattern by adding 1 to its integer
// representation. Used to vary a loop input across iterations without
// charging the benchmark for an unrelated fp add on every iteration.
//
// Only `double` and `fltflt` overloads are provided: on most GPUs an
// int64 add is significantly faster than an fp64 add, and the fltflt
// alternative would dispatch through fltflt_add (~20 fp32 ops). For
// float, fp32 add and int32 add run at the same rate, so call sites
// should keep `x = x + small`.
//
// For fltflt, only the hi component is bumped, leaving the pair
// non-canonical -- benches only care that the value differs from the
// previous iteration. The +1 would be UB at the largest-positive bit
// pattern (a NaN); call sites here never reach that.
__device__ __forceinline__ void bump_ulp(double &x) {
  x = __longlong_as_double(__double_as_longlong(x) + 1LL);
}
__device__ __forceinline__ void bump_ulp(fltflt &x) {
  x.hi = __int_as_float(__float_as_int(x.hi) + 1);
}

// Compute-bound kernel used solely to spin GPU clocks up to steady
// state. Self-contained (doesn't depend on any of the iterative_*
// kernels below) so it can be called from warmup_gpu_once() before
// they're defined.
__global__ void clock_warmup_kernel(float *out, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = static_cast<float>(idx) * 0.001f;
  #pragma unroll 1
  for (int i = 0; i < N; i++) {
    acc = acc * 1.0000001f + 1.0e-6f;
  }
  if (idx == 0) out[0] = acc;
}

// Idempotent process-level GPU warmup. Only the first call in a
// process actually runs warmup launches; subsequent calls are no-ops.
// Brings GPU clocks to steady state before the first nvbench timing
// window so the *first* benchmark to execute (whichever one that may
// be) does not get charged for clock ramp-up.
//
// inner_iters is sized so each launch runs ~50 ms on Blackwell, and
// four launches give ~200 ms total -- comfortably past the GPU's clock
// ramp window. (FMA-bound work retires near peak fp32 throughput, so
// undersized warmups would otherwise be over in a few ms.)
static void warmup_gpu_once()
{
  static bool warmed = false;
  if (warmed) return;
  warmed = true;

  constexpr int block_size = 256;
  constexpr int grid_size  = 1024;
  constexpr int inner_iters = 2'000'000;

  float *tmp = nullptr;
  MATX_CUDA_CHECK(cudaMalloc(&tmp, sizeof(float)));
  for (int w = 0; w < 4; w++) {
    clock_warmup_kernel<<<grid_size, block_size>>>(tmp, inner_iters);
    MATX_CUDA_CHECK_LAST_ERROR();
  }
  MATX_CUDA_CHECK(cudaDeviceSynchronize());
  MATX_CUDA_CHECK(cudaFree(tmp));
}

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
// Addition Throughput Benchmark
//
// Many independent accumulators (ILP_FACTOR=8) and outer-loop unrolling
// expose maximum instruction-level parallelism. Latency-hiding fully covers
// per-call dependency chains, so this measures *throughput*: ops/sec when
// the warp scheduler always has independent work in flight.
//==============================================================================
template <typename PrecisionType>
void fltflt_bench_add_throughput(nvbench::state &state, nvbench::type_list<PrecisionType>)
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

  warmup_gpu_once();
  exec.sync();

  // Benchmark execution
  state.exec([&](nvbench::launch &launch) {
    iterative_add_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
}

NVBENCH_BENCH_TYPES(fltflt_bench_add_throughput, NVBENCH_TYPE_AXES(precision_types))
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_sub_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_mul_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_div_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_sqrt_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
}

NVBENCH_BENCH_TYPES(fltflt_bench_sqrt, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Square Root Fast Benchmark
// For float/double, this is identical to the sqrt benchmark (sqrtf/sqrt).
// For fltflt, this uses fltflt_sqrt_fast instead of fltflt_sqrt.
//==============================================================================
template <typename T>
__global__ void iterative_sqrt_fast_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    T val[ILP_FACTOR];
    const T init_val = static_cast<T>(2.718281828);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      if constexpr (std::is_same_v<T, fltflt>) {
        val[ilp] = fltflt_sqrt_fast(init_val);
      } else {
        val[ilp] = sqrt(init_val);
      }
    }

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 1; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        if constexpr (std::is_same_v<T, fltflt>) {
          val[ilp] = fltflt_sqrt_fast(val[ilp]);
        } else {
          val[ilp] = sqrt(val[ilp]);
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
void fltflt_bench_sqrt_fast(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_sqrt_fast_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
}

NVBENCH_BENCH_TYPES(fltflt_bench_sqrt_fast, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// 3D Norm Benchmark: sqrt(dx^2 + dy^2 + dz^2)
// Each ILP lane has distinct dx values that depend on the previous iteration's
// result, creating a true dependency chain that prevents CSE across lanes.
//==============================================================================
template <typename T>
__global__ void iterative_norm3d_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // Per-lane dx values create independent dependency chains
    T dx[ILP_FACTOR];
    const T dy = static_cast<T>(-487293.18274);
    const T dz = static_cast<T>(183649.27391);

    #pragma unroll
    for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
      dx[ilp] = static_cast<T>(312847.91837) + static_cast<T>(ilp * 0.1);
    }

    #pragma unroll ITER_UNROLL_FACTOR
    for (int32_t i = 0; i < iterations; i++) {
      #pragma unroll
      for (int ilp = 0; ilp < ILP_FACTOR; ilp++) {
        T norm;
        if constexpr (std::is_same_v<T, fltflt>) {
          norm = fltflt_norm3d(dx[ilp], dy, dz);
        } else {
          norm = sqrt(dx[ilp] * dx[ilp] + dy * dy + dz * dz);
        }
        // Feed result back into dx to create a dependency chain.
        // Add the computed norm and subtract off the approximate
        // expected norm to keep dx in a stable range while preventing
        // the compiler from optimizing away the computation.
        if constexpr (std::is_same_v<T, fltflt>) {
          // fltflt addition/subtraction is expensive and we do not want to bias the benchmark
          // too much, so at least keep the expected norm as a float rather than fltflt to
          // reduce the cost of the subtraction.
          dx[ilp] = dx[ilp] + (norm - 607499.4f);
        } else {
          dx[ilp] = dx[ilp] + (norm - static_cast<T>(607499.4));
        }
      }
    }

    T result_val = dx[0];
    #pragma unroll
    for (int ilp = 1; ilp < ILP_FACTOR; ilp++) {
      result_val = result_val + dx[ilp];
    }
    result[idx] = result_val;
  }
}

template <typename PrecisionType>
void fltflt_bench_norm3d(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_norm3d_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  // 3 muls + 2 adds + 1 sqrt + 1 sub + 1 add = 8 ops per inner iteration.
  add_gops_per_sec_summary(state, /*ops_per_op=*/8.0);
}

NVBENCH_BENCH_TYPES(fltflt_bench_norm3d, NVBENCH_TYPE_AXES(precision_types))
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_abs_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_fma_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state, /*ops_per_op=*/2.0);  // FMA = 1 mul + 1 add
}

NVBENCH_BENCH_TYPES(fltflt_bench_fma, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Fused Multiply-Add (Approx) Benchmark
//
// fltflt_fma_approx() drops the second-order a.lo*b.lo cross term. For
// non-fltflt types this kernel is identical to iterative_fma_kernel, so
// the float/double rows act as a baseline for the fltflt fast-path row.
//==============================================================================
template <typename T>
__global__ void iterative_fma_approx_kernel(T* __restrict__ result, int64_t size, int32_t iterations)
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
          acc[ilp] = fltflt_fma_approx(val_a, acc[ilp], val_b);
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

template <typename PrecisionType>
void fltflt_bench_fma_approx(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const index_t size = static_cast<index_t>(state.get_int64("Array Size"));
  const int32_t iterations = static_cast<int32_t>(state.get_int64("Iterations"));
  cudaExecutor exec{0};

  auto result = make_tensor<PrecisionType>({size});

  state.add_element_count(size, "NumElements");
  state.add_global_memory_writes<PrecisionType>(size);

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_fma_approx_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state, /*ops_per_op=*/2.0);  // FMA = 1 mul + 1 add
}

NVBENCH_BENCH_TYPES(fltflt_bench_fma_approx, NVBENCH_TYPE_AXES(precision_types))
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_madd_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state, /*ops_per_op=*/2.0);  // separate mul + add
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
    constexpr T init_val = static_cast<T>(33554432.5);

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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_round_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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
      if constexpr (std::is_same_v<T, float>) {
        // fp32 add is full-rate, no benefit from a bit-twiddle here.
        init_val += 2048.0f;
      } else {
        // Bit-pattern bump avoids an fp64 add (or full fltflt_add) on
        // every iteration just to defeat hoisting of the fmod call.
        bump_ulp(init_val);
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_fmod_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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
      if constexpr (std::is_same_v<T, float>) {
        // fp32 add is full-rate, no benefit from a bit-twiddle here.
        init_val += 2048.0f;
      } else {
        // Bit-pattern bump avoids an fp64 add (or full fltflt_add) on
        // every iteration just to defeat hoisting of the trunc call.
        bump_ulp(init_val);
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_trunc_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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
      if constexpr (std::is_same_v<T, float>) {
        // fp32 add is full-rate, no benefit from a bit-twiddle here.
        init_val += 2048.0f;
      } else {
        // Bit-pattern bump avoids an fp64 add (or full fltflt_add) on
        // every iteration just to defeat hoisting of the floor call.
        bump_ulp(init_val);
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_floor_kernel<<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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
      if constexpr (std::is_same_v<T, float>) {
        // fp32 add is full-rate, no benefit from a bit-twiddle here.
        src_val = src_val + static_cast<T>(0.0001);
      } else {
        // Vary src_val via bit-pattern bump -- keeps the cast2dbl cost
        // un-contaminated by an unrelated fp64 add or fltflt_add per
        // iteration.
        bump_ulp(src_val);
      }
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_cast2dbl_kernel<PrecisionType><<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
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
      if constexpr (std::is_same_v<T, float>) {
        // fp32 add is full-rate, no benefit from a bit-twiddle here.
        src_val = src_val + static_cast<T>(0.0001);
      } else {
        // Vary src_val via bit-pattern bump -- keeps the cast2fltflt
        // cost un-contaminated by an unrelated fp64 add or fltflt_add
        // per iteration.
        bump_ulp(src_val);
      }
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

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    iterative_cast2fltflt_kernel<PrecisionType><<<grid_size, block_size, 0, (cudaStream_t)launch.get_stream()>>>(
      result.Data(), size, iterations);
  });
  add_gops_per_sec_summary(state);
}

NVBENCH_BENCH_TYPES(fltflt_bench_cast2fltflt, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_power_of_two_axis("Array Size", nvbench::range(24, 24, 1))
  .add_int64_axis("Iterations", {250});

//==============================================================================
// Addition Latency Benchmark
//
// Mirrors fltflt_bench_add_throughput but with the opposite scheduling
// posture: a single in-flight accumulator per thread, no ILP, no inner-loop
// unroll, and step varies per iteration (so the compiler cannot hoist or
// reassociate the chain). Each iteration's input depends on the previous
// iteration's output, so per-call dependency chains directly drive runtime.
//
// For fltflt this exposes the depth difference between the production
// fltflt_add (Zhang & Aiken SC'25 Fig 2 FPAN, critical path ~10 fp32 ops)
// and Thall's df64_add (~13 fp32 ops). The "Blocks" axis sweeps the
// latency->throughput transition: at Blocks=1 only one warp runs on one SM,
// fully exposing the chain, while at Blocks=1024 the scheduler has many
// warps in flight and latency is partially hidden.
//==============================================================================
template <typename PrecisionType>
__global__ void chain_add_kernel(int N, PrecisionType *__restrict__ out)
{
  // Construct via float so the same expression compiles for float, double,
  // and fltflt (each has a constructor accepting a float).
  PrecisionType acc{1.0f};
#pragma unroll 1
  for (int i = 0; i < N; i++) {
    // step varies per iteration to defeat loop-invariant hoisting and force
    // a true data dependency on the running accumulator.
    const PrecisionType step{static_cast<float>(i + 1)};
    acc = acc + step;  // dispatches to PrecisionType's operator+
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <typename PrecisionType>
void fltflt_bench_add_latency(nvbench::state &state, nvbench::type_list<PrecisionType>)
{
  const int chain_len = static_cast<int>(state.get_int64("Chain Length"));
  const int blocks    = static_cast<int>(state.get_int64("Blocks"));
  constexpr int threads = 32;  // exactly one warp per block

  cudaExecutor exec{0};
  const size_t total_threads = static_cast<size_t>(blocks) * threads;
  auto result = make_tensor<PrecisionType>({static_cast<index_t>(total_threads)});

  state.add_element_count(static_cast<int64_t>(chain_len) * total_threads, "ops");

  warmup_gpu_once();
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    chain_add_kernel<PrecisionType>
        <<<blocks, threads, 0, (cudaStream_t)launch.get_stream()>>>(
            chain_len, result.Data());
  });
}

NVBENCH_BENCH_TYPES(fltflt_bench_add_latency, NVBENCH_TYPE_AXES(precision_types))
  .add_int64_axis("Chain Length", {1024, 4096, 16384})
  // Blocks=1   : 1 warp on 1 SM, all other SMs idle  -- latency fully exposed
  // Blocks=4   : 4 warps on 4 SMs, each SM has 1 warp -- still latency-bound
  // Blocks=160 : ~1 block per SM on a ~160-SM device  -- partial latency hiding
  // Blocks=1024: many blocks per SM                  -- throughput-bound, latency hides
  .add_int64_axis("Blocks",       {1, 4, 160, 1024});
