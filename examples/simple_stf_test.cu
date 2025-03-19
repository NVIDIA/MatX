////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
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

#include "matx.h"
#include <cassert>
#include <cstdio>
#include <math.h>
#include <memory>

using namespace matx;

/**
 * MatX uses C++ expression templates to build arithmetic expressions that compile into a lazily-evaluated
 * type for executing on the device. Currently, nvcc cannot see certain optimizations
 * when building the expression tree that would be obvious by looking at the code. Specifically any code reusing
 * the same tensor multiple times appears to the compiler as separate tensors, and it may issue multiple load
 * instructions. While caching helps, this can have a slight performance impact when compared to native CUDA
 * kernels. To work around this problem, complex expressions can be placed in a custom operator by adding some
 * boilerplate code around the original expression. This custom operator can then be used either alone or inside
 * other arithmetic expressions, and only a single load is issues for each tensor. 
 * 
 * This example uses the Black-Scholes equtation to demonstrate the two ways to implement the equation in MatX, and
 * shows the performance difference.
 */

/* Arithmetic expression */
template<typename T1>
void compute_black_scholes_matx(tensor_t<T1,1>& K, 
                                tensor_t<T1,1>& S, 
                                tensor_t<T1,1>& V, 
                                tensor_t<T1,1>& r, 
                                tensor_t<T1,1>& T, 
                                tensor_t<T1,1>& output, 
#if 0
                                cudaExecutor& exec)
#else
                                stfExecutor& exec)
#endif
{
    auto VsqrtT = V * sqrt(T);
    auto d1 = (log(S / K) + (r + 0.5 * V * V) * T) / VsqrtT ;
    auto d2 = d1 - VsqrtT;
    auto cdf_d1 = normcdf(d1);
    auto cdf_d2 = normcdf(d2);
    auto expRT = exp(-1 * r * T); 
    (output = S * cdf_d1 - K * expRT * cdf_d2).run(exec);
    //std::cout << "Output : " << std::endl;
    //print(output);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  using dtype = double;

#if 0
  index_t input_size = 100000000;
#else
  index_t input_size = 10000;
#endif
  constexpr uint32_t num_iterations = 1000;
  float time_ms;

  tensor_t<dtype, 1> K_tensor{{input_size}};
  tensor_t<dtype, 1> S_tensor{{input_size}};
  tensor_t<dtype, 1> V_tensor{{input_size}};
  tensor_t<dtype, 1> r_tensor{{input_size}};
  tensor_t<dtype, 1> T_tensor{{input_size}};
  tensor_t<dtype, 1> output_tensor{{input_size}};  

  cudaStream_t stream;
  cudaStreamCreate(&stream);
#if 0
  cudaExecutor exec{stream};
#else
  stfExecutor exec{stream};
  auto ctx = exec.getCtx();
#endif

    /* Albert --- initilizing input .. */
    for (int i = 0; i < input_size; i++) {
        K_tensor(i) = dtype(i+1);
        S_tensor(i) = dtype(i+i+1);
        V_tensor(i) = dtype(i+i+i+1);
        r_tensor(i) = dtype(i+i+i+i+1);
        T_tensor(i) = dtype(i+i+i+i+i+1);
    }

//print(V_tensor);

  //compute_black_scholes_matx(K_tensor, S_tensor, V_tensor, r_tensor, T_tensor, output_tensor, exec);  

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

#if 0
  cudaEventRecord(start, stream);
#else
  cudaEventRecord(start, ctx.task_fence());
#endif
  // Time non-operator version
  for (uint32_t i = 0; i < num_iterations; i++) {
    compute_black_scholes_matx(K_tensor, S_tensor, V_tensor, r_tensor, T_tensor, output_tensor, exec);
  }
#if 0
  cudaEventRecord(stop, stream);
#else
  cudaEventRecord(stop, ctx.task_fence());
#endif
  exec.sync();
#if 1
  ctx.finalize();
#endif
  cudaEventElapsedTime(&time_ms, start, stop);

  //  printf("Output tensor :\n");
  //  print(output_tensor);
  printf("Time without custom operator = %fms per iteration\n",
         time_ms / num_iterations);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
