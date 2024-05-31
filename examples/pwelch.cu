////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2023, NVIDIA Corporation
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
#include <cuda/std/ccomplex>

using namespace matx;

/**
 * PWelch Example
 *
 * This example shows how to estimate the power spectral density of a signal using the pwelch() operator
 * using Welch's method.
 *
 */

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  using complex = cuda::std::complex<float>;

  float exec_time_ms;
  const int num_iterations = 100;
  index_t signal_size = 256;
  index_t nperseg = 32;
  index_t nfft = nperseg;
  index_t noverlap = 8;
  float ftone = 3.0;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaExecutor exec{stream};

  // Create input signal as a complex exponential
  auto sample_index_range = range<0>({signal_size},0.f,1.f);
  auto phase = 2.f * static_cast<float>(M_PI) * ftone * sample_index_range / static_cast<float>(nfft);
  auto tmp_x = expj(phase);
  auto x = make_tensor<complex>({signal_size});
  (x = tmp_x).run(exec); // pre-compute x, tmp_x is otherwise lazily evaluated

  // Create output tensor
  auto Pxx  = make_tensor<typename complex::value_type>({nfft});

  // Run one time to pre-cache the FFT plan
  (Pxx = pwelch(x, nperseg, noverlap, nfft)).run(exec);
  exec.sync();

  // Start the timing
  cudaEventRecord(start, stream);

  // Start the timing
  cudaEventRecord(start, stream);

  for (int iteration = 0; iteration < num_iterations; iteration++) {
    // Use the PWelch operator
    (Pxx = pwelch(x, nperseg, noverlap, nfft)).run(exec);
  }

  cudaEventRecord(stop, stream);
  exec.sync();
  cudaEventElapsedTime(&exec_time_ms, start, stop);

  printf("Output Pxx:\n");
  print(Pxx);
  printf("PWelchOp avg runtime = %.3f ms\n", exec_time_ms / num_iterations);

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
  return 0;
}
