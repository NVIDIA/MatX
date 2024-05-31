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
#include <memory>

using namespace matx;
#define FFT_TYPE CUFFT_C2C

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  using complex = cuda::std::complex<float>;

  index_t num_samp = 100000000;
  index_t num_samp_resamp = 100000;
  index_t N = std::min(num_samp, num_samp_resamp);
  index_t nyq = N / 2 + 1;
  constexpr uint32_t num_iterations = 100;
  float time_ms;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaExecutor exec{stream};

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Create data objects and views
  tensor_t<float, 1> sigView({num_samp});
  tensor_t<complex, 1> sigViewComplex({num_samp / 2 + 1});
  tensor_t<float, 1> resampView({num_samp_resamp});

  (sigView = random<float>({num_samp}, NORMAL)).run(exec);

  (sigViewComplex = fft(sigView)).run(exec);

  // Slice
  auto sliceView = sigViewComplex.Slice({0}, {nyq});

  // Inverse Transform - FFT size based on output
  (resampView = ifft(sliceView)).run(exec);

  cudaEventRecord(start, stream);

  for (uint32_t i = 0; i < num_iterations; i++) {
    // Launch 1D FFT
    (sigViewComplex = fft(sigView)).run(exec);

    // Slice
    auto sv = sigViewComplex.Slice({0}, {nyq});

    // Inverse Transform - FFT size based on output
    (resampView = ifft(sv)).run(exec);
  }

  cudaEventRecord(stop, stream);
  exec.sync();
  cudaEventElapsedTime(&time_ms, start, stop);

  printf("Resample Kernel Time = %.2fms per iteration\n",
         time_ms / num_iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}