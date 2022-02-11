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
#include "matx_filter.h"
#include <cassert>
#include <cstdio>
#include <cuda/std/ccomplex>

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  using complex = cuda::std::complex<float>;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);  

  if (prop.sharedMemPerBlock < 40000) {
    printf("Recursive filter example requires at least 40KB of shared memory to run. Exiting.");
    return 0;
  }

  uint32_t iterations = 100;
  index_t numSamples = 16384000;
  constexpr uint32_t recLen = 2;
  constexpr uint32_t nonRecLen = 2;
  index_t batches = 10;
  float time_ms;

  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "NumSamples: " << numSamples << std::endl;
  std::cout << "Batches: " << batches << std::endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  using OutType = float;
  using InType = float;
  using FilterType = float;

  // Create data objects
  tensor_t<InType, 2> inView({batches, numSamples});
  tensor_t<InType, 2> outView({batches, numSamples});
  tensor_t<InType, 1> solView({numSamples});

  // Create views into data objects
  auto rCoeffs = std::array<FilterType, 2>{0.4f, -0.1f};
  auto nrCoeffs = std::array<FilterType, 2>{2.0f, 1.0f};

  // initialize input data
  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < inView.Size(1); i++) {
      inView(b, i) = {static_cast<float>(i & 32) / 16.0f - 1.0f};
      solView(i) = {0.0};
    }
  }

  // Init non-recursive solution
  for (int i = 0; i < (int)numSamples; i++) {
    for (int i_nr = 0; i_nr < (int)nonRecLen; i_nr++) {
      if ((i - i_nr) >= 0) {
        solView(i) += nrCoeffs[i_nr] * inView(0, i - i_nr);
      }
    }
  }

  // Recursive pieces. We do these separately for debugging purposes since the
  // kernel is done in two stages
  for (int i = 1; i < (int)numSamples; i++) {
    for (int i_r = 0; i_r < (int)recLen; i_r++) {
      if ((i - i_r - 1) >= 0) {
        solView(i) += rCoeffs[i_r] * solView(i - i_r - 1);
      }
    }
  }

  inView.PrefetchDevice(stream);
  outView.PrefetchDevice(stream);

  // Measure recursive runtime
  cudaStreamSynchronize(stream);
  cudaEventRecord(start, stream);

  for (uint32_t i = 0; i < iterations; i++) {
    filter(outView, inView, rCoeffs, nrCoeffs, stream);
  }

  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= static_cast<float>(iterations);

  printf("Recursive kernel time = %.2fus (%.2fGB/s), %.2f billion/s\n",
         time_ms * 1e3,
         static_cast<double>(batches * inView.Size(1) * sizeof(InType) * 2) /
             1e9 / (time_ms / 1e3),
         static_cast<double>(batches * inView.Size(1)) / 1e9 / (time_ms / 1e3));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  matxPrintMemoryStatistics();

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
