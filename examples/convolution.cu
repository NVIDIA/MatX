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

#include "matx_type_utils.h"
#include "matx_tensor.h"
#include "matx_conv.h"
#include <cassert>
#include <cstdio>

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  typedef cuda::std::complex<float> complex;

  uint32_t iterations = 10;
  constexpr index_t numSamples = 1638400;
  constexpr index_t filterLen = 10;
  constexpr index_t batches = 10;
  float time_ms;

  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "NumSamples: " << numSamples << std::endl;
  std::cout << "Batches: " << batches << std::endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  using InType = float;
  using OutType = float;
  using FilterType = float;

  // Create data objects
  auto inView = make_tensor<InType>({batches, numSamples});
  auto outView = make_tensor<OutType>({batches, numSamples + filterLen - 1});
  auto solView = make_tensor<InType>({batches, numSamples + filterLen - 1});
  auto filterView = make_tensor<FilterType>({filterLen});


  // initialize input data
  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < inView.Size(1); i++) {
      inView(b, i) = {static_cast<float>(static_cast<double>(i & 32) / 16.0 - 1)};
      solView(b,i) = {0.0};
    }
  }


  // Init Filters
  filterView(0) = 0.5;
  for (auto f = 1; f < filterLen; f++) {
    filterView(f) = filterView(f-1) * 0.99f;
  }

  inView.PrefetchDevice(stream);
  filterView.PrefetchDevice(stream);

  // Measure recursive runtime
  cudaStreamSynchronize(stream);
  cudaEventRecord(start, stream);

  for (uint32_t i = 0; i < iterations; i++) {
    conv1d(outView, inView, filterView, matxConvCorrMode_t::MATX_C_MODE_FULL,
           stream);
  }
  

  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms /= static_cast<float>(iterations);

  printf("Convolution kernel time = %.2fus (%.2fGB/s), %.2f billion/s\n",
         time_ms * 1e3,
         static_cast<double>(batches * inView.Size(1) * sizeof(InType) * 2) /
             1e9 / (time_ms / 1e3),
         static_cast<double>(batches * inView.Size(1)) / 1e9 / (time_ms / 1e3));

  // 2D convolution of a 4x4 filter with a 10,000x10,000 input signal
  // constexpr int filter_dim_2d = 4;
  // tensor_t<InType, 2> filter2DData(
  //     {filter_dim_2d, filter_dim_2d});
  // auto filter2DView = filter2DData.View();
  // tensor_t<InType, 2> in2DData(
  //     {(uint32_t)10e3, (uint32_t)10e3});
  // auto in2DView = in2DData.View();
  // tensor_t<OutType, 2> out2DData(
  //     {(uint32_t) 10e3 + filter_dim_2d - 1, (uint32_t) 10e3 + filter_dim_2d -
  //     1});
  // auto out2DView = out2DData.View();
  // filter2DData.PrefetchDevice(stream);
  // in2DData.PrefetchDevice(stream);
  // out2DData.PrefetchDevice(stream);

  // Measure recursive runtime
  // cudaEventRecord(start, stream);

  // for (auto i = 0; i < iterations; i++)
  // {
  //   if (matxDirectConv2DM(out2DView, in2DView, filter2DView, stream) !=
  //   matxSuccess) {
  //     printf("Error running convolution\n");
  //   }
  // }

  // cudaEventRecord(stop, stream);
  // cudaStreamSynchronize(stream);
  // cudaEventElapsedTime(&time_ms, start, stop);
  // time_ms /= static_cast<float>(iterations);

  // printf("2D Convolution kernel time = %.2fus (%.2fGB/s), %.2f billion/s\n",
  //        time_ms * 1e3,
  //        static_cast<double>(inView.Size(0) * inView.Size(1) * sizeof(InType) *
  //                            2) /
  //            1e9 / (time_ms / 1e3),
  //        static_cast<double>(inView.Size(0) * inView.Size(1)) / 1e9 /
  //            (time_ms / 1e3));

  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
  // cudaStreamDestroy(stream);

  matxPrintMemoryStatistics();

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
