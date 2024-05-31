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
#include <cmath>
#include <memory>
#include <fstream>
#include <istream>
#include <cuda/std/complex>

using namespace matx;

// This example is used primarily for development purposes to benchmark the performance of the
// polyphase channelizer kernel(s). Typically, the parameters below (batch size, filter
// length, input signal length, and channel range) will be adjusted to a range of interest
// and the benchmark will be run with and without the proposed kernel changes.

constexpr int NUM_WARMUP_ITERATIONS = 2;

// Number of iterations per timed test. Iteration times are averaged in the report.
constexpr int NUM_ITERATIONS = 20;

template <typename InType, typename OutType>
void ChannelizePolyBench(matx::index_t channel_start, matx::index_t channel_stop)
{
  struct {
    matx::index_t num_batches;
    matx::index_t filter_len_per_channel;
    matx::index_t input_len;
  } test_cases[] = {
    { 1, 17, 256 },
    { 1, 17, 3000 },
    { 1, 17, 31000 },
    { 1, 17, 256000 },
    { 42, 17, 256000 },
    { 128, 17, 256000 },
  };

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaExecutor exec{};

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    for (matx::index_t num_channels = channel_start; num_channels <= channel_stop; num_channels++) {
      const matx::index_t num_batches = test_cases[i].num_batches;
      const matx::index_t filter_len = test_cases[i].filter_len_per_channel * num_channels;
      const matx::index_t input_len = test_cases[i].input_len;
      const matx::index_t output_len_per_channel = (input_len + num_channels - 1) / num_channels;

      auto input = matx::make_tensor<InType, 2>({num_batches, input_len});
      auto filter = matx::make_tensor<InType, 1>({filter_len});
      auto output = matx::make_tensor<OutType, 3>({num_batches, output_len_per_channel, num_channels});

      const matx::index_t decimation_factor = num_channels;

      for (int k = 0; k < NUM_WARMUP_ITERATIONS; k++) {
        (output = channelize_poly(input, filter, num_channels, decimation_factor)).run(exec);
      }

      exec.sync();

      float elapsed_ms = 0.0f;
      cudaEventRecord(start, stream);
      for (int k = 0; k < NUM_ITERATIONS; k++) {
        (output = channelize_poly(input, filter, num_channels, decimation_factor)).run(exec);
      }
      cudaEventRecord(stop, stream);
      exec.sync();
      CUDA_CHECK_LAST_ERROR();
      cudaEventElapsedTime(&elapsed_ms, start, stop);

      const double avg_elapsed_us = (static_cast<double>(elapsed_ms)/NUM_ITERATIONS)*1.0e3;
      printf("Batches: %5lld Channels: %5lld FilterLen: %5lld InputLen: %7lld Elapsed Usecs: %12.1f MPts/sec: %12.3f\n",
        num_batches, num_channels, filter_len, input_len, avg_elapsed_us,
        static_cast<double>(num_batches*num_channels*output_len_per_channel)/1.0e6/(avg_elapsed_us/1.0e6));
    }
    printf("\n");
  }

  CUDA_CHECK_LAST_ERROR();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  const matx::index_t channel_start = 3;
  const matx::index_t channel_stop = 10;

  // printf("Benchmarking float -> complex<float>\n");
  // ChannelizePolyBench<float,cuda::std::complex<float>>(channel_start, channel_stop);

  printf("Benchmarking complex<float> -> complex<float>\n");
  ChannelizePolyBench<cuda::std::complex<float>,cuda::std::complex<float>>(channel_start, channel_stop);

  // printf("Benchmarking double -> complex<double>\n");
  // ChannelizePolyBench<double,cuda::std::complex<double>>(channel_start, channel_stop);

  // printf("Benchmarking complex<double> -> complex<double>\n");
  // ChannelizePolyBench<cuda::std::complex<double>,cuda::std::complex<double>>(channel_start, channel_stop);

  MATX_EXIT_HANDLER();
}
