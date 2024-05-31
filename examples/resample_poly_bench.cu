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
// polyphase resampler kernel(s). Typically, the parameters below (batch size, filter
// length, input signal length, up/down factors) will be adjusted to a range of interest
// and the benchmark will be run with and without the proposed kernel changes.

constexpr int NUM_WARMUP_ITERATIONS = 10;

// Number of iterations per timed test. Iteration times are averaged in the report.
constexpr int NUM_ITERATIONS = 20;

template <typename InType>
void ResamplePolyBench()
{
  struct {
    matx::index_t num_batches;
    matx::index_t input_len;
    matx::index_t up;
    matx::index_t down;
  } test_cases[] = {
    { 1, 256, 384, 3125 },
    { 1, 256, 384, 175 },
    { 1, 256, 4, 5 },
    { 1, 256, 1, 4 },
    { 1, 256, 1, 16 },
    { 1, 3000, 384, 3125 },
    { 1, 3000, 384, 175 },
    { 1, 3000, 4, 5 },
    { 1, 3000, 1, 4 },
    { 1, 3000, 1, 16 },
    { 1, 31000, 384, 3125 },
    { 1, 31000, 384, 175 },
    { 1, 31000, 4, 5 },
    { 1, 31000, 1, 4 },
    { 1, 31000, 1, 16 },
    { 1, 256000, 384, 3125 },
    { 1, 256000, 384, 175 },
    { 1, 256000, 64, 37 },
    { 1, 256000, 2, 3 },
    { 1, 256000, 4, 5 },
    { 1, 256000, 7, 64 },
    { 1, 256000, 7, 128 },
    { 1, 256000, 1, 4 },
    { 1, 256000, 1, 8 },
    { 1, 256000, 1, 16 },
    { 1, 256000, 1, 64 },
    { 1, 256000, 4, 1 },
    { 1, 256000, 8, 1 },
    { 1, 256000, 16, 1 },
    { 1, 256000, 64, 1 },
    { 1, 256000, 4, 1 },
    { 1, 256000, 8, 1 },
    { 1, 256000, 16, 1 },
    { 42, 256000, 384, 3125 },
    { 42, 256000, 384, 175 },
    { 42, 256000, 2, 3 },
    { 42, 256000, 4, 5 },
    { 42, 256000, 1, 4 },
    { 42, 256000, 1, 8 },
    { 42, 256000, 1, 16 },
    { 42, 256000, 1, 64 },
    { 1, 100000000, 384, 3125 },
    { 1, 100000000, 384, 175 },
    { 1, 100000000, 2, 3 },
    { 1, 100000000, 4, 5 },
    { 1, 100000000, 7, 64 },
    { 1, 100000000, 7, 128 },
    { 1, 100000000, 1, 2 },
    { 1, 100000000, 1, 4 },
    { 1, 100000000, 1, 8 },
    { 1, 100000000, 1, 16 },
    { 1, 100000000, 1, 192 },
    { 1, 100000000, 4, 1 },
    { 1, 100000000, 8, 1 },
    { 1, 100000000, 16, 1 },
  };

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaExecutor exec{stream};

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
      const matx::index_t num_batches = test_cases[i].num_batches;
      const matx::index_t input_len = test_cases[i].input_len;
      const matx::index_t up = test_cases[i].up;
      const matx::index_t down = test_cases[i].down;
      const matx::index_t half_len = 10 * std::max(up, down);
      const matx::index_t filter_len = 2 * half_len + 1;
      const matx::index_t filter_len_per_phase = (filter_len + up - 1) / up;

      const index_t up_len = input_len * up;
      const index_t output_len = up_len / down + ((up_len % down) ? 1 : 0);

      auto input = matx::make_tensor<InType, 2>({num_batches, input_len}, MATX_DEVICE_MEMORY);
      auto filter = matx::make_tensor<InType, 1>({filter_len}, MATX_DEVICE_MEMORY);
      auto output = matx::make_tensor<InType, 2>({num_batches, output_len}, MATX_DEVICE_MEMORY);

      (input = static_cast<InType>(1.0)).run(exec);
      (filter = static_cast<InType>(1.0)).run(exec);

      exec.sync();

      for (int k = 0; k < NUM_WARMUP_ITERATIONS; k++) {
        (output = matx::resample_poly(input, filter, up, down)).run(exec);
      }

      exec.sync();

      float elapsed_ms = 0.0f;
      cudaEventRecord(start, stream);
      for (int k = 0; k < NUM_ITERATIONS; k++) {
        (output = matx::resample_poly(input, filter, up, down)).run(exec);
      }
      cudaEventRecord(stop, stream);
      exec.sync();
      CUDA_CHECK_LAST_ERROR();
      cudaEventElapsedTime(&elapsed_ms, start, stop);

      const double gflops = static_cast<double>(num_batches*(2*filter_len_per_phase-1)*output_len) / 1.0e9;
      const double avg_elapsed_us = (static_cast<double>(elapsed_ms)/NUM_ITERATIONS)*1.0e3;
      printf("Batches: %5" INDEX_T_FMT "  FilterLen: %5" INDEX_T_FMT "  InputLen: %9" INDEX_T_FMT "  OutputLen: %8" INDEX_T_FMT
      "  Up/Down: %4" INDEX_T_FMT "/%4" INDEX_T_FMT " Elapsed Usecs: %12.1f GFLOPS: %10.3f\n",
        num_batches, filter_len, input_len, output_len, up, down, avg_elapsed_us, gflops/(avg_elapsed_us/1.0e6));
  }

  CUDA_CHECK_LAST_ERROR();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  printf("Benchmarking float\n");
  ResamplePolyBench<float>();

  MATX_EXIT_HANDLER();
}
