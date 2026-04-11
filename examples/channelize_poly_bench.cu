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
#include <cstring>
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

template <typename T>
const char *TypeName() {
  if constexpr (std::is_same_v<T, float>) return "float";
  else if constexpr (std::is_same_v<T, double>) return "double";
  else if constexpr (std::is_same_v<T, cuda::std::complex<float>>) return "complex<float>";
  else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) return "complex<double>";
  else return "unknown";
}

template <typename InType, typename OutType, typename FilterType>
void ChannelizePolyBench(matx::index_t channel_start, matx::index_t channel_stop, matx::index_t oversample_factor = -1)
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
    { 1, 17, 8192*1024 },
    { 42, 17, 8192*1024 }
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
      const matx::index_t decimation_factor = (oversample_factor <= 0) ? num_channels : num_channels / oversample_factor;
      const matx::index_t input_len = test_cases[i].input_len;
      const matx::index_t output_len_per_channel = (input_len + decimation_factor - 1) / decimation_factor;

      if (input_len < num_channels * 100) {
        continue;
      }

      auto input = matx::make_tensor<InType, 2>({num_batches, input_len});
      auto filter = matx::make_tensor<FilterType, 1>({filter_len});
      auto output = matx::make_tensor<OutType, 3>({num_batches, output_len_per_channel, num_channels});
      (input = static_cast<InType>(1)).run(exec);
      (filter = static_cast<FilterType>(1)).run(exec);

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
      MATX_CUDA_CHECK_LAST_ERROR();
      cudaEventElapsedTime(&elapsed_ms, start, stop);

      const double avg_elapsed_us = (static_cast<double>(elapsed_ms)/NUM_ITERATIONS)*1.0e3;
      printf("Batches: %5" MATX_INDEX_T_FMT " Channels: %5" MATX_INDEX_T_FMT " Decimation: %5" MATX_INDEX_T_FMT " FilterLen: %5" MATX_INDEX_T_FMT
        " InputLen: %7" MATX_INDEX_T_FMT " Elapsed Usecs: %12.1f MPts/sec: %12.3f\n",
        num_batches, num_channels, decimation_factor, filter_len, input_len, avg_elapsed_us,
        static_cast<double>(num_batches*num_channels*output_len_per_channel)/1.0e6/(avg_elapsed_us/1.0e6));
    }
    printf("\n");
  }

  MATX_CUDA_CHECK_LAST_ERROR();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
}

enum class Precision { Float, Double };
enum class Domain { Real, Complex };

struct BenchConfig {
  Precision input_prec   = Precision::Float;
  Domain    input_domain = Domain::Complex;
  Precision filter_prec  = Precision::Float;
  Domain    filter_domain = Domain::Real;
  matx::index_t channel_start = 2;
  matx::index_t channel_stop  = 12;
  matx::index_t oversample_factor = -1;
};

void PrintUsage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  --input-type   <type>   Input type: float, double, cf, cd (default: cf)\n");
  printf("  --filter-type  <type>   Filter type: float, double, cf, cd (default: float)\n");
  printf("  --channel-start <N>     First channel count (default: 2)\n");
  printf("  --channel-stop  <N>     Last channel count (default: 12)\n");
  printf("  --oversample    <N>     Oversampling factor; channels/N = decimation (default: none)\n");
  printf("\n");
  printf("Type shorthands: float, double, cf (complex<float>), cd (complex<double>)\n");
}

bool ParseType(const char *s, Precision &prec, Domain &dom) {
  if (strcmp(s, "float") == 0)       { prec = Precision::Float;  dom = Domain::Real;    return true; }
  if (strcmp(s, "double") == 0)      { prec = Precision::Double; dom = Domain::Real;    return true; }
  if (strcmp(s, "cf") == 0)          { prec = Precision::Float;  dom = Domain::Complex; return true; }
  if (strcmp(s, "cd") == 0)          { prec = Precision::Double; dom = Domain::Complex; return true; }
  return false;
}

const char *TypeLabel(Precision prec, Domain dom) {
  if (prec == Precision::Float  && dom == Domain::Real)    return "float";
  if (prec == Precision::Float  && dom == Domain::Complex) return "complex<float>";
  if (prec == Precision::Double && dom == Domain::Real)    return "double";
  if (prec == Precision::Double && dom == Domain::Complex) return "complex<double>";
  return "unknown";
}

template <typename T>
struct ScalarType { using type = T; };
template <typename T>
struct ScalarType<cuda::std::complex<T>> { using type = T; };

template <typename InType, typename FilterType>
void DispatchBench(const BenchConfig &cfg) {
  using in_scalar = typename ScalarType<InType>::type;
  using OutType = cuda::std::complex<in_scalar>;

  printf("Input: %-16s  Filter: %-16s  Output: %-16s\n",
      TypeName<InType>(), TypeName<FilterType>(), TypeName<OutType>());
  printf("Channels: %" MATX_INDEX_T_FMT " - %" MATX_INDEX_T_FMT, cfg.channel_start, cfg.channel_stop);
  if (cfg.oversample_factor > 0) {
    printf("  Oversample: %" MATX_INDEX_T_FMT "x", cfg.oversample_factor);
  }
  printf("\n\n");

  ChannelizePolyBench<InType, OutType, FilterType>(cfg.channel_start, cfg.channel_stop, cfg.oversample_factor);
}

void RunBench(const BenchConfig &cfg) {
  // Dispatch on (input, filter) type combination. Each lambda instantiates
  // only the single combination selected at runtime, keeping compile time
  // proportional to the number of supported types rather than their product.
  auto go = [&](auto in_tag, auto filt_tag) {
    DispatchBench<decltype(in_tag), decltype(filt_tag)>(cfg);
  };

  auto dispatch_filter = [&](auto in_tag) {
    if (cfg.filter_prec == Precision::Float && cfg.filter_domain == Domain::Real)
      go(in_tag, float{});
    else if (cfg.filter_prec == Precision::Double && cfg.filter_domain == Domain::Real)
      go(in_tag, double{});
    else if (cfg.filter_prec == Precision::Float && cfg.filter_domain == Domain::Complex)
      go(in_tag, cuda::std::complex<float>{});
    else
      go(in_tag, cuda::std::complex<double>{});
  };

  if (cfg.input_prec == Precision::Float && cfg.input_domain == Domain::Real)
    dispatch_filter(float{});
  else if (cfg.input_prec == Precision::Double && cfg.input_domain == Domain::Real)
    dispatch_filter(double{});
  else if (cfg.input_prec == Precision::Float && cfg.input_domain == Domain::Complex)
    dispatch_filter(cuda::std::complex<float>{});
  else
    dispatch_filter(cuda::std::complex<double>{});
}

int main(int argc, char **argv)
{
  MATX_ENTER_HANDLER();

  BenchConfig cfg;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      PrintUsage(argv[0]);
      return 0;
    } else if (strcmp(argv[i], "--input-type") == 0 && i + 1 < argc) {
      if (!ParseType(argv[++i], cfg.input_prec, cfg.input_domain)) {
        fprintf(stderr, "Unknown input type: %s\n", argv[i]);
        return 1;
      }
    } else if (strcmp(argv[i], "--filter-type") == 0 && i + 1 < argc) {
      if (!ParseType(argv[++i], cfg.filter_prec, cfg.filter_domain)) {
        fprintf(stderr, "Unknown filter type: %s\n", argv[i]);
        return 1;
      }
    } else if (strcmp(argv[i], "--channel-start") == 0 && i + 1 < argc) {
      cfg.channel_start = static_cast<matx::index_t>(atol(argv[++i]));
    } else if (strcmp(argv[i], "--channel-stop") == 0 && i + 1 < argc) {
      cfg.channel_stop = static_cast<matx::index_t>(atol(argv[++i]));
    } else if (strcmp(argv[i], "--oversample") == 0 && i + 1 < argc) {
      cfg.oversample_factor = static_cast<matx::index_t>(atol(argv[++i]));
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      PrintUsage(argv[0]);
      return 1;
    }
  }

  RunBench(cfg);

  matx::ClearCachesAndAllocations();

  MATX_EXIT_HANDLER();
}
