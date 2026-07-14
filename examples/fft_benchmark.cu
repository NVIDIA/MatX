////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
////////////////////////////////////////////////////////////////////////////////

/**
 * @file fft_benchmark.cu
 * @brief Compare handwritten CUDA + cuFFT with MatX CUDA and JIT execution.
 *
 * Every case computes the same batched signal-processing pipeline:
 *
 *   normalize each complex sample -> FFT -> calibrate -> magnitude -> dB
 *
 * Case 1 explicitly launches two CUDA kernels around cuFFT and materializes
 * both intermediate tensors. Case 2 expresses the complete pipeline in MatX
 * and runs it with cudaExecutor, which uses cuFFT and generated element-wise
 * kernels. Case 3 runs the identical MatX expression with CUDAJITExecutor so a
 * supported pipeline can be fused into one cuFFTDx-backed kernel.
 * When MatX is built with NVPL or FFTW, Case 4 runs that same expression with
 * AllThreadsHostExecutor and reports the CPU result separately.
 *
 * The default problem is 256 batches of 4096 samples with 2000 iterations per
 * timing trial. Override those values with:
 *
 *   ./fft_benchmark [batches] [fft_size] [iterations]
 *
 * JIT compilation is timed and reported separately from steady-state runtime.
 */

#include <matx.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-6f;
constexpr int kWarmupIterations = 10;
constexpr int kTrials = 7;
#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
constexpr int kHostWarmupIterations = 2;
constexpr int kHostTrials = 5;
constexpr int kMaxHostIterations = 10;
#endif

#define BENCH_CUDA_CHECK(call)                                                  \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

#define BENCH_CUFFT_CHECK(call)                                                 \
  do {                                                                          \
    const cufftResult error = (call);                                            \
    if (error != CUFFT_SUCCESS) {                                                \
      throw std::runtime_error(std::string(#call) +                              \
                               " failed with cuFFT error " +                    \
                               std::to_string(static_cast<int>(error)));         \
    }                                                                           \
  } while (false)

// Case 1 preprocessing: materialize signal / (abs(signal) + epsilon) before
// cuFFT. The MatX cases express this operation lazily instead.
__global__ void NormalizeKernel(cufftComplex *normalized,
                                const cufftComplex *signal,
                                size_t element_count)
{
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= element_count) {
    return;
  }

  const cufftComplex value = signal[idx];
  const float scale = 1.0f / (hypotf(value.x, value.y) + kEpsilon);
  normalized[idx] = cufftComplex{value.x * scale, value.y * scale};
}

// Case 1 post-processing: apply frequency-domain calibration, take the complex
// magnitude, and convert it to decibels after cuFFT has completed.
__global__ void MagnitudeDbKernel(float *spectrum_db,
                                  const cufftComplex *spectrum,
                                  const float *calibration,
                                  size_t element_count)
{
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= element_count) {
    return;
  }

  const cufftComplex value = spectrum[idx];
  const float gain = calibration[idx];
  spectrum_db[idx] =
      20.0f * log10f(hypotf(value.x * gain, value.y * gain) + kEpsilon);
}

template <typename RawCuda, typename MatxCuda, typename MatxJit>
std::array<double, 3> MeasureInterleavedMedianMs(
    cudaStream_t stream, int iterations, RawCuda &&run_raw_cuda,
    MatxCuda &&run_matx_cuda, MatxJit &&run_matx_jit)
{
  // Warm every implementation before recording events. This removes plan/cache
  // setup and one-time allocations from the steady-state measurements.
  for (int i = 0; i < kWarmupIterations; i++) {
    run_raw_cuda();
    run_matx_cuda();
    run_matx_jit();
  }
  BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{};
  cudaEvent_t stop{};
  BENCH_CUDA_CHECK(cudaEventCreate(&start));
  BENCH_CUDA_CHECK(cudaEventCreate(&stop));

  std::array<std::vector<double>, 3> trials;
  for (auto &implementation_trials : trials) {
    implementation_trials.reserve(kTrials);
  }

  auto measure = [&](size_t implementation, auto &&enqueue) {
    BENCH_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
      enqueue();
    }
    BENCH_CUDA_CHECK(cudaEventRecord(stop, stream));
    BENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    BENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    trials[implementation].push_back(static_cast<double>(elapsed_ms) /
                                     static_cast<double>(iterations));
  };

  // Rotate the order each trial so GPU clock and thermal changes do not
  // systematically favor the implementation measured first or last. Reporting
  // the median also reduces the effect of an occasional system interruption.
  for (int trial = 0; trial < kTrials; trial++) {
    switch (trial % 3) {
      case 0:
        measure(0, run_raw_cuda);
        measure(1, run_matx_cuda);
        measure(2, run_matx_jit);
        break;
      case 1:
        measure(1, run_matx_cuda);
        measure(2, run_matx_jit);
        measure(0, run_raw_cuda);
        break;
      default:
        measure(2, run_matx_jit);
        measure(0, run_raw_cuda);
        measure(1, run_matx_cuda);
        break;
    }
  }

  BENCH_CUDA_CHECK(cudaEventDestroy(start));
  BENCH_CUDA_CHECK(cudaEventDestroy(stop));

  std::array<double, 3> medians{};
  for (size_t implementation = 0; implementation < trials.size();
       implementation++) {
    auto &implementation_trials = trials[implementation];
    std::sort(implementation_trials.begin(), implementation_trials.end());
    medians[implementation] =
        implementation_trials[implementation_trials.size() / 2];
  }
  return medians;
}

template <typename Run>
double MeasureHostMs(Run &&run)
{
  const auto start = std::chrono::steady_clock::now();
  run();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
template <typename Run>
double MeasureHostMedianMs(int iterations, Run &&run)
{
  // FFTW/NVPL plan creation and unified-memory migration are first-use costs,
  // so warm them before measuring the steady-state host pipeline.
  for (int i = 0; i < kHostWarmupIterations; i++) {
    run();
  }

  std::vector<double> trials;
  trials.reserve(kHostTrials);
  for (int trial = 0; trial < kHostTrials; trial++) {
    const double elapsed_ms = MeasureHostMs([&]() {
      for (int i = 0; i < iterations; i++) {
        run();
      }
    });
    trials.push_back(elapsed_ms / static_cast<double>(iterations));
  }

  std::sort(trials.begin(), trials.end());
  return trials[trials.size() / 2];
}
#endif

struct ErrorSummary {
  double max_abs = 0.0;
  double max_rel = 0.0;
};

ErrorSummary Compare(const float *reference, const float *actual, size_t count)
{
  ErrorSummary error{};
  for (size_t i = 0; i < count; i++) {
    const double abs_error = std::abs(static_cast<double>(actual[i]) -
                                      static_cast<double>(reference[i]));
    const double denominator =
        std::max(1.0, std::abs(static_cast<double>(reference[i])));
    error.max_abs = std::max(error.max_abs, abs_error);
    error.max_rel = std::max(error.max_rel, abs_error / denominator);
  }
  return error;
}

void PrintResult(const char *name, double time_ms, size_t element_count,
                 double baseline_ms)
{
  const double giga_samples_per_second =
      static_cast<double>(element_count) / (time_ms * 1.0e6);
  std::cout << std::left << std::setw(28) << name << std::right << std::fixed
            << std::setprecision(4) << std::setw(11) << time_ms << " ms"
            << std::setw(12) << giga_samples_per_second << " Gsample/s"
            << std::setw(10) << baseline_ms / time_ms << "x\n";
}

#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
void PrintHostResult(const char *backend, double time_ms, size_t element_count,
                     int threads, int iterations)
{
  const double giga_samples_per_second =
      static_cast<double>(element_count) / (time_ms * 1.0e6);
  std::cout << "\nOptional host result (" << backend << ", " << threads
            << " threads, " << iterations << " iterations/trial):\n"
            << "MatX AllThreadsHostExecutor: " << std::fixed
            << std::setprecision(4) << time_ms << " ms, "
            << giga_samples_per_second << " Gsample/s\n";
}
#endif

int ParsePositive(const char *value, const char *name)
{
  const int parsed = std::atoi(value);
  if (parsed <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return parsed;
}

} // namespace

int main(int argc, char **argv)
{
#ifndef MATX_EN_MATHDX
  std::cerr << "This benchmark requires -DMATX_EN_MATHDX=ON\n";
  return 1;
#else
  // Defaults reproduce the shape and timing protocol reported in README.md.
  int batches = 256;
  int fft_size = 4096;
  int iterations = 2000;

  try {
    if (argc > 1) {
      batches = ParsePositive(argv[1], "batches");
    }
    if (argc > 2) {
      fft_size = ParsePositive(argv[2], "fft_size");
    }
    if (argc > 3) {
      iterations = ParsePositive(argv[3], "iterations");
    }
    if ((fft_size & (fft_size - 1)) != 0 || fft_size < 2) {
      throw std::invalid_argument("fft_size must be a power of two greater than one");
    }

    BENCH_CUDA_CHECK(cudaFree(nullptr));
    cudaStream_t stream{};
    BENCH_CUDA_CHECK(cudaStreamCreate(&stream));

    // All three cases share the same tensors, input values, CUDA stream, and
    // output shape. Only the way the pipeline is expressed and launched differs.
    const size_t element_count =
        static_cast<size_t>(batches) * static_cast<size_t>(fft_size);
    auto signal = matx::make_tensor<matx::fcomplex>({batches, fft_size});
    auto calibration = matx::make_tensor<float>({batches, fft_size});
    auto raw_cuda_out = matx::make_tensor<float>({batches, fft_size});
    auto matx_cuda_out = matx::make_tensor<float>({batches, fft_size});
    auto matx_jit_out = matx::make_tensor<float>({batches, fft_size});
#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
    auto matx_host_out = matx::make_tensor<float>({batches, fft_size});
#endif
    static_assert(sizeof(matx::fcomplex) == sizeof(cufftComplex));
    static_assert(alignof(matx::fcomplex) == alignof(cufftComplex));

    // Deterministic, nontrivial inputs make output validation reproducible.
    for (int batch = 0; batch < batches; batch++) {
      for (int sample = 0; sample < fft_size; sample++) {
        const float x = static_cast<float>(sample);
        const float b = static_cast<float>(batch);
        signal(batch, sample) = matx::fcomplex{
            std::sin(0.013f * x + 0.17f * b) + 0.001f * b,
            std::cos(0.021f * x - 0.11f * b) - 0.0005f * x};
        calibration(batch, sample) =
            0.75f + 0.25f * std::cos(0.007f * x + 0.03f * b);
      }
    }

    // Case 1: handwritten CUDA + cuFFT.
    //
    // Unlike the lazy MatX expression below, this path must allocate explicit
    // buffers for the normalized samples and FFT output, create and configure a
    // batched cuFFT plan, calculate launch geometry, and launch both CUDA kernels.
    cufftComplex *normalized{};
    cufftComplex *fft_output{};
    BENCH_CUDA_CHECK(
        cudaMalloc(&normalized, element_count * sizeof(cufftComplex)));
    BENCH_CUDA_CHECK(
        cudaMalloc(&fft_output, element_count * sizeof(cufftComplex)));

    cufftHandle plan{};
    int transform_size[] = {fft_size};
    BENCH_CUFFT_CHECK(cufftPlanMany(&plan, 1, transform_size,
                                    nullptr, 1, fft_size,
                                    nullptr, 1, fft_size,
                                    CUFFT_C2C, batches));
    BENCH_CUFFT_CHECK(cufftSetStream(plan, stream));

    constexpr int threads = 256;
    const int blocks = static_cast<int>((element_count + threads - 1) / threads);
    auto run_raw_cuda = [&]() {
      // Stage 1: evaluate and store the normalized FFT input.
      NormalizeKernel<<<blocks, threads, 0, stream>>>(
          normalized,
          reinterpret_cast<const cufftComplex *>(signal.Data()),
          element_count);
      BENCH_CUDA_CHECK(cudaGetLastError());

      // Stage 2: cuFFT reads that temporary and stores a second temporary.
      BENCH_CUFFT_CHECK(cufftExecC2C(
          plan,
          normalized,
          fft_output,
          CUFFT_FORWARD));

      // Stage 3: read the FFT output, calibrate it, and write the final result.
      MagnitudeDbKernel<<<blocks, threads, 0, stream>>>(
          raw_cuda_out.Data(), fft_output, calibration.Data(), element_count);
      BENCH_CUDA_CHECK(cudaGetLastError());
    };

    // Cases 2 and 3 share this exact MatX expression. Nothing is materialized
    // here: MatX operators are lazy until assignment is run with an executor.
    auto normalized_signal = signal / (matx::abs(signal) + kEpsilon);
    auto pipeline = 20.0f * matx::log10(
        matx::abs(matx::fft(normalized_signal) * calibration) + kEpsilon);
    // cuFFTDx supports a subset of FFT shapes and types. Fail clearly instead of
    // timing a different fallback path when the requested shape cannot be fused.
    if (!matx::jit_supported(pipeline)) {
      throw std::runtime_error(
          "The requested batch/FFT shape is not supported by CUDAJITExecutor");
    }

    matx::cudaExecutor cuda_exec{stream};
    matx::CUDAJITExecutor jit_exec{stream};

    // Case 2: the standard CUDA executor lowers the expression to a generated
    // normalization kernel, cuFFT, and a generated post-processing kernel.
    auto run_matx_cuda = [&]() { (matx_cuda_out = pipeline).run(cuda_exec); };

    // Case 3: only the executor changes. CUDAJITExecutor fuses the compatible
    // normalization, FFT, calibration, magnitude, and dB work into one kernel.
    auto run_matx_jit = [&]() { (matx_jit_out = pipeline).run(jit_exec); };

    // Build the cuFFT plan/cache and materialize non-JIT temporary allocations.
    run_raw_cuda();
    run_matx_cuda();
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Report online compilation separately; it is cached and excluded from the
    // steady-state timings below.
    const double jit_compile_ms = MeasureHostMs([&]() {
      run_matx_jit();
      BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    });

    const auto median_ms = MeasureInterleavedMedianMs(
        stream, iterations, run_raw_cuda, run_matx_cuda, run_matx_jit);
    const double raw_cuda_ms = median_ms[0];
    const double matx_cuda_ms = median_ms[1];
    const double matx_jit_ms = median_ms[2];

#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
    // Case 4 (optional): a configured host FFT backend lets the identical MatX
    // expression run across all available CPU threads. Cap host repetitions
    // independently because this large batched FFT can take much longer on CPU.
    matx::AllThreadsHostExecutor host_exec{};
    auto run_matx_host = [&]() { (matx_host_out = pipeline).run(host_exec); };
    const int host_iterations = std::min(iterations, kMaxHostIterations);
    const double matx_host_ms =
        MeasureHostMedianMs(host_iterations, run_matx_host);
#endif

    // Leave each output populated, then verify that the concise/fused paths still
    // agree with the handwritten CUDA + cuFFT reference within FP32 tolerance.
    run_raw_cuda();
    run_matx_cuda();
    run_matx_jit();
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));

    const ErrorSummary cuda_error = Compare(
        raw_cuda_out.Data(), matx_cuda_out.Data(), element_count);
    const ErrorSummary jit_error = Compare(
        raw_cuda_out.Data(), matx_jit_out.Data(), element_count);
#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
    const ErrorSummary host_error = Compare(
        raw_cuda_out.Data(), matx_host_out.Data(), element_count);
#endif
    constexpr double tolerance = 2.0e-3;
    if (cuda_error.max_rel > tolerance || jit_error.max_rel > tolerance) {
      throw std::runtime_error("output validation exceeded relative tolerance");
    }
#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
    constexpr double host_tolerance = 5.0e-3;
    if (host_error.max_rel > host_tolerance) {
      throw std::runtime_error("host output validation exceeded relative tolerance");
    }
#endif

    cudaDeviceProp properties{};
    BENCH_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cout << "Normalize -> batched FFT -> calibration -> magnitude -> dB\n"
              << "GPU: " << properties.name << "\n"
              << "Shape: " << batches << " x " << fft_size
              << ", iterations/trial: " << iterations
              << ", trials: " << kTrials << " (median)\n\n"
              << std::left << std::setw(28) << "Implementation" << std::right
              << std::setw(14) << "Time" << std::setw(21) << "Throughput"
              << std::setw(10) << "Speedup\n";
    PrintResult("CUDA kernels + cuFFT", raw_cuda_ms, element_count, raw_cuda_ms);
    PrintResult("MatX cudaExecutor", matx_cuda_ms, element_count, raw_cuda_ms);
    PrintResult("MatX CUDAJITExecutor", matx_jit_ms, element_count, raw_cuda_ms);
#if defined(MATX_EN_NVPL)
    PrintHostResult("NVIDIA NVPL", matx_host_ms, element_count,
                    host_exec.GetNumThreads(), host_iterations);
#elif defined(MATX_EN_X86_FFTW)
    PrintHostResult("FFTW", matx_host_ms, element_count,
                    host_exec.GetNumThreads(), host_iterations);
#endif
    std::cout << "\nCUDAJITExecutor first run (compile + execute): "
              << std::fixed << std::setprecision(2) << jit_compile_ms << " ms\n"
              << "Maximum relative error vs CUDA + cuFFT: cudaExecutor="
              << std::scientific << cuda_error.max_rel
              << ", CUDAJITExecutor=" << jit_error.max_rel << "\n";
#if defined(MATX_EN_NVPL) || defined(MATX_EN_X86_FFTW)
    std::cout << "Maximum relative error vs CUDA + cuFFT: host="
              << std::scientific << host_error.max_rel << "\n";
#endif

    BENCH_CUFFT_CHECK(cufftDestroy(plan));
    BENCH_CUDA_CHECK(cudaFree(normalized));
    BENCH_CUDA_CHECK(cudaFree(fft_output));
    BENCH_CUDA_CHECK(cudaStreamDestroy(stream));
  }
  catch (const std::exception &error) {
    std::cerr << "fft_benchmark: " << error.what() << "\n";
    return 1;
  }

  return 0;
#endif
}
