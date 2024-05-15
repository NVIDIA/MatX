#include "matx.h"
#include <nvbench/nvbench.cuh>

using namespace matx;

using fft_types =
    nvbench::type_list<cuda::std::complex<float>, cuda::std::complex<double>>;

/* FFT benchmarks */
template <typename ValueType>
void fft1d_no_batches_pow_2(nvbench::state &state,
                            nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const int x_len = static_cast<int>(state.get_int64("FFT size"));

  tensor_t<ValueType, 1> xv{{x_len}};
  xv.PrefetchDevice(0);

  state.exec(
      [&xv](nvbench::launch &launch) { (xv = fft(xv)).run(cudaExecutor(launch.get_stream())); });
}
NVBENCH_BENCH_TYPES(fft1d_no_batches_pow_2, NVBENCH_TYPE_AXES(fft_types))
    .add_int64_power_of_two_axis("FFT size", nvbench::range(10, 18, 1));

/* GEMM benchmarks */
template <typename ValueType>
void fft1d_no_batches_non_pow_2(nvbench::state &state,
                                nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const int x_len = static_cast<int>(state.get_int64("FFT size"));

  tensor_t<ValueType, 1> xv{{x_len}};
  xv.PrefetchDevice(0);

  state.exec(
      [&xv](nvbench::launch &launch) { (xv = fft(xv)).run(cudaExecutor(launch.get_stream())); });
}
NVBENCH_BENCH_TYPES(fft1d_no_batches_non_pow_2, NVBENCH_TYPE_AXES(fft_types))
    .add_int64_axis("FFT size", nvbench::range(50000, 250000, 50000));

template <typename ValueType>
void fft1d_batches_pow_2(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const int x_len = static_cast<int>(state.get_int64("FFT size"));

  tensor_t<ValueType, 3> xv{{10, 10, x_len}};
  xv.PrefetchDevice(0);

  state.exec(
      [&xv](nvbench::launch &launch) { (xv = fft(xv)).run(cudaExecutor(launch.get_stream())); });
}
NVBENCH_BENCH_TYPES(fft1d_batches_pow_2, NVBENCH_TYPE_AXES(fft_types))
    .add_int64_power_of_two_axis("FFT size", nvbench::range(10, 18, 1));