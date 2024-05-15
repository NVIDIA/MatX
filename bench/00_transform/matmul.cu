#include "matx.h"
#include <nvbench/nvbench.cuh>

using namespace matx;

using matmul_types =
    nvbench::type_list<float, double, cuda::std::complex<float>,
                       cuda::std::complex<double>, matxFp16, matxFp16Complex>;

/* matrix multiplication benchmarks */
template <typename ValueType>
void pow2_matmul_bench(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const index_t M = static_cast<index_t>(state.get_int64("M"));
  const index_t N = static_cast<index_t>(state.get_int64("N"));
  const index_t K = static_cast<index_t>(state.get_int64("K"));

  tensor_t<ValueType, 2> av{{M, K}};
  tensor_t<ValueType, 2> bv{{K, N}};
  tensor_t<ValueType, 2> cv{{M, N}};

  av.PrefetchDevice(0);
  bv.PrefetchDevice(0);
  cv.PrefetchDevice(0);

  // Number of int32s in 256 MiB:
  // const std::size_t num_values = M * N * K;

  // Report throughput stats:

  state.exec([&av, &bv, &cv](nvbench::launch &launch) {
    (cv = matmul(av, bv)).run(cudaExecutor(launch.get_stream()));
  });

  auto seconds =
      state.get_summary("Batch GPU").get_float64("value");
  auto &summ = state.add_summary("TFLOPS");

  summ.set_string("hint", "item_rate");
  summ.set_string("short_name", "TFLOPS");
  summ.set_string("description", "Trillions of operations per second");

  if constexpr (is_complex_v<ValueType>) {
    summ.set_float64("value", static_cast<double>(8 * M * N * K - 2 * M * N) /
                                  seconds / 1e12);
  }
  else {
    summ.set_float64("value", static_cast<double>(2 * M * N * K - M * N) /
                                  seconds / 1e12);
  }
}

NVBENCH_BENCH_TYPES(pow2_matmul_bench, NVBENCH_TYPE_AXES(matmul_types))
    .add_int64_power_of_two_axis("M", nvbench::range(12, 13, 1))
    .add_int64_power_of_two_axis("N", nvbench::range(12, 13, 1))
    .add_int64_power_of_two_axis("K", nvbench::range(12, 13, 1));
