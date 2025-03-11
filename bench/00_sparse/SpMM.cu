#include "matx.h"
#include <nvbench/nvbench.cuh>

using namespace matx;

using matmul_types =
    nvbench::type_list<matxFp16, matxBf16,
                       float, double,
                       cuda::std::complex<float>,
                       cuda::std::complex<double>>;

template <typename ValueType>
void pow2_SpMM_bench(nvbench::state &state, nvbench::type_list<ValueType>)
{
  cudaExecutor exec{0};

  const index_t M = static_cast<index_t>(state.get_int64("M"));
  const index_t N = static_cast<index_t>(state.get_int64("N"));
  const index_t K = static_cast<index_t>(state.get_int64("K"));

  auto A = make_tensor<ValueType, 2>({M, K}, MATX_DEVICE_MEMORY);
  auto B = make_tensor<ValueType, 2>({K, N}, MATX_DEVICE_MEMORY);
  auto C = make_tensor<ValueType, 2>({M, N}, MATX_DEVICE_MEMORY);

  (A = diag(1.0)).run(exec);  // very simple sparse matrix
                              // with density = min(m,k) / m*k
  (B = ones()).run(exec);
  (C = zeros()).run(exec);

  auto Acoo = experimental::make_zero_tensor_coo<ValueType, index_t>({M, K}, MATX_DEVICE_MEMORY);
  (Acoo = dense2sparse(A)).run(exec);

  exec.sync();

  state.exec([&Acoo, &B, &C](nvbench::launch &launch) {
    (C = matmul(Acoo, B)).run(cudaExecutor(launch.get_stream()));
  });
}

NVBENCH_BENCH_TYPES(pow2_SpMM_bench, NVBENCH_TYPE_AXES(matmul_types))
    .add_int64_power_of_two_axis("M", nvbench::range(13, 13, 1))
    .add_int64_power_of_two_axis("N", nvbench::range(13, 13, 1))
    .add_int64_power_of_two_axis("K", nvbench::range(13, 13, 1));
