#include "matx.h"
#include <nvbench/nvbench.cuh>
#include "matx/core/nvtx.h"

using namespace matx;

using svd_types =
    nvbench::type_list<float, double, cuda::std::complex<float>, cuda::std::complex<double>>;

/* SVD benchmarks */
template <typename ValueType>
void svdpi_batch(nvbench::state &state,
                            nvbench::type_list<ValueType>)
{
  using AType = ValueType;
  using SType = typename inner_op_type_t<AType>::type;

  cudaStream_t stream = 0;
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
  cudaExecutor exec{stream};

  const index_t batch = static_cast<index_t>(state.get_int64("batch"));
  const index_t m = static_cast<index_t>(state.get_int64("rows"));
  const index_t n = static_cast<index_t>(state.get_int64("cols"));

  index_t r = std::min(n,m);
  auto A = make_tensor<AType>({batch, m, n});
  auto U = make_tensor<AType>({batch, m, r});
  auto VT = make_tensor<AType>({batch, r, n});
  auto S = make_tensor<SType>({batch, r});

  int iterations = 10;

  (A = random<float>({batch, m, n}, NORMAL)).run(exec);

  (U = 0).run(exec);
  (S = 0).run(exec);
  (VT = 0).run(exec);
  auto x0 = random<float>({batch, r}, NORMAL);

  // warm up
  nvtxRangePushA("Warmup");
  (mtie(U, S, VT) = svdpi(A, x0, iterations, r)).run(exec);
  exec.sync();
  nvtxRangePop();

  MATX_NVTX_START_RANGE( "Exec", matx_nvxtLogLevels::MATX_NVTX_LOG_ALL, 1 )
  state.exec(
   [&U, &S, &VT, &A, &x0, &iterations, &r](nvbench::launch &launch) {
      (mtie(U, S, VT) = svdpi(A, x0, iterations, r)).run(cudaExecutor{launch.get_stream()}); });
  MATX_NVTX_END_RANGE( 1 )

}
NVBENCH_BENCH_TYPES(svdpi_batch, NVBENCH_TYPE_AXES(svd_types))
  .add_int64_axis("cols", {4, 16, 64})
  .add_int64_axis("rows", {4})
  .add_int64_axis("batch", {3000});


template <typename ValueType>
void svdbpi_batch(nvbench::state &state,
                            nvbench::type_list<ValueType>)
{
  using AType = ValueType;
  using SType = typename inner_op_type_t<AType>::type;

  cudaStream_t stream = 0;
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
  cudaExecutor exec{stream};
  const index_t batch = static_cast<index_t>(state.get_int64("batch"));
  const index_t m = static_cast<index_t>(state.get_int64("rows"));
  const index_t n = static_cast<index_t>(state.get_int64("cols"));

  index_t r = std::min(n,m);
  auto A = make_tensor<AType>({batch, m, n});
  auto U = make_tensor<AType>({batch, m, r});
  auto VT = make_tensor<AType>({batch, r, n});
  auto S = make_tensor<SType>({batch, r});

  int iterations = 10;

  (A = random<float>({batch, m, n}, NORMAL)).run(exec);

  (U = 0).run(exec);
  (S = 0).run(exec);
  (VT = 0).run(exec);

  // warm up
  nvtxRangePushA("Warmup");
  (mtie(U, S, VT) = svdbpi(A, iterations)).run(exec);
  exec.sync();
  nvtxRangePop();

  MATX_NVTX_START_RANGE( "Exec", matx_nvxtLogLevels::MATX_NVTX_LOG_ALL, 1 )
  state.exec(
   [&U, &S, &VT, &A, &iterations](nvbench::launch &launch) {
      (mtie(U, S, VT) = svdbpi(A, iterations)).run(cudaExecutor{launch.get_stream()}); });
  MATX_NVTX_END_RANGE( 1 )
}

NVBENCH_BENCH_TYPES(svdbpi_batch, NVBENCH_TYPE_AXES(svd_types))
  .add_int64_axis("cols", {4, 16, 64})
  .add_int64_axis("rows", {4})
  .add_int64_axis("batch", {3000});

