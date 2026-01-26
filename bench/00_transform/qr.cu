#include "matx.h"
#include <nvbench/nvbench.cuh>
#include "matx/core/nvtx.h"

using namespace matx;

using qr_types =
    nvbench::type_list<float, double, cuda::std::complex<float>, cuda::std::complex<double>>;

/* QR benchmarks */
template <typename ValueType>
void qr_batch(nvbench::state &state,
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

  auto A = make_tensor<AType>({batch, m, n});
  auto Q = make_tensor<AType>({batch, m, m});
  auto R = make_tensor<AType>({batch, m, n});
  
  (A = random<float>({batch, m, n}, NORMAL)).run(exec);

  // warm up
  nvtxRangePushA("Warmup");
  (mtie(Q, R) = qr(A)).run(exec);

  exec.sync();
  nvtxRangePop();

  MATX_NVTX_START_RANGE( "Exec", matx_nvxtLogLevels::MATX_NVTX_LOG_ALL, 1 )
  state.exec(
   [&Q, &R, &A](nvbench::launch &launch) {
      (mtie(Q, R) = qr(A)).run(cudaExecutor{launch.get_stream()}); });
  MATX_NVTX_END_RANGE( 1 )

}
NVBENCH_BENCH_TYPES(qr_batch, NVBENCH_TYPE_AXES(qr_types))
  .add_int64_axis("cols", {4, 16, 64})
  .add_int64_axis("rows", {4})
  .add_int64_axis("batch", {3000});

