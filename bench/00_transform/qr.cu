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

  int batch = state.get_int64("batch");
  int m = state.get_int64("rows");
  int n = state.get_int64("cols");

  auto A = make_tensor<AType>({batch, m, n});
  auto Q = make_tensor<AType>({batch, m, m});
  auto R = make_tensor<AType>({batch, m, n});
  
  A.PrefetchDevice(stream);
  Q.PrefetchDevice(stream);
  R.PrefetchDevice(stream);
  
  (A = random<float>({batch, m, n}, NORMAL)).run(stream);

  // warm up
  nvtxRangePushA("Warmup");
  qr(Q, R, A, stream);

  cudaDeviceSynchronize();
  nvtxRangePop();

  MATX_NVTX_START_RANGE( "Exec", matx_nvxtLogLevels::MATX_NVTX_LOG_ALL, 1 )
  state.exec(
   [&Q, &R, &A](nvbench::launch &launch) {
      qr(Q, R, A, launch.get_stream()); });
  MATX_NVTX_END_RANGE( 1 )

}
NVBENCH_BENCH_TYPES(qr_batch, NVBENCH_TYPE_AXES(qr_types))
  .add_int64_axis("cols", {4, 16, 64})
  .add_int64_axis("rows", {4})
  .add_int64_axis("batch", {3000});

