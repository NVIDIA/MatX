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

  int batch = state.get_int64("batch");
  int m = state.get_int64("rows");
  int n = state.get_int64("cols");

  int r = std::min(n,m);
  auto A = make_tensor<AType>({batch, m, n});
  auto U = make_tensor<AType>({batch, m, r});
  auto VT = make_tensor<AType>({batch, r, n});
  auto S = make_tensor<SType>({batch, r});

  randomGenerator_t<AType> gen(batch*m*n,0);
  auto x0 = gen.GetTensorView({batch, r}, NORMAL);

  int iterations = 10;

  auto random = gen.GetTensorView({batch, m, n}, NORMAL);
  (A = random).run(stream);
  
  A.PrefetchDevice(stream);
  U.PrefetchDevice(stream);
  S.PrefetchDevice(stream);
  VT.PrefetchDevice(stream);

  (U = 0).run(stream);
  (S = 0).run(stream);
  (VT = 0).run(stream);

  // warm up
  nvtxRangePushA("Warmup");
  svdpi(U, S, VT, A, x0, iterations, stream, r);
  cudaDeviceSynchronize();
  nvtxRangePop();

  MATX_NVTX_START_RANGE( "Exec", matx_nvxtLogLevels::MATX_NVTX_LOG_ALL, 1 )
  state.exec(
   [&U, &S, &VT, &A, &x0, &iterations, &r](nvbench::launch &launch) {
      svdpi(U, S, VT, A, x0, iterations, launch.get_stream(), r); });
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

  int batch = state.get_int64("batch");
  int m = state.get_int64("rows");
  int n = state.get_int64("cols");

  int r = std::min(n,m);
  auto A = make_tensor<AType>({batch, m, n});
  auto U = make_tensor<AType>({batch, m, r});
  auto VT = make_tensor<AType>({batch, r, n});
  auto S = make_tensor<SType>({batch, r});

  int iterations = 10;
  
  randomGenerator_t<AType> gen(A.TotalSize(),0);

  auto random = gen.GetTensorView({batch, m, n}, NORMAL);
  (A = random).run(stream);
  
  A.PrefetchDevice(stream);
  U.PrefetchDevice(stream);
  S.PrefetchDevice(stream);
  VT.PrefetchDevice(stream);

  (U = 0).run(stream);
  (S = 0).run(stream);
  (VT = 0).run(stream);

  // warm up
  nvtxRangePushA("Warmup");
  svdbpi(U, S, VT, A, iterations, stream);
  cudaDeviceSynchronize();
  nvtxRangePop();

  MATX_NVTX_START_RANGE( "Exec", matx_nvxtLogLevels::MATX_NVTX_LOG_ALL, 1 )
  state.exec(
   [&U, &S, &VT, &A, &iterations, &r](nvbench::launch &launch) {
      svdbpi(U, S, VT, A, iterations, launch.get_stream()); });
  MATX_NVTX_END_RANGE( 1 )
}

NVBENCH_BENCH_TYPES(svdbpi_batch, NVBENCH_TYPE_AXES(svd_types))
  .add_int64_axis("cols", {4, 16, 64})
  .add_int64_axis("rows", {4})
  .add_int64_axis("batch", {3000});

