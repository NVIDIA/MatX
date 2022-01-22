#include <nvbench/nvbench.cuh>
#include "matx.h"
#include "matx_einsum.h"

#if ENABLE_CUTENSOR

using namespace matx;

using einsum_permute_types = nvbench::type_list<float, double, cuda::std::complex<float>, cuda::std::complex<double>>;
template <typename ValueType>
void einsum_permute(nvbench::state &state, nvbench::type_list<ValueType>)
{
  auto x = make_tensor<ValueType>({1000,200,6,300});
  auto y = make_tensor<ValueType>({300,1000,6,200});

  state.add_element_count(x.TotalSize(), "NumElements");
  state.add_global_memory_reads<ValueType>(x.TotalSize(), "DataSize");
  state.add_global_memory_writes<ValueType>(x.TotalSize());    

  x.PrefetchDevice(0);

  cutensor::einsum(y, "ijkl->likj", 0, x);

  state.exec( 
    [&x, &y](nvbench::launch &launch) {
        cutensor::einsum(y, "ijkl->likj", (cudaStream_t)launch.get_stream(), x);
    });
}


NVBENCH_BENCH_TYPES(einsum_permute, NVBENCH_TYPE_AXES(einsum_permute_types));

#endif
