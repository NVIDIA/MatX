#include <nvbench/nvbench.cuh>
#include "matx.h"

using namespace matx;

using vec_add_types = nvbench::type_list<float, double, cuda::std::complex<float>, cuda::std::complex<double>>;

/* Vector adding benchmarks */
template <typename ValueType>
void vector_add(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const int x_len = static_cast<int>(state.get_int64("Vector size"));

  state.add_element_count(x_len, "NumElements");
  state.add_global_memory_reads<ValueType>(2*x_len, "DataSize");
  state.add_global_memory_writes<ValueType>(x_len);  


  tensor_t<ValueType, 1> xv{{x_len}};
  tensor_t<ValueType, 1> xv2{{x_len}};
  xv.PrefetchDevice(0);
  (xv = xv + xv2).run();

  state.exec( 
    [&xv, &xv2](nvbench::launch &launch) {
      (xv = xv + xv2).run((cudaStream_t)launch.get_stream());
    });

}

NVBENCH_BENCH_TYPES(vector_add, NVBENCH_TYPE_AXES(vec_add_types))
  .add_int64_power_of_two_axis("Vector size", nvbench::range(22, 28, 1));
