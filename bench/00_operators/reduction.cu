#include <nvbench/nvbench.cuh>
#include "matx.h"

using namespace matx;

using reduce_types = nvbench::type_list<float, double>;


template <typename ValueType>
void reduce_0d_matx(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const long long int x_len = state.get_int64("Tensor Size");

  state.add_element_count(x_len, "NumElements");
  state.add_global_memory_reads<ValueType>(x_len * x_len * x_len * x_len, "DataSize");
  state.add_global_memory_writes<ValueType>(1);  

  auto xvc = make_tensor<ValueType>({x_len, x_len, x_len, x_len});
  auto xv = xvc.Permute({0,1,3,2});
  auto xv2 = make_tensor<ValueType>();
  xv.PrefetchDevice(0);

  matx::sum(xv2, xv);

  state.exec( 
    [&xv, &xv2](nvbench::launch &launch) {
      matx::sum(xv2, xv, (cudaStream_t)launch.get_stream());
    });

}
NVBENCH_BENCH_TYPES(reduce_0d_matx, NVBENCH_TYPE_AXES(reduce_types))
  .add_int64_power_of_two_axis("Tensor Size", nvbench::range(6, 7, 1));

struct CustomSum
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <typename ValueType>
void reduce_0d_cub(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const long long int x_len = state.get_int64("Tensor Size");

  state.add_element_count(x_len, "NumElements");
  state.add_global_memory_reads<ValueType>(x_len * x_len * x_len * x_len, "DataSize");
  state.add_global_memory_writes<ValueType>(1);  

  auto xv = make_tensor<ValueType>({x_len, x_len, x_len, x_len});
  auto xv2 = make_tensor<ValueType>();
  xv.PrefetchDevice(0);

  sum(xv2, xv, 0);

  state.exec( 
    [&xv, &xv2](nvbench::launch &launch) {
      sum(xv2, xv, (cudaStream_t)launch.get_stream());
    });

}
NVBENCH_BENCH_TYPES(reduce_0d_cub, NVBENCH_TYPE_AXES(reduce_types))
  .add_int64_power_of_two_axis("Tensor Size", nvbench::range(6, 7, 1));

template <typename ValueType>
void reduce_0d_cub_permute(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const long long int x_len = state.get_int64("Tensor Size");

  state.add_element_count(x_len, "NumElements");
  state.add_global_memory_reads<ValueType>(x_len * x_len * x_len * x_len, "DataSize");
  state.add_global_memory_writes<ValueType>(1);  

  auto xvc = make_tensor<ValueType>({x_len, x_len, x_len, x_len});
  auto xv = xvc.Permute({0,1,3,2});

  auto xv2 = make_tensor<ValueType>();
  xv.PrefetchDevice(0);

  cub_reduce<decltype(xv2), decltype(xv), CustomSum>(xv2, xv, 0.0f, 0);

  state.exec( 
    [&xv, &xv2](nvbench::launch &launch) {
      cub_reduce<decltype(xv2), decltype(xv), CustomSum>(xv2, xv, 0.0f, (cudaStream_t)launch.get_stream());
    });

}
NVBENCH_BENCH_TYPES(reduce_0d_cub_permute, NVBENCH_TYPE_AXES(reduce_types))
  .add_int64_power_of_two_axis("Tensor Size", nvbench::range(6, 7, 1));  

