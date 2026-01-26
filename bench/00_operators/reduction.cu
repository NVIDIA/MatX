#include <nvbench/nvbench.cuh>
#include "matx.h"

using namespace matx;

using reduce_types = nvbench::type_list<float, double>;
using softmax_types = nvbench::type_list<float, double, matxFp16, matxBf16>;

template <typename ValueType>
void softmax(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  auto t4    = make_tensor<ValueType>({1,10845,8,16});
  auto t4out = make_tensor<ValueType>({1,10845,8,16});
  t4.PrefetchDevice(0);
  t4out.PrefetchDevice(0);

  (t4out = softmax(t4, {3})).run();

  state.exec( 
    [&t4, &t4out](nvbench::launch &launch) {
      (t4out = softmax(t4)).run((cudaStream_t)launch.get_stream());
    });
}
NVBENCH_BENCH_TYPES(softmax, NVBENCH_TYPE_AXES(softmax_types));


template <typename ValueType>
void reduce_0d_matx(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const index_t x_len = static_cast<index_t>(state.get_int64("Tensor Size"));

  state.add_element_count(x_len, "NumElements");
  state.add_global_memory_reads<ValueType>(x_len * x_len * x_len * x_len, "DataSize");
  state.add_global_memory_writes<ValueType>(1);  

  auto xvc = make_tensor<ValueType>({x_len, x_len, x_len, x_len});
  auto xv = xvc.Permute({0,1,3,2});
  auto xv2 = make_tensor<ValueType>({});
  xv.PrefetchDevice(0);

  (xv2 = matx::sum(xv)).run();

  state.exec( 
    [&xv, &xv2](nvbench::launch &launch) {
      (xv2 = matx::sum(xv)).run((cudaStream_t)launch.get_stream());
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
  const index_t x_len = static_cast<index_t>(state.get_int64("Tensor Size"));

  state.add_element_count(x_len, "NumElements");
  state.add_global_memory_reads<ValueType>(x_len * x_len * x_len * x_len, "DataSize");
  state.add_global_memory_writes<ValueType>(1);  

  auto xv = make_tensor<ValueType>({x_len, x_len, x_len, x_len});
  auto xv2 = make_tensor<ValueType>({});
  xv.PrefetchDevice(0);

  (xv2 = matx::sum(xv)).run();

  state.exec( 
    [&xv, &xv2](nvbench::launch &launch) {
      (xv2 = matx::sum(xv)).run((cudaStream_t)launch.get_stream());
    });

}
NVBENCH_BENCH_TYPES(reduce_0d_cub, NVBENCH_TYPE_AXES(reduce_types))
  .add_int64_power_of_two_axis("Tensor Size", nvbench::range(6, 7, 1));

template <typename ValueType>
void reduce_0d_cub_permute(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const index_t x_len = static_cast<index_t>(state.get_int64("Tensor Size"));

  state.add_element_count(x_len, "NumElements");
  state.add_global_memory_reads<ValueType>(x_len * x_len * x_len * x_len, "DataSize");
  state.add_global_memory_writes<ValueType>(1);  

  auto xvc = make_tensor<ValueType>({x_len, x_len, x_len, x_len});
  auto xv = xvc.Permute({0,1,3,2});

  auto xv2 = make_tensor<ValueType>({});
  xv.PrefetchDevice(0);

  cub_reduce<decltype(xv2), decltype(xv), CustomSum>(xv2, xv, 0.0f, 0);

  state.exec( 
    [&xv, &xv2](nvbench::launch &launch) {
      cub_reduce<decltype(xv2), decltype(xv), CustomSum>(xv2, xv, 0.0f, (cudaStream_t)launch.get_stream());
    });

}
NVBENCH_BENCH_TYPES(reduce_0d_cub_permute, NVBENCH_TYPE_AXES(reduce_types))
  .add_int64_power_of_two_axis("Tensor Size", nvbench::range(6, 7, 1));  


////////////////////////////////////////////////////////////////////////////////
///
///\brief benchmark for 1D Sorts across key element types
///
///\param nvbench::state &state,        benchmark state structutre
///\param nvbench::type_list<ValueType> list of reduce data types to test
///
////////////////////////////////////////////////////////////////////////////////
template <typename ValueType>
void reduce_4d(
           nvbench::state &state,
           nvbench::type_list<ValueType>
           )
{
  cudaExecutor exec{0};
  const index_t size0 = static_cast<index_t>(state.get_int64("Size0"));
  const index_t size1 = static_cast<index_t>(state.get_int64("Size1"));
  const index_t size2 = static_cast<index_t>(state.get_int64("Size2"));
  const index_t size3 = static_cast<index_t>(state.get_int64("Size3"));

  auto t1 = make_tensor<ValueType>({size3});
  auto t4 = make_tensor<ValueType>({size3, size2, size1, size0});

  t1.PrefetchDevice(0);
  t4.PrefetchDevice(0);

  (t4 = random<float>(t4.Shape(), UNIFORM)).run(exec);
  exec.sync();

  state.exec([&t4, &t1](nvbench::launch &launch) { 
    (t1 = matx::sum(t4, {1, 2, 3})).run((cudaStream_t)launch.get_stream()); });

}

NVBENCH_BENCH_TYPES(reduce_4d, NVBENCH_TYPE_AXES(reduce_types))
  .add_int64_power_of_two_axis("Size0", nvbench::range(6, 8, 1))
  .add_int64_power_of_two_axis("Size1", nvbench::range(5, 5, 1))
  .add_int64_power_of_two_axis("Size2", nvbench::range(5, 5, 1))
  .add_int64_power_of_two_axis("Size3", nvbench::range(5, 5, 1));
