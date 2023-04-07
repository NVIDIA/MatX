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
  auto t4out    = make_tensor<ValueType>({1,10845,8,16});
  t4.PrefetchDevice(0);
  t4out.PrefetchDevice(0);

  softmax(t4out, t4, {3});

  state.exec( 
    [&t4, &t4out](nvbench::launch &launch) {
      matx::softmax(t4out, t4, (cudaStream_t)launch.get_stream());
    });
}
NVBENCH_BENCH_TYPES(softmax, NVBENCH_TYPE_AXES(softmax_types));


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
  const int size0 = static_cast<int>(state.get_int64("Size0"));
  const int size1 = static_cast<int>(state.get_int64("Size1"));
  const int size2 = static_cast<int>(state.get_int64("Size2"));
  const int size3 = static_cast<int>(state.get_int64("Size3"));

  auto t1 = make_tensor<ValueType>({size3});
  auto t4 = make_tensor<ValueType>({size3, size2, size1, size0});

  t1.PrefetchDevice(0);
  t4.PrefetchDevice(0);

  randomGenerator_t < ValueType > randData(size0*size1*size2*size3, 0);

  auto r4 = randData.template GetTensorView<t4.Rank()>(t4.Shape(), UNIFORM);

  (t4 = r4).run();
  cudaDeviceSynchronize();

  state.exec([&t4, &t1](nvbench::launch &launch) { matx::sum(t1, t4, (cudaStream_t)launch.get_stream()); });

}

NVBENCH_BENCH_TYPES(reduce_4d, NVBENCH_TYPE_AXES(reduce_types))
  .add_int64_power_of_two_axis("Size0", nvbench::range(6, 8, 1))
  .add_int64_power_of_two_axis("Size1", nvbench::range(5, 5, 1))
  .add_int64_power_of_two_axis("Size2", nvbench::range(5, 5, 1))
  .add_int64_power_of_two_axis("Size3", nvbench::range(5, 5, 1));
