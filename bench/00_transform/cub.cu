#include "matx.h"
#include <nvbench/nvbench.cuh>


using namespace matx;

//
// types of data to run tests on
//
using sort_types = nvbench::type_list< float, double>;


////////////////////////////////////////////////////////////////////////////////
///
///\brief benchmark for 1D Sorts across key element types
///
///\param nvbench::state &state,        benchmark state structutre
///\param nvbench::type_list<ValueType> list of sort data types to test
///
////////////////////////////////////////////////////////////////////////////////
template <typename ValueType>
void sort1d(
           nvbench::state &state,
           nvbench::type_list<ValueType>
           )
{
  const int dataSize = static_cast<int>(state.get_int64("Tensor Size"));

  auto sortedData = matx::make_tensor<ValueType>({dataSize});
  auto randomData = matx::make_tensor<ValueType>({dataSize});

  randomGenerator_t < ValueType > randData(dataSize, 0);

  sortedData.PrefetchDevice(0);
  randomData.PrefetchDevice(0);
  cudaDeviceSynchronize();

  auto randTensorView = randData.template GetTensorView<sortedData.Rank()>(sortedData.Shape(), NORMAL);

  (randomData = randTensorView).run();

  state.exec( [&sortedData, &randomData](nvbench::launch &launch) { matx::sort(sortedData, randomData, SORT_DIR_ASC, (cudaStream_t)launch.get_stream()); });

}

NVBENCH_BENCH_TYPES(sort1d, NVBENCH_TYPE_AXES(sort_types))
  .add_int64_power_of_two_axis("Tensor Size", nvbench::range(7, 14, 1));


////////////////////////////////////////////////////////////////////////////////
///
///\brief benchmark for 2D Sorts across key element types
///
///\param nvbench::state &state,        benchmark state structutre
///\param nvbench::type_list<ValueType> list of sort data types to test
///
////////////////////////////////////////////////////////////////////////////////
template <typename ValueType>
void sort2d(
           nvbench::state &state,
           nvbench::type_list<ValueType>
           )
{


  // state.set_blocking_kernel_timeout(200);

  const int dim1Size = static_cast<int>(state.get_int64("Tensor Dim1 Size"));
  const int dim2Size = static_cast<int>(state.get_int64("Tensor Dim2 Size"));

  auto sortedData = matx::make_tensor<ValueType>({dim1Size, dim2Size});
  auto randomData = matx::make_tensor<ValueType>({dim1Size, dim2Size});

  randomGenerator_t < ValueType > randData(dim1Size*dim2Size, 0);

  auto randTensorView = randData.template GetTensorView<sortedData.Rank()>(sortedData.Shape(), NORMAL);

  (randomData = randTensorView).run();

  state.exec( [&sortedData, &randomData](nvbench::launch &launch) { matx::sort(sortedData, randomData, SORT_DIR_ASC, (cudaStream_t)launch.get_stream()); });

}

NVBENCH_BENCH_TYPES(sort2d, NVBENCH_TYPE_AXES(sort_types))
  .add_int64_power_of_two_axis("Tensor Dim1 Size", nvbench::range(3, 8, 1))
  .add_int64_power_of_two_axis("Tensor Dim2 Size", nvbench::range(7, 14, 1));
