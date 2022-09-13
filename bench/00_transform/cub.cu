#include "matx.h"
#include <nvbench/nvbench.cuh>


using namespace matx;

//
// types of data to run tests on
//
// using sort_types = nvbench::type_list<matxFp16Complex, cuda::std::complex<float>, cuda::std::complex<double>, float, double>;
using sort_types = nvbench::type_list<double>;

////////////////////////////////////////////////////////////////////////////////
///
///\brief benchmark for 1D convolutions
///
///\param param in
///
////////////////////////////////////////////////////////////////////////////////
template <typename ValueType>
void sort1d_range(
                 nvbench::state &state,
                 nvbench::type_list<ValueType>
                 )
{
  const int dataSize = static_cast<int>(state.get_int64("Tensor Size"));

  auto sortedData = matx::make_tensor<ValueType>({dataSize});
  auto randomData = matx::make_tensor<ValueType>({dataSize});

  randomGenerator_t < ValueType > randData(dataSize, 0);

  auto randTensorView = randData.template GetTensorView<sortedData.Rank()>(sortedData.Shape(), NORMAL);

  (randomData = randTensorView).run();

  state.exec( [&sortedData, &randomData](nvbench::launch &launch) { matx::sort(sortedData, randomData, SORT_DIR_ASC, launch.get_stream()); });





}

NVBENCH_BENCH_TYPES(sort1d_range, NVBENCH_TYPE_AXES(sort_types))
  .add_int64_power_of_two_axis("Tensor Size", nvbench::range(7, 12, 1));