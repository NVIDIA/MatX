#pragma once

#include "matx_tensor.h"
#include "matx_type_utils.h"
#include <complex>
#include <cuda.h>
#include <iomanip>
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace matx {
template <typename T, int RANK> class tensor_t;
}

namespace matx {

template <typename T, int RANK, typename ... Args>
__global__ void PrintKernel(tensor_t<T, RANK> v, Args ...dims)
{
  v.InternalPrint(dims...);
}


} // namespace matx
