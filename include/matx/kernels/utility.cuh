#pragma once

#include <complex>
#include <cuda.h>
#include <iomanip>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "matx/core/tensor.h"
#include "matx/core/type_utils.h"


namespace matx {

template <typename Tensor, typename ... Args>
__global__ void PrintKernel(Tensor v, Args ...dims)
{
  v.InternalPrint(dims...);
}


} // namespace matx
