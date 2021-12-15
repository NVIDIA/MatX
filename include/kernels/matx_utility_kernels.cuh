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

template <typename Tensor, typename ... Args>
__global__ void PrintKernel(Tensor v, Args ...dims)
{
  v.InternalPrint(dims...);
}


} // namespace matx
