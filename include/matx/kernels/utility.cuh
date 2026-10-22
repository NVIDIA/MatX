#pragma once

#include <cuda_runtime.h>


namespace matx {

template <typename Tensor, typename ... Args>
__global__ void PrintKernel(Tensor v, Args ...dims)
{
  v.InternalPrint(dims...);
}


} // namespace matx
