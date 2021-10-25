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

template <typename T, int RANK>
__global__ void PrintKernel(tensor_t<T, RANK> v)
{
  v.InternalPrint();
}

template <typename T, int RANK>
__global__ void PrintKernel(tensor_t<T, RANK> v, index_t k)
{
  v.InternalPrint(k);
}

template <typename T, int RANK>
__global__ void PrintKernel(tensor_t<T, RANK> v, const index_t k,
                            const index_t l)
{
  v.InternalPrint(k, l);
}

template <typename T, int RANK>
__global__ void PrintKernel(tensor_t<T, RANK> v, const index_t j,
                            const index_t k, const index_t l)
{
  v.InternalPrint(j, k, l);
}

template <typename T, int RANK>
__global__ void PrintKernel(tensor_t<T, RANK> v, const index_t i,
                            const index_t j, const index_t k, const index_t l)
{
  v.InternalPrint(i, j, k, l);
}

} // namespace matx
