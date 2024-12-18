
#pragma once

#include <complex>
#include <cuda.h>
#include <iomanip>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "cuComplex.h"
#include "matx/core/type_utils.h"

namespace matx {

#ifdef __CUDACC__

namespace matx_transpose_detail {
  // Tile dims for one block
  constexpr size_t TILE_DIM = 32;
}
using namespace matx_transpose_detail;

/* Out of place. Adapted from:
   https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/. Works
   for both square and rectangular matrices. */
template <typename OutputTensor, typename InputTensor>
__global__ void transpose_kernel_oop(OutputTensor out,
                                     const InputTensor in)
{
  using T = typename OutputTensor::value_type;
  constexpr int RANK = OutputTensor::Rank();

  extern __shared__ float
      tile[]; // Need to swap complex types also, so cast when needed
  T *shm_tile = reinterpret_cast<T *>(&tile[0]);
  index_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  index_t y = blockIdx.y * TILE_DIM + threadIdx.y;

  if constexpr (RANK == 2) {
    if (x < in.Size(RANK - 1) && y < in.Size(RANK - 2)) {
      shm_tile[threadIdx.y * (TILE_DIM + 1) + threadIdx.x] = in(y, x);
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (y < out.Size(RANK - 2) && x < out.Size(RANK - 1)) {
      out(y, x) = shm_tile[threadIdx.x * (TILE_DIM + 1) + threadIdx.y];
    }
  }
  else if constexpr (RANK == 3) {
    index_t z = blockIdx.z;

    if (x < in.Size(RANK - 1) && y < in.Size(RANK - 2)) {
      shm_tile[threadIdx.y * (TILE_DIM + 1) + threadIdx.x] = in(z, y, x);
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (y < out.Size(RANK - 2) && x < out.Size(RANK - 1)) {
      out(z, y, x) = shm_tile[threadIdx.x * (TILE_DIM + 1) + threadIdx.y];
    }
  }
  else if constexpr (RANK == 4) {
    index_t z = blockIdx.z % in.Size(1);
    index_t w = blockIdx.z / in.Size(1);

    if (x < in.Size(RANK - 1) && y < in.Size(RANK - 2)) {
      shm_tile[threadIdx.y * (TILE_DIM + 1) + threadIdx.x] = in(w, z, y, x);
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (y < out.Size(RANK - 2) && x < out.Size(RANK - 1)) {
      out(w, z, y, x) = shm_tile[threadIdx.x * (TILE_DIM + 1) + threadIdx.y];
    }
  }
}
#endif

}; // namespace matx
