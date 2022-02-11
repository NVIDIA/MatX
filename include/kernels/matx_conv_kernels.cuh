
#pragma once

#include "cuComplex.h"
#include "matx_type_utils.h"
#include <complex>
#include <cuda.h>
#include <iomanip>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#define BLOCK_SIZE_NON_RECURSIVE 1024

namespace matx {

typedef enum {
  MATX_C_MODE_FULL, // Default. Keep all elements of ramp up/down
  MATX_C_MODE_SAME, // Only keep elements where entire filter was present
  MATX_C_MODE_VALID
} matxConvCorrMode_t;

typedef enum {
  MATX_C_METHOD_DIRECT,
  MATX_C_METHOD_FFT,
  MATX_C_METHOD_AUTO,
} matxConvCorrMethod_t;

#ifdef __CUDACC__  
template <typename OutType, typename InType, typename FilterType>
__global__ void Conv1D(OutType d_out, InType d_in, FilterType d_filter,
                       index_t signal_len, index_t filter_len,
                       matxConvCorrMode_t mode)
{
  extern __shared__ float s_exch[]; // Filter + halo
  using ftype_strip = typename FilterType::scalar_type;
  using intype_strip = typename InType::scalar_type;
  using outtype_strip = typename OutType::scalar_type;

  // Adjustment to keep base shm size as float, but if the filter is complex we
  // need to adjust it
  constexpr float filt_size_adj =
      static_cast<float>(sizeof(ftype_strip)) / sizeof(s_exch[0]);

  ftype_strip *s_filter = reinterpret_cast<ftype_strip *>(&s_exch[0]);
  intype_strip *s_data;

  // If the data type has a higher alignment type than the filter, we need to
  // adjust our shm pointer
  if constexpr (std::alignment_of_v < intype_strip >>
                std::alignment_of_v<ftype_strip>) {
    s_data =
        matx::detail::AlignAddr<intype_strip>((uint8_t *)&s_exch[static_cast<index_t>(
            filter_len * filt_size_adj)]); // Start data portion after 2x the
                                           // filter to remove conditionals and
                                           // multiply by 0
  }
  else {
    s_data = reinterpret_cast<intype_strip *>(&s_exch[static_cast<index_t>(
        filter_len *
        filt_size_adj)]); // Start data portion after 2x the filter to
                          // remove conditionals and multiply by 0
  }

  index_t full_len = signal_len + filter_len - 1;

  // This is the location that is written in memory. Note that there will be
  // duplicate tids based on this formula, but not all threads write out to
  // memory. Some are only there to fetch data, while others both fetch and
  // compute output
  const index_t tid =
      static_cast<index_t>(blockIdx.x) * (blockDim.x - filter_len + 1) +
      threadIdx.x;
  int offset = tid - filter_len + 1;

  outtype_strip val = 0;
  [[maybe_unused]] index_t bdims[2];

  // Zero out shared memory since it's used later to index into where we want
  // 0-valued taps
  for (index_t i = threadIdx.x; i < filter_len + blockDim.x; i += blockDim.x) {
    s_data[i] = 0.0;
  }

  if constexpr (d_in.Rank() == 4) {
    bdims[0] = blockIdx.z / d_in.Size(1);
    bdims[1] = blockIdx.z - (bdims[0] * d_in.Size(1));
  }

  __syncthreads();

  if (threadIdx.x < filter_len) {
    s_filter[threadIdx.x] = d_filter(threadIdx.x);
  }

  __syncthreads();

  if (blockIdx.x == 0) {
    // We want all blocks to process uniformly, so the first block's last few
    // threads are idle to match what all other blocks do
    s_data[threadIdx.x] = 0;

    __syncthreads();

    // The first block just grabs all the data from the start of the sequence
    if (threadIdx.x < signal_len &&
        (threadIdx.x < blockDim.x - filter_len + 1)) {
      if constexpr (d_in.Rank() == 4) {
        s_data[threadIdx.x + filter_len - 1] =
            d_in(bdims[0], bdims[1], blockIdx.y, threadIdx.x);
      }
      else if constexpr (d_in.Rank() == 3) {
        s_data[threadIdx.x + filter_len - 1] =
            d_in(blockIdx.z, blockIdx.y, threadIdx.x);
      }
      else if constexpr (d_in.Rank() == 2) {
        s_data[threadIdx.x + filter_len - 1] = d_in(blockIdx.y, threadIdx.x);
      }
      else if constexpr (d_in.Rank() == 1) {
        s_data[threadIdx.x + filter_len - 1] = d_in(threadIdx.x);
      }
    }
  }
  else if (offset > 0 && offset < signal_len) {
    // Each block processes blockDim.x-filt_len+1 samples, but needs to fetch
    // all blockDim.x worth
    if constexpr (d_in.Rank() == 4) {
      s_data[threadIdx.x] = d_in(bdims[0], bdims[1], blockIdx.y, offset);
    }
    else if constexpr (d_in.Rank() == 3) {
      s_data[threadIdx.x] = d_in(blockIdx.z, blockIdx.y, offset);
    }
    else if constexpr (d_in.Rank() == 2) {
      s_data[threadIdx.x] = d_in(blockIdx.y, offset);
    }
    else if constexpr (d_in.Rank() == 1) {
      // For all blocks > 0 we need to grab only as much as each block outputs
      s_data[threadIdx.x] = d_in(offset);
    }
  }

  __syncthreads();

  // Even though all threads in the block fetched data, there is only enough
  // data in shared memory for blockDim-filt_len+1 to operate on. The rest sit
  // idle through this process.
  if (tid < full_len && (threadIdx.x < blockDim.x - filter_len + 1)) {
#pragma unroll
    for (index_t r = 0; r < filter_len; r++) {
      val = val + s_filter[r] * s_data[threadIdx.x + filter_len - 1 - r];
    }

    if (mode == MATX_C_MODE_FULL) {
      if constexpr (d_in.Rank() == 4) {
        d_out(bdims[0], bdims[1], blockIdx.y, tid) = val;
      }
      else if constexpr (d_in.Rank() == 3) {
        d_out(blockIdx.z, blockIdx.y, tid) = val;
      }
      else if constexpr (d_in.Rank() == 2) {
        d_out(blockIdx.y, tid) = val;
      }
      else if constexpr (d_in.Rank() == 1) {
        d_out(tid) = val;
      }
    }
    else if (mode == MATX_C_MODE_SAME) {
      int start_tid, stop_tid;
      if (filter_len & 1) {
        start_tid = (filter_len - 1) >> 1;
        stop_tid = signal_len - ((filter_len - 1) >> 1);
      }
      else {
        start_tid = filter_len >> 1;
        stop_tid = signal_len - (filter_len >> 1) + 1;
      }

      if (tid >= start_tid && tid <= stop_tid) {
        if constexpr (d_in.Rank() == 4) {
          d_out(bdims[0], bdims[1], blockIdx.y, tid - start_tid) = val;
        }
        else if constexpr (d_in.Rank() == 3) {
          d_out(blockIdx.z, blockIdx.y, tid - start_tid) = val;
        }
        else if constexpr (d_in.Rank() == 2) {
          d_out(blockIdx.y, tid - start_tid) = val;
        }
        else if constexpr (d_in.Rank() == 1) {
          d_out(tid - start_tid) = val;
        }
      }
    }
    else {
      // not supported yet
    }
  }
}

template <typename OutType, typename InType, typename FilterType>
__global__ void Conv2D(OutType d_out, InType d_in, FilterType d_filter,
                       matxConvCorrMode_t mode)
{
  extern __shared__ float s_exch[];
  using ftype_strip = typename FilterType::scalar_type;
  using intype_strip = typename InType::scalar_type;
  using outtype_strip = typename OutType::scalar_type;
  [[maybe_unused]] index_t bdims[2];
  ftype_strip *s_filter = reinterpret_cast<ftype_strip *>(&s_exch[0]);
  int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
  int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
  int ix_size, iy_size;
  // For rank 4 we need to break up the batch dimensions into the two outer
  // components. Other ranks store the batch index directly in the grid
  // index
  if constexpr (d_in.Rank() == 4) {
    bdims[0] = blockIdx.z / d_in.Size(1);
    bdims[1] = blockIdx.z - (bdims[0] * d_in.Size(1));
    ix_size = d_in.Size(3);
    iy_size = d_in.Size(2);
  }
  if constexpr (d_in.Rank() == 3) {
    ix_size = d_in.Size(2);
    iy_size = d_in.Size(1);
  }
  else if constexpr (d_in.Rank() == 2) {
    ix_size = d_in.Size(1);
    iy_size = d_in.Size(0);
  }

  if ((threadIdx.x < d_filter.Size(1)) && (threadIdx.y < d_filter.Size(0))) {
    s_filter[d_filter.Size(1) * threadIdx.y + threadIdx.x] =
        d_filter(threadIdx.y, threadIdx.x);
  }

  __syncthreads();

  outtype_strip val = 0;

  for (int x = 0; x < d_filter.Size(1); x++) {
    if ((tid_x - static_cast<int>(d_filter.Size(1)) + 1 + x < 0) ||
        (tid_x - static_cast<int>(d_filter.Size(1)) + 1 + x >= ix_size)) {
      continue;
    }

    for (int y = 0; y < d_filter.Size(0); y++) {
      if ((tid_y - static_cast<int>(d_filter.Size(0)) + 1 + y < 0) ||
          (tid_y - static_cast<int>(d_filter.Size(0)) + 1 + y >= iy_size)) {
        continue;
      }

      if constexpr (d_in.Rank() == 4) {
        val += s_filter[y * d_filter.Size(1) + x] *
               d_in(bdims[0], bdims[1], tid_y - d_filter.Size(0) + 1 + y,
                    tid_x - d_filter.Size(1) + 1 + x);
      }
      else if constexpr (d_in.Rank() == 3) {
        val += s_filter[y * d_filter.Size(1) + x] *
               d_in(blockIdx.z, tid_y - d_filter.Size(0) + 1 + y,
                    tid_x - d_filter.Size(1) + 1 + x);
      }
      else if constexpr (d_in.Rank() == 2) {
        val += s_filter[y * d_filter.Size(1) + x] *
               d_in(tid_y - d_filter.Size(0) + 1 + y,
                    tid_x - d_filter.Size(1) + 1 + x);
      }
    }
  }

  if constexpr (d_out.Rank() == 4) {
    if (bdims[0] < d_out.Size(0) && bdims[1] < d_out.Size(1) &&
        tid_y < d_out.Size(2) && tid_x < d_out.Size(3)) {
      d_out(bdims[0], bdims[1], tid_y, tid_x) = val;
    }
  }
  else if constexpr (d_out.Rank() == 3) {
    if (blockIdx.z < d_out.Size(0) && tid_y < d_out.Size(1) &&
        tid_x < d_out.Size(2)) {
      d_out(blockIdx.z, tid_y, tid_x) = val;
    }
  }
  else if constexpr (d_out.Rank() == 2) {
    if (tid_y < d_out.Size(0) && tid_x < d_out.Size(1)) {
      d_out(tid_y, tid_x) = val;
    }
  }
}
#endif

}; // namespace matx
