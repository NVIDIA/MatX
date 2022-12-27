
#pragma once

#include <complex>
#include <cuda.h>
#include <iomanip>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "cuComplex.h"
#include "matx/core/utils.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"

#define CONV1D_ELEMENTS_PER_BLOCK 512

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
template <int THREADS, int EPT, typename OutType, typename InType, typename FilterType>
__launch_bounds__(THREADS)
__global__ void Conv1D(OutType d_out, InType d_in, FilterType d_filter,
                       index_t signal_len,
                       matxConvCorrMode_t mode)
{

  /* strategy:
     1 thread per EPT outputs.  
     Each block produces EPT * THREADS outputs
     Full convolution is computed and results are windowed down based on the request
     Filter is fully loaded into shared memory
     Chunk of signal is loaded into shared memory with filter_len pandding on the negative side.  
     If out of range then we pad with zeros.
     */
  static_assert(InType::Rank() == FilterType::Rank());

  const int Rank = InType::Rank();

  extern __shared__ char s_exch1d[]; // Filter + halo
  using ftype_strip = typename FilterType::scalar_type;
  using intype_strip = typename InType::scalar_type;
  using outtype_strip = typename OutType::scalar_type;
  uint32_t filter_len = d_filter.Size(Rank-1);
  uint32_t full_len = signal_len + filter_len - 1;

  int batch_idx = blockIdx.x;

  // All but the last dim will be populated
  auto bdims = BlockToIdx(d_out, batch_idx, 1);

  ftype_strip *s_filter = reinterpret_cast<ftype_strip *>(&s_exch1d[0]);

  size_t filter_bytes = filter_len * sizeof(ftype_strip);
  // pad bytes to alignmetn of InType
  int align = std::alignment_of_v<intype_strip>;
  filter_bytes = (filter_bytes + align - 1) / align * align;

  intype_strip *s_data = reinterpret_cast<intype_strip*>(&s_exch1d[filter_bytes]);

  // load filter
  for (uint32_t idx = threadIdx.x;  idx < filter_len; idx += THREADS) {
    bdims[Rank - 1] = idx;
    detail::mapply([&](auto &&...args) {
        s_filter[idx] = d_filter.operator()(args...);
        }, bdims);
  }

  // number of chunks in the signal, number of output elements / chunk size rounded up
  uint32_t num_chunks = (signal_len + filter_len -1 + CONV1D_ELEMENTS_PER_BLOCK - 1) / CONV1D_ELEMENTS_PER_BLOCK;

  // number of chunks per Y block, rounded up
  num_chunks = (num_chunks + gridDim.y - 1) / gridDim.y;

#pragma unroll 1
  for(uint32_t n = 0; n < num_chunks; n++) {
    // compute current chunk idx
    uint32_t chunk_idx = blockIdx.y + n * gridDim.y;

    // ensure s_data is consumed from last iteration of chunk loop
    if( n > 0 )
      __syncthreads();

    // load signal,  pad extra elements with zeros
    for (int32_t lidx = threadIdx.x, gidx  = chunk_idx * CONV1D_ELEMENTS_PER_BLOCK - filter_len + 1 + threadIdx.x;  
        gidx < static_cast<int32_t>((chunk_idx+1) * CONV1D_ELEMENTS_PER_BLOCK) ; 
        gidx += THREADS, lidx += THREADS) {

      // some elements may be out of range.  We set their values to 0.
      intype_strip val(0);

      if( gidx >= 0 && gidx < signal_len) { 
        bdims[Rank - 1] = gidx;
        detail::mapply([&](auto &&...args) {
            val = d_in.operator()(args...);
            }, bdims);
      }

      s_data[lidx] = val;
    }

    // wait for signal to load
    __syncthreads();

    // register array for output data  
    outtype_strip oval[EPT] = {0}; 

    // Below will use pointer modification instead of offsets to change IMADS into IADS.  IMADS go through FMA pipe.

    // offset s_data to last element in the filter
    s_data += threadIdx.x + filter_len - 1;

    // for each tap
    for(uint32_t f = 0; f < filter_len; f++) {
      // load filter value into registers
      ftype_strip fval = s_filter[0];

      // next filter value
      s_filter++;

      // register array for signal data
      intype_strip ival[EPT];
      // load N elements of the signal into registers

#pragma unroll
      for(uint32_t i = 0; i < EPT; i++) {
        ival[i] = s_data[i*THREADS];
      }
      s_data--; // next signal value

      // compute N elements of the convolution
#pragma unroll
      for(uint32_t i = 0; i < EPT; i++) {
        oval[i] = detail::madd(ival[i], fval, oval[i]);
      }
    }

    // restore shared pointers for next loop
    s_filter -= filter_len;
    s_data -= (threadIdx.x - 1);

    // We have computed the full convolution here.  we now need to output the correct range.
    // compute output range
    uint32_t start;
    uint32_t stop;

    if (mode == MATX_C_MODE_FULL) {
      start = 0;
      stop = full_len - 1;
    } else if ( mode == MATX_C_MODE_SAME) {
      if( filter_len & 1) {
        start = (filter_len - 1) / 2;
      } else {
        start = filter_len / 2 - 1;
      }
      stop = full_len - filter_len / 2 - 1;
    } else {
      start = filter_len - 1;
      stop = full_len - filter_len;
    }

#pragma unroll
    for (uint32_t i = 0; i < EPT; i++) {
      // index for the computation
      uint32_t idx = chunk_idx * CONV1D_ELEMENTS_PER_BLOCK + i * THREADS + threadIdx.x;
      // output index is shifted by start
      int32_t gidx = idx - start;

      if(idx >= start && idx <= stop) {
        bdims[Rank - 1] = gidx; 
        detail::mapply([&](auto &&...args) {
            d_out.operator()(args...) = oval[i];
            }, bdims);        
      }
    }
  } // end chunk loop
}

template <typename OutType, typename InType1, typename InType2>
__launch_bounds__(1024)
__global__ void Conv2D(OutType d_out, InType1 d_in1, InType2 d_in2,
                       matxConvCorrMode_t mode, int num_batch)
{
  using in1type = typename InType1::scalar_type;
  using in2type = typename InType1::scalar_type;
  using outtype = typename OutType::scalar_type;

  constexpr int Rank = OutType::Rank();

  index_t oN = d_out.Size(Rank-2);
  index_t oM = d_out.Size(Rank-1);

  index_t i1N = d_in1.Size(Rank-2);
  index_t i1M = d_in1.Size(Rank-1);

  index_t i2N = d_in2.Size(Rank-2);
  index_t i2M = d_in2.Size(Rank-1);

  int dy = 0, dx = 0;

  // TODO detect mode based on output size?
  if( mode == MATX_C_MODE_SAME) {
    dy = i2N / 2;
    dx = i2M / 2;
#if 0
    // uncomment this to match matlab
    if(i2N % 2 == 0) dy--;
    if(i2M % 2 == 0) dx--;
#endif

  } else if ( mode == MATX_C_MODE_FULL) {
    dy = i2N - 1;
    dx = i2M - 1;
  }

  // TODO benchmark
  // TODO lots of optimization

  for(int batch_idx = blockIdx.z; batch_idx < num_batch; batch_idx+=gridDim.z) {
    // All but the last 2 dims will be populated
    auto bdims = BlockToIdx(d_out, batch_idx, 2);

    // for each output
    for( index_t i = blockIdx.y * blockDim.y + threadIdx.y; i < oN; i+= blockDim.y * gridDim.y) {
      for( index_t j = blockIdx.x * blockDim.x + threadIdx.x; j < oM; j+= blockDim.x * gridDim.x) {

        outtype sum(0);

        for (index_t n=0; n < i2N; n++) {
          index_t k = i2N - 1 - n; // flip filter
          index_t y = i + n - dy;
          for (index_t m=0; m < i2M; m++) {
            index_t l = i2M - 1 - m; // flip filter
            index_t x = j + m - dx;

//            if(i==0 && j == 0) printf("(i: %lld, j: %lld), (y:%lld, x: %lld), (k: %lld, l: %lld)\n",i,j,y,x,k,l);
            if( x >= 0 && x < i1M && y >=0 && y < i1N) {

              in1type i1; 
              in2type i2;

              // Signal Dims
              bdims[Rank - 2] = y;
              bdims[Rank - 1] = x;
              detail::mapply([&](auto &&...args) { i1 = d_in1.operator()(args...); }, bdims);

              // Filter Dims
              bdims[Rank - 2] = k;
              bdims[Rank - 1] = l;
              detail::mapply([&](auto &&...args) { i2 = d_in2.operator()(args...); }, bdims);

//              if(i==0 && j == 0) printf("madd: %d, %d)\n", i1, i2);
              sum = detail::madd(i1, i2, sum);
            }
          }
        }

        bdims[Rank - 2] = i;
        bdims[Rank - 1] = j;
        detail::mapply([&](auto &&...args) { d_out.operator()(args...) = sum; }, bdims);        

      }
    }
  }
}
#endif

}; // namespace matx
