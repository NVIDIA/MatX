
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


namespace matx {

namespace matx_conv1d_detail {
  constexpr size_t CONV1D_ELEMENTS_PER_BLOCK = 512;
};
using namespace matx_conv1d_detail;

typedef enum {
  MATX_C_MODE_FULL, // Default. Keep all elements of ramp up/down
  MATX_C_MODE_SAME, // Only keep elements where entire filter was present
  MATX_C_MODE_VALID
} matxConvCorrMode_t;

typedef enum {
  MATX_C_METHOD_DIRECT,
  MATX_C_METHOD_FFT
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
  using ftype_strip = typename FilterType::value_type;
  using intype_strip = typename InType::value_type;
  using outtype_strip = typename OutType::value_type;
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
    cuda::std::apply([&, d_filter](auto &&...args) {
        s_filter[idx] = d_filter.operator()(args...);
        }, bdims);
  }

  // number of chunks in the signal, number of output elements / chunk size rounded up
  uint32_t num_chunks = (signal_len + filter_len -1 + CONV1D_ELEMENTS_PER_BLOCK - 1) / CONV1D_ELEMENTS_PER_BLOCK;

  // number of chunks per Y block, rounded up
  num_chunks = (num_chunks + gridDim.y - 1) / gridDim.y;

  MATX_LOOP_DO_NOT_UNROLL
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
        cuda::std::apply([&val, d_in](auto &&...args) {
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

MATX_LOOP_UNROLL
      for(uint32_t i = 0; i < EPT; i++) {
        ival[i] = s_data[i*THREADS];
      }
      s_data--; // next signal value

      // compute N elements of the convolution
MATX_LOOP_UNROLL
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

MATX_LOOP_UNROLL
    for (uint32_t i = 0; i < EPT; i++) {
      // index for the computation
      uint32_t idx = chunk_idx * CONV1D_ELEMENTS_PER_BLOCK + i * THREADS + threadIdx.x;
      // output index is shifted by start
      int32_t gidx = idx - start;

      if(idx >= start && idx <= stop) {
        bdims[Rank - 1] = gidx;
        cuda::std::apply([&](auto &&...args) {
            d_out.operator()(args...) = oval[i];
            }, bdims);
      }
    }
  } // end chunk loop
}


template <typename T>
struct Uninitialized {
    __host__ __device__ constexpr Uninitialized() {};
    T data;
};

template <typename T, int X_LEN>
struct ShmBuffer2D {
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ ShmBuffer2D(char *p) {
    ptr = reinterpret_cast<T*>(p);
  }
  __MATX_INLINE__  __MATX_DEVICE__ __MATX_HOST__ const T &operator()(index_t y, index_t x) const noexcept {
    return *(ptr + y * X_LEN + x);
  }

  __MATX_INLINE__  __MATX_DEVICE__ __MATX_HOST__ T &operator()(index_t y, index_t x) noexcept {
    return *(ptr + y * X_LEN + x);
  }

  T *ptr;
};

template <typename OutType, typename InType1, typename InType2,
  int BLOCK_DIM_X,      // blockDim.x
  int BLOCK_DIM_Y,      // blockDim.y
  int FILTER_SHARED_CHUNK_X, // Filter shared memory tile in X
  int FILTER_SHARED_CHUNK_Y, // Filter shared memory tile in Y
  int FILTER_REG_CHUNK_X, // Filter register tile in X
  int FILTER_REG_CHUNK_Y, // Filter register tile in Y
  int ILPY>  //  number of elements per thread in Y dimension.
__launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y)
__global__ void Conv2D(OutType d_out, InType1 d_in1, InType2 d_in2,
                       matxConvCorrMode_t mode, int num_batch)
{
  using in1type = typename InType1::value_type;
  using in2type = typename InType2::value_type;
  using outtype = typename OutType::value_type;

  // length of signal chunk in shared memory
  constexpr int SIGNAL_CHUNK_X = BLOCK_DIM_X + FILTER_SHARED_CHUNK_X;
  constexpr int SIGNAL_CHUNK_Y = BLOCK_DIM_Y * ILPY + FILTER_SHARED_CHUNK_Y + ILPY;

  constexpr int Rank = OutType::Rank();

  constexpr int type2off = MATX_ROUND_UP(sizeof(in1type) * SIGNAL_CHUNK_Y * SIGNAL_CHUNK_X, sizeof(in2type));
  __shared__ char shared_buf[type2off + sizeof(in2type) * FILTER_SHARED_CHUNK_Y * FILTER_SHARED_CHUNK_X];

  // __shared__ Uninitialized<in1type> s_signal[SIGNAL_CHUNK_Y][SIGNAL_CHUNK_X];
  // __shared__ Uninitialized<in2type> s_filter[FILTER_SHARED_CHUNK_Y][FILTER_SHARED_CHUNK_X];

  // Workaround for ARM compiler bug that will not allow the union type above
  ShmBuffer2D<in1type, SIGNAL_CHUNK_X> s_signal{&shared_buf[0]};
  ShmBuffer2D<in2type, FILTER_SHARED_CHUNK_X> s_filter{&shared_buf[type2off]};

  in2type r_filter[FILTER_REG_CHUNK_Y][FILTER_REG_CHUNK_X];

  index_t oN = d_out.Size(Rank-2);
  index_t oM = d_out.Size(Rank-1);

  index_t i1N = d_in1.Size(Rank-2);
  index_t i1M = d_in1.Size(Rank-1);

  index_t i2N = d_in2.Size(Rank-2);
  index_t i2M = d_in2.Size(Rank-1);

  int dy = 0, dx = 0;

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

  // grid stride loop over batches
  for(int batch_idx = blockIdx.z; batch_idx < num_batch; batch_idx+=gridDim.z) {
    // All but the last 2 dims will be populated
    auto bdims = BlockToIdx(d_out, batch_idx, 2);

    // for each output (converged loops), ILPY elements per thread
    for( index_t bi = blockIdx.y * blockDim.y * ILPY; bi < oN; bi+= blockDim.y * gridDim.y * ILPY) {
      for( index_t bj = blockIdx.x * blockDim.x; bj < oM; bj+= blockDim.x * gridDim.x) {
        index_t i = bi + threadIdx.y * ILPY;
        index_t j = bj + threadIdx.x;

        outtype sum[ILPY] = {0};

        // for each shared memory filter chunk
        for (index_t nStart = 0; nStart < i2N; nStart+=FILTER_SHARED_CHUNK_Y) {
          for (index_t mStart = 0; mStart < i2M; mStart+=FILTER_SHARED_CHUNK_X) {
            __syncthreads();
            // load filter from global to shared
            for(int ii = threadIdx.y; ii < FILTER_SHARED_CHUNK_Y; ii+=blockDim.y) {
              for(int jj = threadIdx.x; jj < FILTER_SHARED_CHUNK_X; jj+=blockDim.x) {
                in2type val = in2type(0);
                // compute filter index
                index_t k = i2N - 1 - (nStart+ii);
                index_t l = i2M - 1 - (mStart+jj);
                // if filter in range
                if( k >= 0 && l >= 0) {
                  // Filter Dims
                  bdims[Rank - 2] = k;
                  bdims[Rank - 1] = l;
                  // load filter value
                  cuda::std::apply([&](auto &&...args) { val = d_in2.operator()(args...); }, bdims);
                }
                // store in shared
                s_filter(ii, jj) = val;
              }
            }

            // load signal from global to shared, if out of range set to zero
            for(int ii = threadIdx.y; ii < SIGNAL_CHUNK_Y; ii+=blockDim.y) {
              for(int jj = threadIdx.x; jj < SIGNAL_CHUNK_X; jj+=blockDim.x) {
                in1type val = in1type(0);
                // compute filter index
                index_t y = bi + nStart + ii - dy;
                index_t x = bj + mStart + jj - dx;

                // if signal in range
                if( x >= 0 && x < i1M && y >=0 && y < i1N) {
                  // Signal Dims
                  bdims[Rank - 2] = y;
                  bdims[Rank - 1] = x;
                  cuda::std::apply([&](auto &&...args) { val = d_in1.operator()(args...); }, bdims);
                }

                // store in shared
                s_signal(ii,jj) = val;
              }
            }

            __syncthreads();

            in1type i1[ILPY];

            // loop through shared memory filter one chunk at a time
MATX_LOOP_UNROLL
            for (int mm = 0; mm < FILTER_SHARED_CHUNK_X; mm+=FILTER_REG_CHUNK_X) {
MATX_LOOP_UNROLL
              for (int nn = 0; nn < FILTER_SHARED_CHUNK_Y; nn+=FILTER_REG_CHUNK_Y) {


                // Copy chunk from shared memory in to registers
MATX_LOOP_UNROLL
                for(int ii = 0; ii < FILTER_REG_CHUNK_Y; ii++) {
MATX_LOOP_UNROLL
                  for(int jj = 0; jj < FILTER_REG_CHUNK_X; jj++) {
                    r_filter[ii][jj] = s_filter(nn+ii, mm+jj);
                  }
                }


                // convolution loop:  convolve filter and signal.
                // Keep signal in registers as much as possible by shifting.
MATX_LOOP_UNROLL
                for (int m = 0; m < FILTER_REG_CHUNK_X; m++) {

MATX_LOOP_UNROLL
                  for (int n = 0; n < FILTER_REG_CHUNK_Y; n++) {

                    in2type i2 = r_filter[n][m];

                    // if FILTER_REG_CHUNK_X > 1 then we need to reload i1 every m loop
                    if( nn == 0 ||
                        (FILTER_REG_CHUNK_X > 1 && n == 0)) {
                    // load ILPY signal points
MATX_LOOP_UNROLL
                      for(int u = 0; u < ILPY; u++) {
                        i1[u] = s_signal(nn+n+threadIdx.y*ILPY + u, mm+m+threadIdx.x);
                      }
                    } else {
                      // advance/shift signal points in registers
MATX_LOOP_UNROLL
                      for(int u = 0; u < ILPY - 1; u++) {
                        i1[u]=i1[u+1];
                      }

                      // load new signal point at end of the array
                      i1[ILPY-1] = s_signal(nn+n+threadIdx.y*ILPY + ILPY - 1,mm+m+threadIdx.x);
                    }

                    // inner convolution loop
MATX_LOOP_UNROLL
                    for(int u = 0; u < ILPY; u++) {
                      sum[u] = detail::madd(i1[u], i2, sum[u]);
                    }
                  } // end n loop
                } // end m loop
              } // end nn loop
            } // end mm loop
          }  // end jj loop
        } // end ii loop

        // finally output the solution
MATX_LOOP_UNROLL
        for(int u = 0; u < ILPY; u++) {
          if(i + u < oN && j < oM) {
            bdims[Rank - 2] = i + u;
            bdims[Rank - 1] = j;
            cuda::std::apply([&](auto &&...args) { d_out.operator()(args...) = sum[u]; }, bdims);
          }
        }

      }
    }
  }
}
#endif

}; // namespace matx
