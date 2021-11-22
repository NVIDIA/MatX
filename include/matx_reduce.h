////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "matx_cub.h"
#include "matx_error.h"
#include "matx_get_grid_dims.h"
#include "matx_tensor.h"
#include "matx_type_utils.h"
#include <cfloat>

#ifdef __CUDACC__  
/**
 * Warp shuffle down with a complex float
 *
 * Shuffles both the real and imaginary components down by the specified delta
 * over the thread mask.
 *
 * @param mask
 *   Thread mask of warp to participate in shuffle
 * @param var
 *   Variable to shuffle
 * @param delta
 *   Amount to shuffle
 * @returns
 *   Value shuffled*
 */
__MATX_DEVICE__ inline auto __shfl_down_sync(unsigned mask,
                                        cuda::std::complex<float> var,
                                        unsigned int delta)
{
  var.real(__shfl_down_sync(mask, var.real(), delta));
  var.imag(__shfl_down_sync(mask, var.imag(), delta));
  return var;
}

/**
 * Warp shuffle down with a complex double
 *
 * Shuffles both the real and imaginary components down by the specified delta
 * over the thread mask.
 *
 * @param mask
 *   Thread mask of warp to participate in shuffle
 * @param var
 *   Variable to shuffle
 * @param delta
 *   Amount to shuffle
 * @returns
 *   Value shuffled
 */
__MATX_DEVICE__ inline auto __shfl_down_sync(unsigned mask,
                                        cuda::std::complex<double> var,
                                        unsigned int delta)
{
  var.real(__shfl_down_sync(mask, var.real(), delta));
  var.imag(__shfl_down_sync(mask, var.imag(), delta));
  return var;
}

/**
 * Atomic min version for floats
 *
 * Computes the minimum of two floating point values atomically
 *
 * @param addr
 *   Source and destination for new minimum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicMin(float *addr, float val)
{
  unsigned int *address_as_uint = (unsigned int *)addr;
  unsigned int old = *address_as_uint, assumed;
  unsigned int val_uint = __float_as_uint(val);

  // nan should be ok here but should verify
  while (val < __uint_as_float(old)) {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed, val_uint);
  }
};

/**
 * Atomic max version for floats
 *
 * Computes the maximum of two floating point values atomically
 *
 * @param addr
 *   Source and destination for new maximum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicMax(float *addr, float val)
{
  unsigned int *address_as_uint = (unsigned int *)addr;
  unsigned int old = *address_as_uint, assumed;
  unsigned int val_uint = __float_as_uint(val);

  // nan should be ok here but should verify
  while (val > __uint_as_float(old)) {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed, val_uint);
  }
};

/**
 * Atomic any version for floats
 *
 * Computes whether either of two floating point values are != 0
 *
 * @param addr
 *   Source and destination for new any
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicAny(float *addr, float val)
{
  // We don't actually need an atomic operation here since we only write to the
  // location if any thread has the correct value.
  if (val != 0) {

    *addr = 1.0;
  }
};

/**
 * Atomic multiply version for uint32_t
 *
 * Computes the product of two uint32_t atomically
 *
 * @param address
 *   Source and destination for new any
 * @param val
 *   Value to compare against
 */
template <typename T> __MATX_DEVICE__ T atomicMul(T *address, T val)
{
  T old = *address, assumed;

  do {
    assumed = old;
    if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) {
      old = atomicCAS(reinterpret_cast<unsigned long long *>(address),
                      static_cast<unsigned long long>(assumed),
                      static_cast<unsigned long long>(val * assumed));
    }
    else {
      old = atomicCAS(address, assumed, val * assumed);
    }

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return old;
}

/**
 * Atomic multiply version for doubles
 *
 * Computes the product of two doubles atomically
 *
 * @param address
 *   Source and destination for new any
 * @param val
 *   Value to compare against
 */
template <> __MATX_DEVICE__ inline float atomicMul(float *address, float val)
{
  unsigned int *address_as_uint = (unsigned int *)address;
  unsigned int old = *address_as_uint, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed,
                    __float_as_uint(val * __uint_as_float(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __uint_as_float(old);
}

/**
 * Atomic all version for floats
 *
 * Computes whether both floating point values are != 0
 *
 * @param addr
 *   Source and destination for new any
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicAll(float *addr, float val)
{
  unsigned int *address_as_uint = (unsigned int *)addr;
  unsigned int old = *address_as_uint, assumed;
  unsigned int val_uint = __float_as_uint(val);

  // nan should be ok here but should verify
  while (val == 0.0 && old != 0.0) {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed, 0.0);
  }
};

/**
 * Atomic multiply version for doubles
 *
 * Computes the product of two doubles atomically
 *
 * @param address
 *   Source and destination for new any
 * @param val
 *   Value to compare against
 */
template <> __MATX_DEVICE__ inline double atomicMul(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val * __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

/**
 * Atomic min version for doubles
 *
 * Computes the minimum of two floating point values atomically
 *
 * @param addr
 *   Source and destination for new minimum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicMin(double *addr, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;
  unsigned long long int val_ull = __double_as_longlong(val);

  // nan should be ok here but should verify
  while (val < __longlong_as_double(old)) {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, val_ull);
  }
};

/**
 * Atomic max version for doubles
 *
 * Computes the maximum of two floating point values atomically
 *
 * @param addr
 *   Source and destination for new maximum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicMax(double *addr, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;
  unsigned long long int val_ull = __double_as_longlong(val);

  // nan should be ok here but should verify
  while (val > __longlong_as_double(old)) {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, val_ull);
  }
};

/**
 * Atomic any version for doubles
 *
 * Computes whether either of two double floating point values are non-zero
 * atomically
 *
 * @param addr
 *   Source and destination for new maximum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicAny(double *addr, double val)
{
  // We don't actually need an atomic operation here since we only write to the
  // location if any thread has the correct value.
  if (val != 0) {
    *addr = 1.0;
  }
};

/**
 * Atomic all version for double
 *
 * Computes whether both doublest values are != 0
 *
 * @param addr
 *   Source and destination for new any
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicAll(double *addr, float val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;
  unsigned long long int val_ull = __double_as_longlong(val);

  // nan should be ok here but should verify
  while (val == 0.0 && old != 0.0) {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, 0.0);
  }
}

/**
 * Atomic add for complex floats
 *
 * Atomically adds two complex floating point numbers. Note that
 * the real and imaginary components are added separately, so while
 * each of those alone are atomic, both together are not.
 *
 * @param addr
 *   Source and destination for result
 * @param val
 *   Value to add
 */
__MATX_DEVICE__ inline void atomicAdd(cuda::std::complex<float> *addr,
                                 cuda::std::complex<float> val)
{
  float *addrf = reinterpret_cast<float *>(addr);
  atomicAdd(addrf, val.real());
  atomicAdd(addrf + 1, val.imag());
}

/**
 * Atomic add for complex doubles
 *
 * Atomically adds two complex double floating point numbers. Note that
 * the real and imaginary components are added separately, so while
 * each of those alone are atomic, both together are not.
 *
 * @param addr
 *   Source and destination for result
 * @param val
 *   Value to add
 */
__MATX_DEVICE__ inline void atomicAdd(cuda::std::complex<double> *addr,
                                 cuda::std::complex<double> val)
{
  double *addrp = reinterpret_cast<double *>(addr);
  atomicAdd(addrp, val.real());
  atomicAdd(addrp + 1, val.imag());
}
#endif

namespace matx {

#ifdef __CUDACC__  
template <typename T> constexpr inline __MATX_HOST__ __MATX_DEVICE__ T maxVal();
template <typename T> constexpr inline __MATX_HOST__ __MATX_DEVICE__ T minVal();

/* Returns the max value of an int32_t at compile time */
template <> constexpr inline __MATX_HOST__ __MATX_DEVICE__ int32_t maxVal<int32_t>()
{
  return INT_MAX;
}
/* Returns the min value of an int32_t at compile time */
template <> constexpr inline __MATX_HOST__ __MATX_DEVICE__ int32_t minVal<int32_t>()
{
  return INT_MIN;
}
/* Returns the max value of a uint32_t at compile time */
template <> constexpr inline __MATX_HOST__ __MATX_DEVICE__ uint32_t maxVal<uint32_t>()
{
  return UINT_MAX;
}
/* Returns the min value of a uint32_t at compile time */
template <> constexpr inline __MATX_HOST__ __MATX_DEVICE__ uint32_t minVal<uint32_t>()
{
  return 0;
}
/* Returns the max value of a float at compile time */
template <> constexpr inline __MATX_HOST__ __MATX_DEVICE__ float maxVal<float>()
{
  return FLT_MAX;
}
/* Returns the min value of a float at compile time */
template <> constexpr inline __MATX_HOST__ __MATX_DEVICE__ float minVal<float>()
{
  return -FLT_MAX;
}
/* Returns the max value of a double at compile time */
template <> constexpr inline __MATX_HOST__ __MATX_DEVICE__ double maxVal<double>()
{
  return DBL_MAX;
}
/* Returns the min value of a double at compile time */
template <> constexpr inline __MATX_HOST__ __MATX_DEVICE__ double minVal<double>()
{
  return -DBL_MAX;
}


/**
 * Operator for performing a sum reduction
 *
 * Performs a reduction of two values of type T by summing the
 * values
 */
template <typename T> class reduceOpSum {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ inline T Reduce(T v1, T v2) { return v1 + v2; }
  __MATX_HOST__ __MATX_DEVICE__ inline T Init() { return T(0); }
  __MATX_DEVICE__ inline void atomicReduce(T *addr, T val) { atomicAdd(addr, val); }
};

/**
 * Operator for performing a product reduction
 *
 * Performs a reduction of two values of type T by multiplying the
 * values
 */
template <typename T> class reduceOpProd {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ inline T Reduce(T v1, T v2) { return v1 * v2; }
  __MATX_HOST__ __MATX_DEVICE__ inline T Init() { return T(1); }
  __MATX_DEVICE__ inline void atomicReduce(T *addr, T val) { atomicMul(addr, val); }
};

/**
 * Operator for performing a max reduction
 *
 * Performs a reduction of two values of type T by taking the max
 * of the two values. Type must have operator> defined to perform
 * max
 */
template <typename T> class reduceOpMax {
public:
  using matx_reduce = bool;
  using matx_reduce_index = bool;
  __MATX_HOST__ __MATX_DEVICE__ inline T Reduce(T v1, T v2) { return v1 > v2 ? v1 : v2; }
  __MATX_HOST__ __MATX_DEVICE__ inline T Init() { return minVal<T>(); }
  __MATX_DEVICE__ inline void atomicReduce(T *addr, T val) { atomicMax(addr, val); }
};

/**
 * Operator for performing an any reduction
 *
 * Performs a reduction of two values of type T by returning 1 if either
 * of the values are non-zero.
 */
template <typename T> class reduceOpAny {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ inline T Reduce(T v1, T v2)
  {
    return (v1 != 0) || (v2 != 0);
  }
  __MATX_HOST__ __MATX_DEVICE__ inline T Init() { return (T)(0); }
  __MATX_DEVICE__ inline void atomicReduce(T *addr, T val) { atomicAny(addr, val); }
};

/**
 * Operator for performing an any reduction
 *
 * Performs a reduction of two values of type T by returning 1 if either
 * of the values are non-zero.
 */
template <typename T> class reduceOpAll {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ inline T Reduce(T v1, T v2)
  {
    return (v1 != 0) && (v2 != 0);
  }
  __MATX_HOST__ __MATX_DEVICE__ inline T Init() { return (T)(1); }
  __MATX_DEVICE__ inline void atomicReduce(T *addr, T val) { atomicAll(addr, val); }
};

/**
 * Operator for performing a min reduction
 *
 * Performs a reduction of two values of type T by taking the min
 * of the two values. Type must have operator< defined to perform
 * min
 */
template <typename T> class reduceOpMin {
public:
  using matx_reduce = bool;
  using matx_reduce_index = bool;
  __MATX_HOST__ __MATX_DEVICE__ inline T Reduce(T v1, T v2) { return v1 < v2 ? v1 : v2; }
  __MATX_HOST__ __MATX_DEVICE__ inline T Init() { return maxVal<T>(); }
  __MATX_DEVICE__ inline void atomicReduce(T *addr, T val) { atomicMin(addr, val); }
};

#if 0
  template<typename T>
    class reduceOpSumMax {
      public:
        using matx_reduce = bool;
        __MATX_HOST__ __MATX_DEVICE__ inline T Reduce(T v1, T v2) {
          T val;
          val.x = reduceOpSum<decltype(T::x)>().Reduce(v1,v2);
          val.y = reduceOpMax<decltype(T::y)>().Reduce(v1,v2); 
          return val;
        }
        __MATX_HOST__ __MATX_DEVICE__ inline T Init() {
          T val;
          val.x = reduceOpSum<decltype(T::x)>().Init();
          val.y = reduceOpMax<decltype(T::y)>().Init();
          return val;
        }
        __MATX_DEVICE__ inline void atomicReduce(T *addr, T val) {
          reduceOpSum<decltype(T::x)>().atomicReduce(&(addr->x), val.x);
          reduceOpMax<decltype(T::y)>().atomicReduce(&(addr->y), val.y);
        }
    };
#endif

template <typename T, typename Op>
__MATX_DEVICE__ inline T warpReduceOp(T val, Op op, uint32_t size)
{
  // breaking this out so common case is faster without branches
  if (size > 16) {
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 16));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 8));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 4));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 2));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 1));
  }
  else if (size > 8) {
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 8));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 4));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 2));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 1));
  }
  else if (size > 4) {
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 4));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 2));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 1));
  }
  else if (size > 2) {
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 2));
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 1));
  }
  else if (size > 1) {
    val = op.Reduce(val, __shfl_down_sync(0xffffffff, val, 1));
  }
  return val;
}


template <typename T, int RANK, typename InType, typename ReduceOp>
__global__ void matxReduceKernel(tensor_t<T, RANK> dest, InType in,
                                 ReduceOp red)
{

  constexpr int DRANK = InType::Rank() - RANK;
  using scalar_type = typename InType::scalar_type;
  // This is for 2 stage reduction

  // nvcc limitation here.  we have to declare shared memory with the same type
  // across all template functions then recast to the type we want
  extern __shared__ char smemc_[];
  scalar_type *smem_ = reinterpret_cast<scalar_type *>(smemc_);

  int s2_size, soff;
  scalar_type *smem;

  // if blockDim.x > 32 we need a 2 stage reduction
  if (blockDim.x > 32) {

    // number of shared memory entries per xdim of block
    s2_size = blockDim.x / 32;
    // offset into shared memory
    soff = threadIdx.z * blockDim.y * s2_size + threadIdx.y * s2_size;

    // offset shared memory
    smem = smem_ + soff;
  }

  // Read input
  typename InType::scalar_type in_val = red.Init();
  [[maybe_unused]] index_t idx, idy, idz, idw;

  if constexpr (InType::Rank() == 1) {
    idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < in.Size(0)) {
      in_val = in(idx);
    }
  }
  else if constexpr (InType::Rank() == 2) {
    idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    idy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (idy < in.Size(0) && idx < in.Size(1)) {
      in_val = in(idy, idx);
    }
  }
  else if constexpr (InType::Rank() == 3) {
    idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    idy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    idz = static_cast<index_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (idz < in.Size(0) && idy < in.Size(1) && idx < in.Size(2)) {
      in_val = in(idz, idy, idx);
    }
  }
  else if constexpr (InType::Rank() == 4) {
    idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    index_t nmy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    idy = nmy % in.Size(2);
    idz = nmy / in.Size(2);
    idw = blockIdx.z * blockDim.z + threadIdx.z;
    if (idw < in.Size(0) && idz < in.Size(1) && idy < in.Size(2) &&
        idx < in.Size(3)) {
      in_val = in(idw, idz, idy, idx);
    }
  }

  // Compute offset index based on rank difference
  if constexpr (DRANK == 1) {
    // Shift ranks by 1
    if constexpr (InType::Rank() >= 2)
      idx = idy;
    if constexpr (InType::Rank() >= 3)
      idy = idz;
    if constexpr (InType::Rank() >= 4)
      idz = idw;
  }
  else if constexpr (DRANK == 2) {
    // Shift ranks by 2
    if constexpr (InType::Rank() >= 3)
      idx = idz;
    if constexpr (InType::Rank() >= 4)
      idy = idw;
  }
  else if constexpr (DRANK == 3) {
    // shift ranks by 3
    if constexpr (InType::Rank() >= 4)
      idx = idw;
  }

  // Compute output location
  T *out = nullptr;

  // compute output offsets
  if constexpr (RANK == 0) {
    out = &dest();
  }
  else if constexpr (RANK == 1) {
    if (idx < dest.Size(0))
      out = &dest(idx);
  }
  else if constexpr (RANK == 2) {
    if (idx < dest.Size(1) && idy < dest.Size(0))
      out = &dest(idy, idx);
  }
  else if constexpr (RANK == 3) {
    if (idx < dest.Size(2) && idy < dest.Size(1) && idz < dest.Size(0))
      out = &dest(idz, idy, idx);
  }

  // reduce along x dim (warp)
  in_val = warpReduceOp(in_val, red, blockDim.x);

  if (blockDim.x > 32) {
    // enter 2 stage reduction

    // first thread of warp write to shared memory
    if (threadIdx.x % 32 == 0) {
      smem[threadIdx.x / 32] = in_val;
    }

    // wait for write
    __syncthreads();

    if (threadIdx.x < s2_size) {
      in_val = smem[threadIdx.x];
    }

    // data is all on first warp now
    // reduce one more time
    in_val = warpReduceOp(in_val, red, s2_size);
  }

  if (out != nullptr && threadIdx.x == 0) {
    // thread 0 update global memory
    red.atomicReduce(out, in_val);
  }
}

template <typename T, int RANK, typename InType>
__global__ void matxIndexKernel(tensor_t<index_t, RANK> idest, tensor_t<T, RANK> dest, InType in)
{
  typename InType::scalar_type in_val;
  [[maybe_unused]] index_t idx, idy, idz, idw;
  constexpr int DRANK = InType::Rank() - RANK;
  index_t abs_idx;

  if constexpr (InType::Rank() == 1) {
    idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < in.Size(0)) {
      in_val = in(idx);
      abs_idx = idx;
    }
  }
  else if constexpr (InType::Rank() == 2) {
    idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    idy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (idy < in.Size(0) && idx < in.Size(1)) {
      in_val = in(idy, idx);
      abs_idx = idy*in.Size(1) + idx;
    }
  }
  else if constexpr (InType::Rank() == 3) {
    idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    idy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    idz = static_cast<index_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (idz < in.Size(0) && idy < in.Size(1) && idx < in.Size(2)) {
      in_val = in(idz, idy, idx);
      abs_idx = idz*in.Size(0)*in.Size(1) + idy*in.Size(1) + idx;
    }
  }
  else if constexpr (InType::Rank() == 4) {
    idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    index_t nmy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    idy = nmy % in.Size(2);
    idz = nmy / in.Size(2);
    idw = blockIdx.z * blockDim.z + threadIdx.z;
    if (idw < in.Size(0) && idz < in.Size(1) && idy < in.Size(2) &&
        idx < in.Size(3)) {
      in_val = in(idw, idz, idy, idx);
      abs_idx = idw*in.Size(0)*in.Size(1)*in.Size(2) + idz*in.Size(0)*in.Size(1) + idy*in.Size(1) + idx;
    }
  }

  // Compare value to reduced
  // Compute offset index based on rank difference
  if constexpr (DRANK == 1) {
    // Shift ranks by 1
    if constexpr (InType::Rank() >= 2)
      idx = idy;
    if constexpr (InType::Rank() >= 3)
      idy = idz;
    if constexpr (InType::Rank() >= 4)
      idz = idw;
  }
  else if constexpr (DRANK == 2) {
    // Shift ranks by 2
    if constexpr (InType::Rank() >= 3)
      idx = idz;
    if constexpr (InType::Rank() >= 4)
      idy = idw;
  }
  else if constexpr (DRANK == 3) {
    // shift ranks by 3
    if constexpr (InType::Rank() >= 4)
      idx = idw;
  }

  // Compute output location
  T *out = nullptr;
  index_t *iout = nullptr;
  bool valid = false;

  // compute output offsets
  if constexpr (RANK == 0) {
    out  = &dest();
    iout = &idest();
    valid = true;
  }
  else if constexpr (RANK == 1) {
    if (idx < dest.Size(0)) {
      out = &dest(idx);
      iout = &idest(idx);
      valid = true;
    }
  }
  else if constexpr (RANK == 2) {
    if (idx < dest.Size(1) && idy < dest.Size(0))
      out = &dest(idy, idx);
      iout = &idest(idy, idx);
      valid = true;
  }
  else if constexpr (RANK == 3) {
    if (idx < dest.Size(2) && idy < dest.Size(1) && idz < dest.Size(0))
      out = &dest(idz, idy, idx);
      iout = &idest(idz, idy, idx);
      valid = true;
  }

  if (valid) {  
    __threadfence();

    // Value matches
    if (*out == in_val) {    
      atomicMin(iout, abs_idx);
    }
  }  
}
#endif

/**
 * Perform a reduction and preserves indices
 *
 * Performs a reduction from tensor "in" into values tensor "dest" and index tensor idest using reduction
 * operation ReduceOp. The output tensor rank dictates which elements the reduction
 * is performed over. In general, the reductions are performed over the
 * innermost dimensions, where the number of dimensions is the difference
 * between the input and output tensor ranks. For example, for a 0D (scalar)
 * output tensor, the reduction is performed over the entire tensor. For
 * anything higher, the reduction is performed across the number of ranks below
 * the input tensor that the output tensor is. For example, if the input tensor
 * is a 4D tensor and the output is a 1D tensor, the reduction is performed
 * across the innermost dimension of the input. If the output is a 2D tensor,
 * the reduction is performed across the two innermost dimensions of the input,
 * and so on.
 *
 * @tparam T
 *   Output data type
 * @param RANK
 *   Rank of output value tensor
 * @tparam InType
 *   Input data type
 * @tparam ReduceOp
 *   Reduction operator to apply
 *
 * @param dest
 *   Destination view of values reduced
 * @param idest
 *   Destination view of indices
 * @param in
 *   Input data to reduce
 * @param op
 *   Reduction operator
 * @param stream
 *   CUDA stream
 * @param init
 *   if true dest will be initialized with ReduceOp::Init()
 *   otherwise the values in the destination will be included
 *   in the reduction.
 */
template <typename T, int RANK, typename InType, typename ReduceOp>
void inline reduce(tensor_t<T, RANK> dest, tensor_t<index_t, RANK> idest, InType in, ReduceOp op,
                   cudaStream_t stream = 0, bool init = true)
{
#ifdef __CUDACC__  
  using scalar_type = typename InType::scalar_type;

  static_assert(RANK < InType::Rank());
  if (idest.Data() != nullptr) {
    MATX_ASSERT_STR(is_matx_index_reduction_v<ReduceOp>, matxInvalidParameter, "Must use a reduction operator capable of saving indices");
  }

  static_assert(is_matx_reduction_v<ReduceOp>,  "Must use a reduction operator for reducing");    


  if constexpr (RANK > 0) {
    for (uint32_t i = 0; i < RANK; i++) {
      MATX_ASSERT(dest.Size(i) == in.Size(i), matxInvalidDim);
    }
  }
  dim3 blocks, threads;

  if constexpr (InType::Rank() == 1) {
    get_grid_dims(blocks, threads, in.Size(0));
  }
  else if constexpr (InType::Rank() == 2) {
    get_grid_dims(blocks, threads, in.Size(0), in.Size(1));
  }
  else if constexpr (InType::Rank() == 3) {
    get_grid_dims(blocks, threads, in.Size(0), in.Size(1), in.Size(2));
  }
  else if constexpr (InType::Rank() == 4) {
    get_grid_dims(blocks, threads, in.Size(0), in.Size(1), in.Size(2),
                  in.Size(3));
  }
  if (init) {
    (dest = static_cast<promote_half_t<T>>(op.Init())).run(stream);
  }

  matxReduceKernel<<<blocks, threads, sizeof(scalar_type) * 32, stream>>>(
      dest, in, ReduceOp());

  // If we need the indices too, launch that kernel
  if (idest.Data() != nullptr) {
    (idest = std::numeric_limits<index_t>::max()).run(stream);
    matxIndexKernel<<<blocks, threads, 0, stream>>>(
        idest, dest, in);     
  }
#endif  
}



/**
 * Perform a reduction
 *
 * Performs a reduction from tensor "in" into tensor "dest" using reduction
 * operation ReduceOp. The output tensor dictates which elements the reduction
 * is performed over. In general, the reductions are performed over the
 * innermost dimensions, where the number of dimensions is the difference
 * between the input and output tensor ranks. For example, for a 0D (scalar)
 * output tensor, the reduction is performed over the entire tensor. For
 * anything higher, the reduction is performed across the number of ranks below
 * the input tensor that the output tensor is. For example, if the input tensor
 * is a 4D tensor and the output is a 1D tensor, the reduction is performed
 * across the innermost dimension of the input. If the output is a 2D tensor,
 * the reduction is performed across the two innermost dimensions of the input,
 * and so on.
 *
 * @tparam T
 *   Output data type
 * @param RANK
 *   Rank of output value tensor
 * @tparam InType
 *   Input data type
 * @tparam ReduceOp
 *   Reduction operator to apply
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param op
 *   Reduction operator
 * @param stream
 *   CUDA stream
 * @param init
 *   if true dest will be initialized with ReduceOp::Init()
 *   otherwise the values in the destination will be included
 *   in the reduction.
 */
template <typename T, int RANK, typename InType, typename ReduceOp>
void inline reduce(tensor_t<T, RANK> dest, InType in, ReduceOp op,
                   cudaStream_t stream = 0, bool init = true)
{
  auto tmp = tensor_t<index_t, RANK>{nullptr, dest.Shape()};
  reduce(dest, tmp, in, op, stream, init);
}

/**
 * Calculate the mean of values in a tensor
 *
 * Performs a sum reduction from tensor "in" into tensor "dest" , followed by
 * a division by the number of elements in the reduction. Similar to the reduce
 * function, the type of reduction is dependent on the rank of the output
 * tensor. A single value denotes a reduction over the entire input, a 1D tensor
 * denotes a reduction over each row independently, etc.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline mean(tensor_t<T, RANK> &dest, const InType &in,
                 cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  float scale = 1.0;

  reduce(dest, in, reduceOpSum<T>(), stream);

  // The reduction is performed over the difference in ranks between input and
  // output. This loop computes the number of elements it was performed over.
  for (int i = 1; i <= InType::Rank() - RANK; i++) {
    scale *= static_cast<float>(in.Size(InType::Rank() - i));
  }

  (dest = dest * 1.0 / scale).run(stream);
#endif  
}

/**
 * Calculate the median of values in a tensor
 *
 * Calculates the median of rows in a tensor. The median is computed by sorting
 * the data into a temporary tensor, then picking the middle element of each
 * row. For an even number of items, the mean of the two middle elements is
 * selected. Currently only works on tensor views as input since it uses CUB
 * sorting as a backend, and the tensor views must be rank 2 reducing to rank 1,
 * or rank 1 reducing to rank 0.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam RANK_IN
 *   Input rank
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, int RANK_IN>
void inline median(tensor_t<T, RANK> &dest,
                   const tensor_t<T, RANK_IN> &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  static_assert(RANK_IN <= 2 && (RANK_IN == RANK + 1));

  tensor_t<T, RANK_IN> tmp_sort(in.Shape());

  // If the rank is 0 we're finding the median of a vector
  if constexpr (RANK_IN == 1) {
    matxCubPlan_t<T, T, RANK_IN, CUB_OP_RADIX_SORT> splan{
        tmp_sort, in, {}, stream};

    splan.ExecSort(tmp_sort, in, stream, SORT_DIR_ASC);

    // Store median
    if (tmp_sort.Lsize() & 1) {
      auto middlev =
          tmp_sort.template Slice<0>({tmp_sort.Lsize() / 2}, {matxDropDim});
      copy(dest, middlev, stream);
    }
    else {
      auto middle1v =
          tmp_sort.template Slice<0>({tmp_sort.Lsize() / 2 - 1}, {matxDropDim});
      auto middle2v =
          tmp_sort.template Slice<0>({tmp_sort.Lsize() / 2}, {matxDropDim});
      (dest = (middle1v + middle2v) / 2.0f).run(stream);
    }
  }
  else if (RANK_IN == 2) {
    MATX_ASSERT(dest.Size(0) == in.Size(0), matxInvalidSize);

    matxCubPlan_t<T, T, RANK_IN, CUB_OP_RADIX_SORT> splan{
        tmp_sort, in, {}, stream};
    splan.ExecSort(tmp_sort, in, stream, SORT_DIR_ASC);

    if (tmp_sort.Lsize() & 1) {
      auto sv = tmp_sort.template Slice<1>({0, tmp_sort.Lsize() / 2},
                                           {matxEnd, matxDropDim});
      (dest = self(sv)).run(stream);
    }
    else {
      auto sv = tmp_sort.template Slice<1>({0, tmp_sort.Lsize() / 2 - 1},
                                           {matxEnd, matxDropDim});
      auto sv2 = tmp_sort.template Slice<1>({0, tmp_sort.Lsize() / 2},
                                            {matxEnd, matxDropDim});
      (dest = (sv + sv2) / 2.0f).run(stream);
    }
  }
#endif  
}

/**
 * Compute sum of numbers
 *
 * Returns a vector representing the sum of all numbers in the reduction
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline sum(tensor_t<T, RANK> dest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  reduce(dest, in, reduceOpSum<T>(), stream, true);
#endif  
}

/**
 * Compute product of numbers
 *
 * Returns a vector representing the product of all numbers in the reduction
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline prod(tensor_t<T, RANK> dest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  reduce(dest, in, reduceOpProd<T>(), stream, true);
#endif  
}

/**
 * Compute max reduction of a tensor
 *
 * Returns a vector representing the max of all numbers in the reduction
 *
 * @note This function uses the name rmax instead of max to not collide with the
 * element-wise operator max.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline rmax(tensor_t<T, RANK> dest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  reduce(dest, in, reduceOpMax<T>(), stream, true);
#endif  
}

/**
 * Compute maxn reduction of a tensor and returns value + index
 *
 * Returns a tensor with maximums and indices
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction values
 * @param idest
 *   Destination view of reduction indices
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline argmax(tensor_t<T, RANK> dest, tensor_t<index_t, RANK> idest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  reduce(dest, idest, in, reduceOpMax<T>(), stream, true);
#endif  
}

/**
 * Compute min reduction of a tensor
 *
 * Returns a vector representing the min of all numbers in the reduction
 *
 * @note This function uses the name rmin instead of min to not collide with the
 * element-wise operator min.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline rmin(tensor_t<T, RANK> dest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  reduce(dest, in, reduceOpMin<T>(), stream, true);
#endif  
}

/**
 * Compute min reduction of a tensor and returns value + index
 *
 * Returns a tensor with minimums and indices
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction values
 * @param idest
 *   Destination view of reduction indices
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline argmin(tensor_t<T, RANK> dest, tensor_t<index_t, RANK> idest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  reduce(dest, idest, in, reduceOpMin<T>(), stream, true);
#endif  
}

/**
 * Find if any value is != 0
 *
 * Returns a boolean value indicating whether any value in the set of inputs are
 * non-zero. The same aggregation rules apply for input vs output tensor size
 * and what type of reduction is done.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline any(tensor_t<T, RANK> dest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  reduce(dest, in, reduceOpAny<T>(), stream, true);
#endif  
}

/**
 * Find if all values are != 0
 *
 * Returns a boolean value indicating whether all values in the set of inputs
 * are non-zero. The same aggregation rules apply for input vs output tensor
 * size and what type of reduction is done.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline all(tensor_t<T, RANK> dest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  reduce(dest, in, reduceOpAll<T>(), stream, true);
#endif  
}

/**
 * Compute a variance reduction
 *
 * Computes the variance of the input according to the output tensor rank and
 * size
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline var(tensor_t<T, RANK> dest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  T *tmps;
  matxAlloc((void **)&tmps, dest.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
  auto tmpv = tensor_t<T, RANK>(dest);
  tmpv.SetData(tmps);

  // Compute mean of each dimension
  mean(tmpv, in, stream);

  // Subtract means from each value, square the result, and sum
  sum(dest, pow(in - tmpv, 2), stream);

  // The length of what we are taking the variance over is equal to the product
  // of the outer dimensions covering the different in input/output ranks
  index_t N = in.Size(in.Rank() - 1);
  for (int i = 2; i <= in.Rank() - RANK; i++) {
    N *= in.Size(in.Rank() - i);
  }

  // Sample variance for an unbiased estimate
  (dest = dest / static_cast<double>(N - 1)).run(stream);

  matxFree(tmps);
#endif  
}

/**
 * Compute a standard deviation reduction
 *
 * Computes the standard deviation of the input according to the output tensor
 * rank and size
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream
 */
template <typename T, int RANK, typename InType>
void inline stdd(tensor_t<T, RANK> dest, InType in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  var(dest, in, stream);
  (dest = sqrt(dest)).run(stream);
#endif  
}

} // end namespace matx
