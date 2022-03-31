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
template <typename T>
__MATX_DEVICE__ inline void atomicAny(T *addr, T val)
{
  // We don't actually need an atomic operation here since we only write to the
  // location if any thread has the correct value.
  if (val != T(0)) {
    *addr = T(1);
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

__MATX_DEVICE__ inline void atomicAll(int *addr, int val)
{
  int assumed;
  int old = *addr;

  // nan should be ok here but should verify
  while (val == 0 && old != 0) {
    assumed = old;
    old = atomicCAS(addr, assumed, 0);
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
__MATX_DEVICE__ inline void atomicAll(double *addr, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;

  // nan should be ok here but should verify
  while (val == 0.0 && old != 0.0) {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, 0.0);
  }
}


/**
 * Atomic min version for int64_t
 *
 * Computes the minimum of two int64_t values atomically
 *
 * @param addr
 *   Source and destination for new minimum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicMin(int64_t *addr, int64_t val)
{
  atomicMin(reinterpret_cast<long long int*>(addr), static_cast<long long int>(val));
};

/**
 * Atomic max version for int64_t
 *
 * Computes the maximum of two int64_t values atomically
 *
 * @param addr
 *   Source and destination for new maximum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicMax(int64_t *addr, int64_t val)
{
  atomicMax(reinterpret_cast<long long int*>(addr), static_cast<long long int>(val));
};


/**
 * Atomic all version for int64_t
 *
 * Computes whether both int64_t values are != 0
 *
 * @param addr
 *   Source and destination for new any
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicAll(int64_t *addr, int64_t val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;

  // nan should be ok here but should verify
  while (val == 0 && old != 0) {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, 0);
  }
}

/**
 * Atomic min version for uint64_t
 *
 * Computes the minimum of two uint64_t values atomically
 *
 * @param addr
 *   Source and destination for new minimum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicMin(uint64_t *addr, uint64_t val)
{
  atomicMin(reinterpret_cast<unsigned long long int*>(addr), static_cast<unsigned long long int>(val));
};

/**
 * Atomic max version for uint64_t
 *
 * Computes the maximum of two uint64_t values atomically
 *
 * @param addr
 *   Source and destination for new maximum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicMax(uint64_t *addr, uint64_t val)
{
  atomicMax(reinterpret_cast<unsigned long long int*>(addr), static_cast<unsigned long long int>(val));
};


/**
 * Atomic all version for uint64_t
 *
 * Computes whether both uint64_t values are != 0
 *
 * @param addr
 *   Source and destination for new any
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ inline void atomicAll(uint64_t *addr, uint64_t val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;

  // nan should be ok here but should verify
  while (val == 0 && old != 0) {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, 0);
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
namespace detail {

#ifdef __CUDACC__  
template <typename T> constexpr inline __MATX_HOST__ __MATX_DEVICE__ T maxVal() { return std::numeric_limits<T>::max(); }
template <typename T> constexpr inline __MATX_HOST__ __MATX_DEVICE__ T minVal() { return std::numeric_limits<T>::min(); }


/**
 * Operator for performing a sum reduction
 *
 * Performs a reduction of two values of type T by summing the
 * values
 */
template <typename T> class reduceOpSum {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Reduce(const T &v1, const T &v2) { return v1 + v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) { Reduce(v1, v2); }  
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return T(0); }
  __MATX_DEVICE__ __MATX_INLINE__ void atomicReduce(T *addr, T val) { atomicAdd(addr, val); }
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
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Reduce(const T &v1, const T &v2) { return v1 * v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(T &v1, T &v2) { v1 *= v2; return v1; }  
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return T(1); }
  __MATX_DEVICE__ __MATX_INLINE__ void atomicReduce(T *addr, T val) { atomicMul(addr, val); }
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
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Reduce(const T &v1, const T &v2) { return v1 > v2 ? v1 : v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) { Reduce(v1, v2); }  
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return minVal<T>(); }
  __MATX_DEVICE__ __MATX_INLINE__ void atomicReduce(T *addr, T val) { atomicMax(addr, val); }
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
  using matx_no_cub_reduce = bool; // Don't use CUB for this reduction type
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Reduce(const T &v1, const T &v2)
  {
    return (v1 != 0) || (v2 != 0);
  }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(T &v1, T &v2) { v1 = ((v1 != 0) || (v2 != 0)); return v1; }  
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return (T)(0); }
  __MATX_DEVICE__ __MATX_INLINE__ void atomicReduce(T *addr, T val) { atomicAny(addr, val); }
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
  using matx_no_cub_reduce = bool; // Don't use CUB for this reduction type
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Reduce(const T &v1, const T &v2)
  {
    return (v1 != 0) && (v2 != 0);
  }

  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(T &v1, T &v2) { v1 = ((v1 != 0) && (v2 != 0)); return v1; }  
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return (T)(1); }
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
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Reduce(const T &v1, const T &v2) { return v1 < v2 ? v1 : v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) { Reduce(v1, v2); } 
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return maxVal<T>(); }
  __MATX_DEVICE__ __MATX_INLINE__ void atomicReduce(T *addr, T val) { atomicMin(addr, val); }
};


template <typename T, typename Op>
__MATX_DEVICE__ __MATX_INLINE__ T warpReduceOp(T val, Op op, uint32_t size)
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


template <typename TensorType, typename InType, typename ReduceOp>
__global__ void matxReduceKernel(TensorType dest, InType in,
                                 ReduceOp red, [[maybe_unused]] index_t mult)
{
  constexpr uint32_t RANK = TensorType::Rank();
  constexpr uint32_t DRANK = InType::Rank() - RANK;
  std::array<index_t, InType::Rank()> indices;
  using scalar_type = typename InType::scalar_type;
  using T = typename TensorType::scalar_type;
  [[maybe_unused]] bool valid;
  
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
  T in_val = red.Init();

  if constexpr (InType::Rank() == 1) {
    indices[InType::Rank()-1] = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (indices[InType::Rank()-1] < in.Size(0)) {
      in_val = in(indices[InType::Rank()-1]);
    }
  }
  else if constexpr (InType::Rank() == 2) {
    indices[InType::Rank()-1] = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    indices[InType::Rank()-2] = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (indices[InType::Rank()-2] < in.Size(0) && indices[InType::Rank()-1] < in.Size(1)) {
      in_val = in(indices[InType::Rank()-2], indices[InType::Rank()-1]);
    }
  }
  else if constexpr (InType::Rank() == 3) {
    indices[InType::Rank()-1] = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    indices[InType::Rank()-2] = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    indices[InType::Rank()-3] = static_cast<index_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (indices[InType::Rank()-3] < in.Size(0) && indices[InType::Rank()-2] < in.Size(1) && indices[InType::Rank()-1] < in.Size(2)) {
      in_val = in(indices[InType::Rank()-3], indices[InType::Rank()-2], indices[InType::Rank()-1]);
    }
  }
  else if constexpr (InType::Rank() == 4) {
    indices[InType::Rank()-1] = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    index_t nmy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    indices[InType::Rank()-2] = nmy % in.Size(2);
    indices[InType::Rank()-3] = nmy / in.Size(2);
    indices[InType::Rank()-4] = blockIdx.z * blockDim.z + threadIdx.z;

    if (indices[InType::Rank()-4] < in.Size(0) && indices[InType::Rank()-3] < in.Size(1) && indices[InType::Rank()-2] < in.Size(2) &&
        indices[InType::Rank()-1] < in.Size(3)) {
      in_val = in(indices[InType::Rank()-4], indices[InType::Rank()-3], indices[InType::Rank()-2], indices[InType::Rank()-1]);   
    }
  }
  else {
    // Compute the index into the operator for this thread. N-D tensors require more computations
    // since we're limited to 3 dimensions in both grid and block, so we need to iterate to compute
    // our index.
    index_t x_abs = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    index_t ttl = mult * in.Size(0);
    valid = x_abs < ttl;
    #pragma unroll
    for (int r = 0; r < InType::Rank(); r++) {
      indices[r] = x_abs / mult;
      x_abs -= indices[r] * mult;      
      mult /= in.Size(r+1); 
    }
    
    if (valid) {
      in_val = in(indices);  
    }     
  }  

  // Compute offset index based on rank difference
  #pragma unroll
  for (int r = 0; r < InType::Rank() - DRANK; r++) {
    indices[InType::Rank() - r - 1] = indices[InType::Rank() - (DRANK + 1 + r)];
  }

  // Compute output location
  T *out = nullptr;

  // compute output offsets
  if constexpr (RANK == 0) {
    out = &dest();
  }
  else if constexpr (RANK == 1) {
    if (indices[InType::Rank()-1] < dest.Size(0))
      out = &dest(indices[InType::Rank()-1]);
  }
  else if constexpr (RANK == 2) {
    if (indices[InType::Rank()-1] < dest.Size(1) && indices[InType::Rank()-2] < dest.Size(0))
      out = &dest(indices[InType::Rank()-2], indices[InType::Rank()-1]);
  }
  else if constexpr (RANK == 3) {
    if (indices[InType::Rank()-1] < dest.Size(2) && indices[InType::Rank()-2] < dest.Size(1) && indices[InType::Rank()-3] < dest.Size(0))
      out = &dest(indices[InType::Rank()-3], indices[InType::Rank()-2], indices[InType::Rank()-1]);
  }
  else {
    // Calculate valid here
    valid = true;
    for (int r = 0; r < RANK - 1; r++) {
      if (indices[r] >= dest.Size(r)) {
        valid = false;
      }
    }

    if (valid) {
      out  = mapply([&] (auto... param) { return dest.GetPointer(param...); }, indices); 
    }   
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

template <typename TensorType, typename TensorIndexType, typename InType>
__global__ void matxIndexKernel(TensorType dest, TensorIndexType idest, InType in, [[maybe_unused]] index_t mult)
{
  using index_type = typename TensorIndexType::scalar_type;
  using T = typename TensorType::scalar_type;
  index_type in_val;
  constexpr uint32_t RANK = TensorIndexType::Rank();
  constexpr uint32_t DRANK = InType::Rank() - RANK;  
  std::array<index_t, InType::Rank()> indices;
  index_t abs_idx;
  bool valid = false;
  
  if constexpr (InType::Rank() == 1) {
    indices[0] = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (indices[InType::Rank()-1] < in.Size(0)) {
      in_val = in(indices[0]);
      abs_idx = indices[InType::Rank()-1];
    }
  }
  else if constexpr (InType::Rank() == 2) {
    indices[InType::Rank()-1] = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    indices[InType::Rank()-2] = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (indices[InType::Rank()-2] < in.Size(0) && indices[InType::Rank()-1] < in.Size(1)) {
      in_val = in(indices[InType::Rank()-2], indices[InType::Rank()-1]);
      abs_idx = indices[InType::Rank()-2]*in.Size(1) + indices[InType::Rank()-1];
    }
  }
  else if constexpr (InType::Rank() == 3) {
    indices[InType::Rank()-1] = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    indices[InType::Rank()-2] = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    indices[InType::Rank()-3] = static_cast<index_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (indices[InType::Rank()-3] < in.Size(0) && indices[InType::Rank()-2] < in.Size(1) && indices[InType::Rank()-1] < in.Size(2)) {
      in_val = in(indices[InType::Rank()-3], indices[InType::Rank()-2], indices[InType::Rank()-1]);
      abs_idx = indices[InType::Rank()-3]*in.Size(0)*in.Size(1) + indices[InType::Rank()-2]*in.Size(1) + indices[InType::Rank()-1];
    }
  }
  else if constexpr (InType::Rank() == 4) {
    indices[InType::Rank()-1] = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    index_t nmy = static_cast<index_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    indices[InType::Rank()-2] = nmy % in.Size(2);
    indices[InType::Rank()-3] = nmy / in.Size(2);
    indices[InType::Rank()-4] = blockIdx.z * blockDim.z + threadIdx.z;
    if (indices[InType::Rank()-4] < in.Size(0) && indices[InType::Rank()-3] < in.Size(1) && indices[InType::Rank()-2] < in.Size(2) &&
        indices[InType::Rank()-1] < in.Size(3)) {
      in_val = in(indices[InType::Rank()-4], indices[InType::Rank()-3], indices[InType::Rank()-2], indices[InType::Rank()-1]);
      abs_idx = indices[InType::Rank()-4]*in.Size(0)*in.Size(1)*in.Size(2) + indices[InType::Rank()-3]*in.Size(0)*in.Size(1) + indices[InType::Rank()-2]*in.Size(1) + indices[InType::Rank()-1];
    }
  }
  else {
    // Compute the index into the operator for this thread. N-D tensors require more computations
    // since we're limited to 3 dimensions in both grid and block, so we need to iterate to compute
    // our index.
    index_t x_abs = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    abs_idx = x_abs;
    index_t ttl = mult * in.Size(0);
    valid = x_abs < ttl;
    #pragma unroll
    for (int r = 0; r < InType::Rank(); r++) {
      indices[r] = x_abs / mult;
      x_abs -= indices[r] * mult;      
      mult /= in.Size(r+1); 
    }    

    if (valid) {
      in_val = in(indices);  
    }
  }

  #pragma unroll
  for (int r = 0; r < InType::Rank() - DRANK; r++) {
    indices[InType::Rank() - r - 1] = indices[InType::Rank() - (DRANK + 1 + r)];
  }

  // Compute output location
  T *out = nullptr;
  index_type *iout = nullptr;

  // compute output offsets
  if constexpr (RANK == 0) {
    out  = &dest();
    iout = &idest();
    valid = true;
  }
  else if constexpr (RANK == 1) {
    if (indices[InType::Rank()-1] < dest.Size(0)) {
      out = &dest(indices[InType::Rank()-1]);
      iout = &idest(indices[InType::Rank()-1]);
      valid = true;
    }
  }
  else if constexpr (RANK == 2) {
    if (indices[InType::Rank()-1] < dest.Size(1) && indices[InType::Rank()-2] < dest.Size(0))
      out = &dest(indices[InType::Rank()-2], indices[InType::Rank()-1]);
      iout = &idest(indices[InType::Rank()-2], indices[InType::Rank()-1]);
      valid = true;
  }
  else if constexpr (RANK == 3) {
    if (indices[InType::Rank()-1] < dest.Size(2) && indices[InType::Rank()-2] < dest.Size(1) && indices[InType::Rank()-3] < dest.Size(0))
      out = &dest(indices[InType::Rank()-3], indices[InType::Rank()-2], indices[InType::Rank()-1]);
      iout = &idest(indices[InType::Rank()-3], indices[InType::Rank()-2], indices[InType::Rank()-1]);
      valid = true;
  }
  else {
    // Calculate valid here
    valid = true;
    for (int r = RANK-1; r >= 0; r++) {
      if (indices[r] >= dest.Size(r)) {
        valid = false;
      }
    }
    if (valid) {
      iout = mapply([&](auto... param) { return idest.GetPointer(param...); }, indices);
      out  = mapply([&] (auto... param) { return dest.GetPointer(param...); }, indices); 
    }   
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

} // namespace detail

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
 * @tparam TensorType
 *   Output data type
 * @tparam TensorIndexType
 *   Output index type
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
template <typename TensorType, typename TensorIndexType, typename InType, typename ReduceOp,
  std::enable_if_t<is_matx_reduction_v<ReduceOp>, bool> = true>
void inline reduce(TensorType dest, [[maybe_unused]] TensorIndexType idest, InType in, ReduceOp op,
                   cudaStream_t stream = 0, bool init = true)
{
#ifdef __CUDACC__  
  using scalar_type = typename InType::scalar_type;
  using T = typename TensorType::scalar_type;

  static_assert(TensorType::Rank() < InType::Rank());
  static_assert(is_matx_reduction_v<ReduceOp>,  "Must use a reduction operator for reducing");    


  if constexpr (TensorType::Rank() > 0) {
    for (int i = 0; i < TensorType::Rank(); i++) {
      MATX_ASSERT(dest.Size(i) == in.Size(i), matxInvalidDim);
    }
  }
  dim3 blocks, threads;
  std::array<index_t, in.Rank()> sizes;
  for (int i = 0; i < in.Rank(); i++) {
    sizes[i] = in.Size(i);
  }   

  detail::get_grid_dims<in.Rank()>(blocks, threads, sizes);
  
  if (init) {
    (dest = static_cast<promote_half_t<T>>(op.Init())).run(stream);
  }

  auto mult = std::accumulate(sizes.begin() + 1, sizes.end(), 1, std::multiplies<index_t>());
  detail::matxReduceKernel<<<blocks, threads, sizeof(scalar_type) * 32, stream>>>(
      dest, in, ReduceOp(), mult);

  // If we need the indices too, launch that kernel
  if constexpr (!std::is_same_v<TensorIndexType, std::nullopt_t>) {
    (idest = std::numeric_limits<index_t>::max()).run(stream);
    detail::matxIndexKernel<<<blocks, threads, 0, stream>>>(
        dest, idest, in, mult);     
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
 * @tparam TensorType
 *   Output data type
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
template <typename TensorType, typename InType, typename ReduceOp>
void inline reduce(TensorType &dest, const InType &in, ReduceOp op,
                   cudaStream_t stream = 0, [[maybe_unused]] bool init = true)
{
  constexpr bool use_cub = TensorType::Rank() == 0 || (TensorType::Rank() == 1 && InType::Rank() == 2);
  // Use CUB implementation if we have a tensor on the RHS and it's not blocked from using CUB
  if constexpr (!is_matx_no_cub_reduction_v<ReduceOp> && use_cub) {
    cub_reduce<TensorType, InType, ReduceOp>(dest, in, op.Init(), stream);
  }
  else { // Fall back to the slow path of custom implementation
    reduce(dest, std::nullopt, in, op, stream, init);
  }
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
template <typename TensorType, typename InType>
void inline mean(TensorType &dest, const InType &in,
                 cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  float scale = 1.0;

  sum(dest, in, stream);

  // The reduction is performed over the difference in ranks between input and
  // output. This loop computes the number of elements it was performed over.
  for (int i = 1; i <= InType::Rank() - TensorType::Rank(); i++) {
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
template <typename TensorType, typename TensorInType>
void inline median(TensorType &dest,
                   const TensorInType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  using T = typename TensorType::scalar_type;
  constexpr int RANK_IN = TensorInType::Rank();
  static_assert(RANK_IN <= 2 && (RANK_IN == TensorType::Rank() + 1));

  auto tmp_sort = make_tensor<T>(in.Shape());

  // If the rank is 0 we're finding the median of a vector
  if constexpr (RANK_IN == 1) {
    matx::sort(tmp_sort, in, SORT_DIR_ASC, stream);

    // Store median
    if (tmp_sort.Lsize() & 1) {
      auto middlev =
          tmp_sort.template Slice<0>({tmp_sort.Lsize() / 2}, {matxDropDim});
      matx::copy(dest, middlev, stream);
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

    matx::sort(tmp_sort, in, SORT_DIR_ASC, stream);

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
template <typename TensorType, typename InType>
void inline sum(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  constexpr bool use_cub = TensorType::Rank() == 0 || (TensorType::Rank() == 1 && InType::Rank() == 2);
  // Use CUB implementation if we have a tensor on the RHS
  if constexpr (use_cub) {
    cub_sum<TensorType, InType>(dest, in, stream);
  }
  else { // Fall back to the slow path of custom implementation
    reduce(dest, in, detail::reduceOpSum<typename TensorType::scalar_type>(), stream, true);
  }
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
template <typename TensorType, typename InType>
void inline prod(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  reduce(dest, in, detail::reduceOpProd<typename TensorType::scalar_type>(), stream, true);
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
template <typename TensorType, typename InType>
void inline rmax(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  constexpr bool use_cub = TensorType::Rank() == 0 || (TensorType::Rank() == 1 && InType::Rank() == 2);
  // Use CUB implementation if we have a tensor on the RHS
  if constexpr (use_cub) {
    cub_max<TensorType, InType>(dest, in, stream);
  }
  else { // Fall back to the slow path of custom implementation
    reduce(dest, in, detail::reduceOpMax<typename TensorType::scalar_type>(), stream, true);
  }
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
template <typename TensorType, typename TensorIndexType, typename InType>
void inline argmax(TensorType &dest, TensorIndexType &idest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  reduce(dest, idest, in, detail::reduceOpMax<typename TensorType::scalar_type>(), stream, true);
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
template <typename TensorType, typename InType>
void inline rmin(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  constexpr bool use_cub = TensorType::Rank() == 0 || (TensorType::Rank() == 1 && InType::Rank() == 2);
  // Use CUB implementation if we have a tensor on the RHS
  if constexpr (use_cub) {
    cub_min<TensorType, InType>(dest, in, stream);
  }
  else { // Fall back to the slow path of custom implementation
    reduce(dest, in, detail::reduceOpMin<typename TensorType::scalar_type>(), stream, true);
  }
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
template <typename TensorType, typename TensorIndexType, typename InType>
void inline argmin(TensorType &dest, TensorIndexType &idest, const InType &in, cudaStream_t stream = 0)
{
  static_assert(TensorType::Rank() == TensorIndexType::Rank());
#ifdef __CUDACC__  
  reduce(dest, idest, in, detail::reduceOpMin<typename TensorType::scalar_type>(), stream, true);
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
template <typename TensorType, typename InType>
void inline any(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  reduce(dest, in, detail::reduceOpAny<typename TensorType::scalar_type>(), stream, true);
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
template <typename TensorType, typename InType>
void inline all(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  reduce(dest, in, detail::reduceOpAll<typename TensorType::scalar_type>(), stream, true);
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
template <typename TensorType, typename InType>
void inline var(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__    
  typename TensorType::scalar_type *tmps;
  matxAlloc((void **)&tmps, dest.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
  auto tmpv = make_tensor(tmps, dest.Descriptor());

  // Compute mean of each dimension
  mean(tmpv, in, stream);

  // Subtract means from each value, square the result, and sum
  sum(dest, pow(in - tmpv, 2), stream);

  // The length of what we are taking the variance over is equal to the product
  // of the outer dimensions covering the different in input/output ranks
  index_t N = in.Size(in.Rank() - 1);
  for (int i = 2; i <= in.Rank() - TensorType::Rank(); i++) {
    N *= in.Size(in.Rank() - i);
  }

  // Sample variance for an unbiased estimate
  (dest = dest / static_cast<double>(N - 1)).run(stream);
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
template <typename TensorType, typename InType>
void inline stdd(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  var(dest, in, stream);
  (dest = sqrt(dest)).run(stream);
#endif  
}

/**
 * Computes the trace of a tensor
 *
 * Computes the trace of a square matrix by summing the diagonal
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
template <typename TensorType, typename InType>
void inline trace(TensorType &dest, const InType &in, cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  auto d = diag(in);
  sum(dest, d, stream);
#endif  
}

} // end namespace matx
