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

#include <cfloat>

#include "matx/core/cache.h"
#include "matx/core/error.h"
#include "matx/core/get_grid_dims.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/type_utils.h"
#include "matx/core/utils.h"
#include "matx/transforms/cub.h"
#include "matx/core/half.h"

union HalfBits {
  constexpr HalfBits(short x) : i(x) {}
  HalfBits() = default;
  short i;
  __half h;
  __nv_bfloat16 b;
};

union PascalHalfBits {
  constexpr PascalHalfBits(unsigned short x) : i(x) {}
  PascalHalfBits() = default;
  unsigned int i;
  __half h[2];
  __nv_bfloat16 b[2];
};

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
__MATX_DEVICE__ __MATX_INLINE__ auto __shfl_down_sync(unsigned mask,
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
__MATX_DEVICE__ __MATX_INLINE__ auto __shfl_down_sync(unsigned mask,
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicMin(float *addr, float val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicMax(float *addr, float val)
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
 * Atomic max version for bfloat16
 *
 * Computes the maximum of two matxBf16 values atomically
 *
 * @param addr
 *   Source and destination for new maximum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ __MATX_INLINE__ __nv_bfloat16 atomicMax(__nv_bfloat16 *addr, __nv_bfloat16 val)
{
#if __CUDA_ARCH__ > 600      
  HalfBits tmpval;
  HalfBits old;
  unsigned short *address_as_other = (unsigned short *)addr;

  unsigned short assumed;
  old.i = *address_as_other;
  tmpval.b = val;

  // nan should be ok here but should verify
  while (val > old.b) {
    assumed = old.i;
    old.b = static_cast<float>(atomicCAS(address_as_other, assumed, tmpval.i));
  }

  return old.b;  
#else // Pascal doesn't support short atomicCAS
  PascalHalfBits tmpval;
  PascalHalfBits old;
  unsigned int *address_as_other;
  int offset;

  // We need to move our pointer back 
  if ((uintptr_t)addr & 0x10) {
    address_as_other = (unsigned int *)(reinterpret_cast<uint8_t*>(addr) - 2);
    offset = 1;
  }
  else {
    address_as_other = (unsigned int *)(addr);
    offset = 0;
  }

  unsigned short assumed;
  old.i = *address_as_other;
  tmpval.b[offset] = val;

  // nan should be ok here but should verify
  while (val > old.b[offset]) {
    assumed = old.i;
    old.b[offset] = static_cast<float>(atomicCAS(address_as_other, assumed, tmpval.i));
  }

  return old.b[offset];
#endif  
};


/**
 * Atomic max version for fp16
 *
 * Computes the maximum of two matxFp16 values atomically
 *
 * @param addr
 *   Source and destination for new maximum
 * @param val
 *   Value to compare against
 */
__MATX_DEVICE__ __MATX_INLINE__ __half atomicMax(__half *addr, __half val)
{
#if __CUDA_ARCH__ > 600      
  HalfBits tmpval;
  HalfBits old;
  unsigned short *address_as_other = (unsigned short *)addr;

  unsigned short assumed;
  old.i = *address_as_other;
  tmpval.h = val;

  // nan should be ok here but should verify
  while (val > old.h) {
    assumed = old.i;
    old.h = atomicCAS(address_as_other, assumed, tmpval.i);
  }

  return old.h;  
#else // Pascal doesn't support short atomicCAS
  PascalHalfBits tmpval;
  PascalHalfBits old;
  unsigned int *address_as_other;
  int offset;

  // We need to move our pointer back to align to a 2b boundary
  if ((uintptr_t)addr & 0x10) {
    address_as_other = (unsigned int *)(reinterpret_cast<uint8_t*>(addr) - 2);
    offset = 1;
  }
  else {
    address_as_other = (unsigned int *)(addr);
    offset = 0;
  }

  unsigned short assumed;
  old.i = *address_as_other;
  tmpval.h[offset] = val;

  // nan should be ok here but should verify
  while (val > old.h[offset]) {
    assumed = old.i;
    old.h[offset] = atomicCAS(address_as_other, assumed, tmpval.i);
  }

  return old.h[offset];
#endif    
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicAny(T *addr, T val)
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
template <> __MATX_DEVICE__ __MATX_INLINE__ float atomicMul(float *address, float val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicAll(float *addr, float val)
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

__MATX_DEVICE__ __MATX_INLINE__ void atomicAll(int *addr, int val)
{
  int assumed;
  int old = *addr;

  while (val == 0 && old != 0) {
    assumed = old;
    old = atomicCAS(addr, assumed, 0);
  }
};

__MATX_DEVICE__ __MATX_INLINE__ void atomicAll(unsigned int *addr, unsigned int val)
{
  unsigned int assumed;
  unsigned int old = *addr;

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
template <> __MATX_DEVICE__ __MATX_INLINE__ double atomicMul(double *address, double val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicMin(double *addr, double val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicMax(double *addr, double val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicAny(double *addr, double val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicAll(double *addr, double val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicMin(int64_t *addr, int64_t val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicMax(int64_t *addr, int64_t val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicAll(int64_t *addr, int64_t val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicMin(uint64_t *addr, uint64_t val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicMax(uint64_t *addr, uint64_t val)
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicAll(uint64_t *addr, uint64_t val)
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
 * Atomic add for int64_t
 *
 * @param addr
 *   Source and destination for result
 * @param val
 *   Value to add
 */
__MATX_DEVICE__ __MATX_INLINE__ void atomicAdd(int64_t *addr,
                                 int64_t val)
{
  unsigned long long int *addri = reinterpret_cast<unsigned long long int *>(addr);
  atomicAdd(addri, (unsigned long long int)val);
}

/**
 * Atomic add for uint64_t
 *
 * @param addr
 *   Source and destination for result
 * @param val
 *   Value to add
 */
__MATX_DEVICE__ __MATX_INLINE__ void atomicAdd(uint64_t *addr,
                                 uint64_t val)
{
  unsigned long long int *addri = reinterpret_cast<unsigned long long int *>(addr);
  atomicAdd(addri, (unsigned long long int)val);
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicAdd(cuda::std::complex<float> *addr,
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
__MATX_DEVICE__ __MATX_INLINE__ void atomicAdd(cuda::std::complex<double> *addr,
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
template <typename T> constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T maxVal() { 
  if constexpr (std::is_same_v<convert_matx_type_t<T>, __half>) {
    constexpr HalfBits tmp{0x7BFF};
    return tmp.h;
  }
  if constexpr (std::is_same_v<convert_matx_type_t<T>, __nv_bfloat16>) {
    constexpr HalfBits tmp{0x7F7F};
    return tmp.b;
  }  
  else {
    return cuda::std::numeric_limits<T>::max(); 
  }
}

template <typename T> constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T minVal() { 
  if constexpr (std::is_same_v<convert_matx_type_t<T>, __half>) {
    constexpr HalfBits tmp{0x0400};
    return tmp.h;
  }
  if constexpr (std::is_same_v<convert_matx_type_t<T>, __nv_bfloat16>) {
    constexpr HalfBits tmp{0x0080};
    return tmp.b;
  }  
  else {  
    return cuda::std::numeric_limits<T>::lowest(); 
  }
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
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Reduce(const T &v1, const T &v2) const { return v1 + v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) const { return Reduce(v1, v2); }  
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return T(0); }
  __MATX_DEVICE__ __MATX_INLINE__ void atomicReduce(T *addr, T val) { 
    if constexpr (is_complex_v<T>) {
      if constexpr (is_matx_half_v<typename T::value_type>) {
        atomicAdd(&reinterpret_cast<typename T::value_type::value_type *>(addr)[0], static_cast<typename T::value_type::value_type>(val.real())); 
        atomicAdd(&reinterpret_cast<typename T::value_type::value_type *>(addr)[1], static_cast<typename T::value_type::value_type>(val.imag())); 
      }
      else {
        atomicAdd(&reinterpret_cast<typename T::value_type *>(addr)[0], val.real()); 
        atomicAdd(&reinterpret_cast<typename T::value_type *>(addr)[1], val.imag()); 
      }      
    }
    else {    
      if constexpr (is_matx_half_v<T>) {
        atomicAdd(reinterpret_cast<typename T::value_type *>(addr), static_cast<typename T::value_type>(val)); 
      }
      else {
        atomicAdd(addr, val); 
      }
    }
  }
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
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Reduce(const T &v1, const T &v2) const { return v1 * v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) const { return Reduce(v1, v2); }  
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
  __MATX_DEVICE__ __MATX_INLINE__ void atomicReduce(T *addr, T val) { 
    atomicMax(reinterpret_cast<convert_matx_type_t<T> *>(addr), static_cast<convert_matx_type_t<T>>(val)); 
  }
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
  __MATX_DEVICE__ __MATX_INLINE__ void atomicReduce(T *addr, T val) { atomicAll(addr, val); }
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
  if constexpr (is_complex_v<T>) {
    typename T::value_type re;
    typename T::value_type im;
    if constexpr (!is_matx_half_v<typename T::value_type>) {
      if (size > 16) {
        re = __shfl_down_sync(0xffffffff, val.real(), 16);
        im = __shfl_down_sync(0xffffffff, val.imag(), 16);
        val = op.Reduce(val, {re, im});
        re = __shfl_down_sync(0xffffffff, val.real(), 8);
        im = __shfl_down_sync(0xffffffff, val.imag(), 8);
        val = op.Reduce(val, {re, im});        
        re = __shfl_down_sync(0xffffffff, val.real(), 4);
        im = __shfl_down_sync(0xffffffff, val.imag(), 4);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, val.real(), 2);
        im = __shfl_down_sync(0xffffffff, val.imag(), 2);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, val.real(), 1);
        im = __shfl_down_sync(0xffffffff, val.imag(), 1);
        val = op.Reduce(val, {re, im});                                       
      }
      else if (size > 8) {
        re = __shfl_down_sync(0xffffffff, val.real(), 8);
        im = __shfl_down_sync(0xffffffff, val.imag(), 8);
        val = op.Reduce(val, {re, im});        
        re = __shfl_down_sync(0xffffffff, val.real(), 4);
        im = __shfl_down_sync(0xffffffff, val.imag(), 4);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, val.real(), 2);
        im = __shfl_down_sync(0xffffffff, val.imag(), 2);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, val.real(), 1);
        im = __shfl_down_sync(0xffffffff, val.imag(), 1);
        val = op.Reduce(val, {re, im});    
      }
      else if (size > 4) {
        re = __shfl_down_sync(0xffffffff, val.real(), 4);
        im = __shfl_down_sync(0xffffffff, val.imag(), 4);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, val.real(), 2);
        im = __shfl_down_sync(0xffffffff, val.imag(), 2);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, val.real(), 1);
        im = __shfl_down_sync(0xffffffff, val.imag(), 1);
        val = op.Reduce(val, {re, im});   
      }
      else if (size > 2) {
        re = __shfl_down_sync(0xffffffff, val.real(), 2);
        im = __shfl_down_sync(0xffffffff, val.imag(), 2);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, val.real(), 1);
        im = __shfl_down_sync(0xffffffff, val.imag(), 1);
        val = op.Reduce(val, {re, im}); 
      }
      else if (size > 1) {
        re = __shfl_down_sync(0xffffffff, val.real(), 1);
        im = __shfl_down_sync(0xffffffff, val.imag(), 1);
        val = op.Reduce(val, {re, im}); 
      }
      return val;
    }
    else {
      if (size > 16) {
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 16);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 16);
        val = op.Reduce(val, {re, im});
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 8);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 8);
        val = op.Reduce(val, {re, im});        
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 4);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 4);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 2);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 2);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 1);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 1);
        val = op.Reduce(val, {re, im});                                       
      }
      else if (size > 8) {
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 8);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 8);
        val = op.Reduce(val, {re, im});        
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 4);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 4);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 2);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 2);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 1);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 1);
        val = op.Reduce(val, {re, im});    
      }
      else if (size > 4) {
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 4);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 4);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 2);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 2);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 1);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 1);
        val = op.Reduce(val, {re, im});   
      }
      else if (size > 2) {
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 2);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 2);
        val = op.Reduce(val, {re, im});  
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 1);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 1);
        val = op.Reduce(val, {re, im}); 
      }
      else if (size > 1) {
        re = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.real()), 1);
        im = __shfl_down_sync(0xffffffff, static_cast<typename T::value_type::value_type>(val.imag()), 1);
        val = op.Reduce(val, {re, im}); 
      }
      return val;      
    }
  }
  else {  
    // breaking this out so common case is faster without branches
    if constexpr (!is_matx_half_v<T>) {
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
    else {
      if (size > 16) {
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 16));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 8));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 4));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 2));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 1));
      }
      else if (size > 8) {
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 8));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 4));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 2));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 1));
      }
      else if (size > 4) {
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 4));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 2));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 1));
      }
      else if (size > 2) {
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 2));
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 1));
      }
      else if (size > 1) {
        val = op.Reduce(val, __shfl_down_sync(0xffffffff, static_cast<typename T::value_type>(val), 1));
      }
      return val;
    }
  }
}


template <typename OutType, typename InType, typename ReduceOp>
__launch_bounds__(1024, 1)
__global__ void matxReduceKernel(OutType dest, const InType in,
                                 ReduceOp red, [[maybe_unused]] index_t mult)
{
  constexpr uint32_t RANK = OutType::Rank();
  constexpr uint32_t DRANK = InType::Rank() - RANK;
  std::array<index_t, InType::Rank()> indices;
  using scalar_type = typename InType::scalar_type;
  using T = typename OutType::scalar_type;
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

template <typename OutType, typename TensorIndexType, typename InType>
__global__ void matxIndexKernel(OutType dest, TensorIndexType idest, InType in, [[maybe_unused]] index_t mult)
{
  using index_type = typename TensorIndexType::scalar_type;
  using T = typename OutType::scalar_type;
  T in_val;
  constexpr uint32_t RANK = TensorIndexType::Rank();
  constexpr uint32_t DRANK = InType::Rank() - RANK;  
  std::array<index_t, InType::Rank()> indices;
  index_t abs_idx = -1;
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

  if (abs_idx != -1) {
    // compute output offsets
    if constexpr (RANK == 0) {
      out  = &dest();
      iout = &idest();
      valid = true;
    }
    else if constexpr (RANK == 1) {
        out = &dest(indices[InType::Rank()-1]);
        iout = &idest(indices[InType::Rank()-1]);
        valid = true;
    }
    else if constexpr (RANK == 2) {
      out = &dest(indices[InType::Rank()-2], indices[InType::Rank()-1]);
      iout = &idest(indices[InType::Rank()-2], indices[InType::Rank()-1]);
      valid = true;
    }
    else if constexpr (RANK == 3) {
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
 * @tparam OutType
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
template <typename OutType, typename TensorIndexType, typename InType, typename ReduceOp,
  std::enable_if_t<is_matx_reduction_v<ReduceOp>, bool> = true>
void __MATX_INLINE__ reduce(OutType dest, [[maybe_unused]] TensorIndexType idest, const InType &in, ReduceOp op,
                   cudaStream_t stream = 0, bool init = true)
{
#ifdef __CUDACC__  
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  using scalar_type = typename InType::scalar_type;
  using T = typename OutType::scalar_type;

  static_assert(OutType::Rank() < InType::Rank());
  static_assert(is_matx_reduction_v<ReduceOp>,  "Must use a reduction operator for reducing");    

  if constexpr (OutType::Rank() > 0) {
    for (int i = 0; i < OutType::Rank(); i++) {
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
 * @tparam OutType
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
template <typename OutType, typename InType, typename ReduceOp>
void __MATX_INLINE__ reduce(OutType dest, const InType &in, ReduceOp op,
                   cudaStream_t stream = 0, [[maybe_unused]] bool init = true)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  constexpr bool use_cub = OutType::Rank() == 0 || (OutType::Rank() == 1 && InType::Rank() == 2);
  // Use CUB implementation if we have a tensor on the RHS and it's not blocked from using CUB
  if constexpr (!is_matx_no_cub_reduction_v<ReduceOp> && use_cub) {
    cub_reduce<OutType, InType, ReduceOp>(dest, in, op.Init(), stream);
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
 * @param exec
 *   CUDA executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ mean(OutType dest, const InType &in,
                 cudaExecutor exec = 0)
{
#ifdef __CUDACC__  
  MATX_NVTX_START("mean(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  static_assert(OutType::Rank() < InType::Rank(), "reduction dimensions must be <= Rank of input");
  
  using inner_type = typename inner_op_type_t<typename InType::scalar_type>::type;
  inner_type scale = 1;

  cudaStream_t stream = exec.getStream();

  sum(dest, in, stream);

  // The reduction is performed over the difference in ranks between input and
  // output. This loop computes the number of elements it was performed over.
  for (int i = 1; i <= InType::Rank() - OutType::Rank(); i++) {
    scale *= static_cast<inner_type>(in.Size(InType::Rank() - i));
  }

  (dest = dest * static_cast<inner_type>(1) / scale).run(stream);
#endif  
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
 * @param exec
 *   Single thread host executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ mean(OutType dest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("mean(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(OutType::Rank() < InType::Rank(), "reduction dimensions must be <= Rank of input");
  using inner_type = typename inner_op_type_t<typename InType::scalar_type>::type;
  inner_type scale = 1;

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      auto ts = TotalSize(in);
      *lout = std::reduce(lin, lin + ts) / static_cast<inner_type>(ts); 
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        *(lout + b) = std::reduce(lin + lbegin[b], lin + lend[b]) / static_cast<inner_type>(lin.Size(1)); 
      }
    }
  };
  
  ReduceInput(ft, dest, in);
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
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Rank of dimension array
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Dimensions to compute the mean over
 * @param exec
 *   Executor type
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ mean(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("mean(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  typename detail::exec_type_t<Executor> etype{exec};

  mean(dest, permute(in, perm), etype);
#endif  
}


/**
 * Calculate the softmax of values in a tensor treated as a flat vector
 *
 * softmax computes the exponential of each value divided by the sum of the exponentials
 * of items in the reduced set. By default the sum is performed over all dimensions. Note
 * that traditional definitions of softmax are simply exp(x)/sum(exp(x)), but this is not
 * how most libraries are implemented. Instead, x is biased by a correction factor of max(x).
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam RANK
 *   Rank of output tensor
 *
 * @param dest
 *   Destination for softmax output
 * @param in
 *   Input data to compute the softmax 
 * @param stream
 *   CUDA stream
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ softmax(OutType dest, const InType &in,
                 cudaStream_t stream = 0)
{
#ifdef __CUDACC__  
  MATX_NVTX_START("softmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto tmp_sum = make_tensor<typename InType::scalar_type>(MATX_ASYNC_DEVICE_MEMORY, stream);
  auto tmp_max = make_tensor<typename InType::scalar_type>(MATX_ASYNC_DEVICE_MEMORY, stream);
  rmax(tmp_max, in);
  sum(tmp_sum, exp(in - tmp_max), stream);
  (dest = exp(in - tmp_max) / tmp_sum).run(stream);
#endif  
}

/**
 * Calculate the softmax of values in a tensor treated as a flat vector
 *
 * softmax computes the exponential of each value divided by the sum of the exponentials
 * of items in the reduced set. The axes in which to perform the softmax over determine
 * which axes the sum will be computed over, but the input tensor rank and sizes match
 * between input and output. Note that traditional definitions of softmax are simply 
 * exp(x)/sum(exp(x)), but this is not how most libraries are implemented. Instead, x 
 * is biased by a correction factor of max(x).
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Rank of dimension array
 *
 * @param dest
 *   Destination for softmax output
 * @param in
 *   Input data to compute the softmax 
 * @param dims
 *   C-style array containing the dimensions to sum over
 * @param stream
 *   CUDA stream
 */
template <typename OutType, typename InType, int D>
void __MATX_INLINE__ softmax(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("softmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "softmax dimensions must be <= Rank of input");
  static_assert(OutType::Rank() == InType::Rank(), "softmax output rank must equal input rank");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);

  // Create the shape of the summed tensor based on the permutation params
  std::array<index_t, in.Rank() - D> red_shape{};
  #pragma unroll
  for (int r = 0; r < in.Rank() - D; r++) {
    red_shape[r] = in.Size(perm[r]);
  }

  // With the sum calculated, we have a tensor that's not compatible in sizes with the new one for dividing.
  // We need to clone the summed tensor on the appropriate dims for the final divide.
  std::array<index_t, InType::Rank()> clone_dims;
  int axis_ptr = 0;
  #pragma unroll
  for (int r = 0; r < InType::Rank(); r++) {
    if (axis_ptr >= 0 && dims[axis_ptr] == r) {
      clone_dims[r] = in.Size(r);
      if (++axis_ptr == D) {
        axis_ptr = -1;
      }
    }
    else {
      clone_dims[r] = matxKeepDim;
    }
  }  

  auto tmp_sum = make_tensor<typename InType::scalar_type>(red_shape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto tmp_max = make_tensor<typename InType::scalar_type>(red_shape, MATX_ASYNC_DEVICE_MEMORY, stream);
  rmax(tmp_max, permute(in, perm), stream);
  sum(tmp_sum, exp(permute(in, perm) - clone<InType::Rank()>(tmp_max, clone_dims)), stream);

  (dest = exp(in - clone<InType::Rank()>(tmp_max, clone_dims)) / clone<InType::Rank()>(tmp_sum, clone_dims)).run(stream);
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
 * @param exec
 *   CUDA executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ median(OutType dest,
                   const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__  
  if constexpr ( OutType::Rank() <= 1 && InType::Rank() <=2 ) {
    MATX_NVTX_START("median(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
    using T = typename OutType::scalar_type;
    constexpr int RANK_IN = InType::Rank();
    static_assert(RANK_IN <= 2 && (RANK_IN == OutType::Rank() + 1));

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    cudaStream_t stream = exec.getStream();

    auto tmp_sort = make_tensor<T>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);

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
    else if constexpr (RANK_IN == 2) {
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
  } else {

#if 1  // sort doesn't currently work on non-tensor input
    static_assert(InType::Rank() <= 2 && OutType::Rank() <= 1, "median only supported with output rank <= 1 and input rank <= 2");
#else 
    constexpr int out_dims = OutType::Rank();
    constexpr int red_dims = InType::Rank() - OutType::Rank();

    if constexpr ( out_dims > 1) {  
      // collapse batch dimensions to a single dimension
      auto oop = lcollapse<out_dims>(dest);
      auto iop = lcollapse<out_dims>(in);

      static_assert(oop.Rank() == 1);
      median(oop, iop, stream);

    } else if constexpr ( red_dims > 1) { 

      // collapse reduction dim to a single dim
      auto iop = rcollapse<red_dims>(in);

      static_assert(dest.Rank() <= 1);
      static_assert(iop.Rank() <= 2);
      median(dest, iop, stream);
    } else {
      static_assert(false, "median ranks not supported");
    }
#endif
  }
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
 * @param exec
 *   Single thread host executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ median(OutType dest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("median(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      auto insize = TotalSize(in);
      auto tin = new typename InType::scalar_type[insize];      
      std::partial_sort_copy( lin, 
                              lin + insize, 
                              tin, 
                              tin + insize);
      if ((insize % 2) == 0) {
        *lout = (tin[insize / 2] + tin[insize / 2 - 1]) / 2.0f;
      }
      else {
        *lout = tin[insize / 2];
      }

      delete [] tin;          
    }
    else {
      auto insize = lin.Size(1);
      auto tin = new typename InType::scalar_type[insize];      
      for (index_t b = 0; b < lin.Size(0); b++) {
        std::partial_sort_copy( lin + lbegin[b], 
                                lin + lend[b], 
                                tin, 
                                tin + insize);

        if ((insize % 2) == 0) {        
          *(lout + b) = (tin[insize / 2] + tin[insize / 2 - 1]) / 2.0f;
        }
        else {
          *(lout + b) = tin[insize / 2];
        }     
      }

      delete [] tin;            
    }
  };
  
  ReduceInput(ft, dest, in);  
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
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Rank of dimension array
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Dimensions to compute the mean over
 * @param exec
 *   Single thread host executor
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ median(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("median(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);

  typename detail::exec_type_t<Executor> etype{exec};

  median(dest, permute(in, perm), etype);
#endif  
}

/**
 * Compute sum of numbers
 *
 * Returns a tensor representing the sum of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ sum(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("sum(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  cub_sum<OutType, InType>(dest, in, stream);
#endif  
}

/**
 * Compute sum of numbers
 *
 * Returns a tensor representing the sum of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single thread host executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ sum(OutType dest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("sum(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      // auto b = std::accumulate(lin, lin + lin.Size(0), static_cast<typename InType::scalar_type>(0)); 
      // std::string aa = b;
      // std::string aad = *lout;
      // std::string aagd = lout;
      *lout = std::accumulate(lin, lin + lin.Size(0), static_cast<typename InType::scalar_type>(0)); 
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        *(lout + b) = std::accumulate(lin + lbegin[b], lin + lend[b], static_cast<typename InType::scalar_type>(0)); 
      }
    }
  };
  
  ReduceInputNoConvert(ft, dest, in);  
}

/**
 * Compute sum of numbers
 *
 * Returns a tensor representing the sum of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Rank of dimension array
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Dimensions to compute the mean over
 * @param exec
 *   Executor type
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ sum(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
  MATX_NVTX_START("sum(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);

  typename detail::exec_type_t<Executor> etype{exec};

  sum(dest, permute(in, perm), etype);
}




/**
 * Compute product of numbers
 *
 * Returns a tensor representing the product of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ prod(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("prod(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  reduce(dest, in, detail::reduceOpProd<typename OutType::scalar_type>(), stream, true);
#endif  
}

/**
 * Compute product of numbers
 *
 * Returns a tensor representing the product of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single thread host executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ prod(OutType dest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("prod(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      *lout = std::accumulate(lin, 
                              lin + TotalSize(in), 
                              static_cast<typename InType::scalar_type>(1), 
                              std::multiplies<typename InType::scalar_type>()); 
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        *(lout + b) = std::accumulate(lin + lbegin[b], 
                                      lin + lend[b], 
                                      static_cast<typename InType::scalar_type>(1), 
                                      std::multiplies<typename InType::scalar_type>()); 
      }
    }
  };
  
  ReduceInput(ft, dest, in);  
}

/**
 * Compute product of numbers
 *
 * Returns a tensor representing the product of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Rank of dimension array
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Dimensions to compute the mean over
 * @param exec
 *   Executor for reduction
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ prod(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("prod(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  typename detail::exec_type_t<Executor> etype{exec};

  prod(dest, permute(in, perm), etype);
#endif  
}

/**
 * Compute max reduction of a tensor
 *
 * Returns a tensor representing the max of all numbers in the reduction
 *
 * @note This function uses the name rmax instead of max to not collide with the
 * element-wise operator max.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ rmax(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("rmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  cub_max<OutType, InType>(dest, in, stream);
#endif  
}

/**
 * Compute max reduction of a tensor
 *
 * Returns a tensor representing the max of all numbers in the reduction
 *
 * @note This function uses the name rmax instead of max to not collide with the
 * element-wise operator max.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ rmax(OutType dest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("rmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      *lout = *std::max_element(lin, lin + TotalSize(in)); 
    }
    else {
      auto els = lend[1] - lbegin[0];
      for (index_t b = 0; b < els; b++) {
        lout[b] = *std::max_element(lin + lbegin[b], lin + lend[b]); 
      }
    }
  };
  
  ReduceInput(ft, dest, in);    
}

/**
 * Compute max reduction of a tensor
 *
 * Returns a tensor representing the max of all numbers in the reduction
 *
 * @note This function uses the name rmax instead of max to not collide with the
 * element-wise operator max.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @param exec
 *   Executor to use for reduction
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ rmax(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("rmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  typename detail::exec_type_t<Executor> etype{exec};

  rmax(dest, permute(in, perm), etype);
#endif  
}

/**
 * Compute maxn reduction of a tensor and returns value + index
 *
 * Returns a tensor with maximums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename TensorIndexType, typename InType>
void __MATX_INLINE__ argmax(OutType dest, TensorIndexType &idest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__  
  MATX_NVTX_START("argmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  reduce(dest, idest, in, detail::reduceOpMax<typename OutType::scalar_type>(), stream, true);
#endif  
}

/**
 * Compute maxn reduction of a tensor and returns value + index
 *
 * Returns a tensor with maximums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename TensorIndexType, typename InType>
void __MATX_INLINE__ argmax(OutType dest, TensorIndexType &idest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("argmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      *lout = std::max_element(lin, lin + TotalSize(in)) - lin; 
    }
    else {
      auto els = lend[0] - lbegin[0];
      for (index_t b = 0; b < els; b++) {
        lout[b] = std::max_element(lin + lbegin[b], lin + lend[b]) - lin; 
      }
    }
  };
  
  // This could be more efficient by not running two reductions to find the same values, but
  // for brevity this is faster
  ReduceInput(ft, idest, in);
  rmax(dest, in, exec);
}

/**
 * Compute maxn reduction of a tensor and returns value + index
 *
 * Returns a tensor with maximums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @param exec
 *   Executor to use for reduction
 */
template <typename OutType, typename TensorIndexType, typename InType, int D, typename Executor>
void __MATX_INLINE__ argmax(OutType dest, const TensorIndexType &idest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("argmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  typename detail::exec_type_t<Executor> etype{exec};

  argmax(dest, idest, permute(in, perm), etype);
#endif  
}


/**
 * Compute min reduction of a tensor
 *
 * Returns a tensor representing the min of all numbers in the reduction
 *
 * @note This function uses the name rmin instead of min to not collide with the
 * element-wise operator min.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ rmin(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__  
  MATX_NVTX_START("rmin(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  cub_min<OutType, InType>(dest, in, stream);
#endif  
}

/**
 * Compute min reduction of a tensor
 *
 * Returns a tensor representing the min of all numbers in the reduction
 *
 * @note This function uses the name rmin instead of min to not collide with the
 * element-wise operator min.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ rmin(OutType dest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{ 
  MATX_NVTX_START("rmin(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      *lout = *std::min_element(lin, lin + TotalSize(in)); 
    }
    else {
      auto els = lend[1] - lbegin[0];
      for (index_t b = 0; b < els; b++) {
        lout[b] = *std::min_element(lin + lbegin[b], lin + lend[b]); 
      }
    }
  };
  
  ReduceInput(ft, dest, in);   
}

/**
 * Compute min reduction of a tensor
 *
 * Returns a tensor representing the min of all numbers in the reduction
 *
 * @note This function uses the name rmin instead of min to not collide with the
 * element-wise operator min.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @param exec
 *   Executor to use for reduction
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ rmin(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("rmin(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  typename detail::exec_type_t<Executor> etype{exec};

  rmin(dest, permute(in, perm), etype);

#endif  
}

/**
 * Compute min reduction of a tensor and returns value + index
 *
 * Returns a tensor with minimums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename TensorIndexType, typename InType>
void __MATX_INLINE__ argmin(OutType dest, TensorIndexType &idest, const InType &in, cudaExecutor exec = 0)
{
  static_assert(OutType::Rank() == TensorIndexType::Rank());
#ifdef __CUDACC__  
  MATX_NVTX_START("argmin(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  reduce(dest, idest, in, detail::reduceOpMin<typename OutType::scalar_type>(), stream, true);
#endif  
}

/**
 * Compute min reduction of a tensor and returns value + index
 *
 * Returns a tensor with minimums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   SIngle host executor
 */
template <typename OutType, typename TensorIndexType, typename InType>
void __MATX_INLINE__ argmin(OutType dest, TensorIndexType &idest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("argmin(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      *lout = std::min_element(lin, lin + TotalSize(in)) - lin; 
    }
    else {
      auto els = lend[1] - lbegin[0];
      for (index_t b = 0; b < els; b++) {
        lout[b] = std::min_element(lin + lbegin[b], lin + lend[b]) - lin; 
      }
    }
  };
  
  // This could be more efficient by not running two reductions to find the same values, but
  // for brevity this is faster
  ReduceInput(ft, idest, in);
  rmin(dest, in, exec);
}

/**
 * Compute min reduction of a tensor and returns value + index
 *
 * Returns a tensor with minimums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @param exec
 *   Executor to use for reduction
 */
template <typename OutType, typename TensorIndexType, typename InType, int D, typename Executor>
void __MATX_INLINE__ argmin(OutType dest, TensorIndexType &idest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("argmin(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  typename detail::exec_type_t<Executor> etype{exec};

  argmin(dest, idest, permute(in, perm), etype);
#endif  
}

/**
 * Find if any value is != 0
 *
 * Returns a boolean value indicating whether any value in the set of inputs are
 * non-zero. The same aggregation rules apply for input vs output tensor size
 * and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ any(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__  
  MATX_NVTX_START("any(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  cudaStream_t stream = exec.getStream();
  reduce(dest, in, detail::reduceOpAny<typename OutType::scalar_type>(), stream, true);
#endif  
}

/**
 * Find if any value is != 0
 *
 * Returns a boolean value indicating whether any value in the set of inputs are
 * non-zero. The same aggregation rules apply for input vs output tensor size
 * and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ any(OutType dest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("any(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      *lout = std::any_of(lin, lin + TotalSize(in), [](typename InType::scalar_type vin) {
          return vin != 0;
        }); 
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        lout[b] = std::any_of(lin + lbegin[b], lin + lend[b], [](typename InType::scalar_type vin) {
          return vin != 0;
        }); 
      }
    }
  };
  
  ReduceInput(ft, dest, in);
}

/**
 * Find if any value is != 0
 *
 * Returns a boolean value indicating whether any value in the set of inputs are
 * non-zero. The same aggregation rules apply for input vs output tensor size
 * and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @param exec
 *   Executor to use for reduction
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ any(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("any(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);

  typename detail::exec_type_t<Executor> etype{exec};
  any(dest, permute(in, perm), etype);
#endif  
}

/**
 * Find if all values are != 0
 *
 * Returns a boolean value indicating whether all values in the set of inputs
 * are non-zero. The same aggregation rules apply for input vs output tensor
 * size and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ all(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__ 
  MATX_NVTX_START("all(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  cudaStream_t stream = exec.getStream();
  reduce(dest, in, detail::reduceOpAll<typename OutType::scalar_type>(), stream, true);
#endif  
}

/**
 * Find if all values are != 0
 *
 * Returns a boolean value indicating whether all values in the set of inputs
 * are non-zero. The same aggregation rules apply for input vs output tensor
 * size and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ all(OutType dest, const InType &in, [[maybe_unused]] SingleThreadHostExecutor exec)
{
  MATX_NVTX_START("all(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) { 
    if constexpr (OutType::Rank() == 0) {
      *lout = std::all_of(lin, lin + TotalSize(in), [](typename InType::scalar_type vin) {
          return vin != 0;
        }); 
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        lout[b] = std::all_of(lin + lbegin[b], lin + lend[b], [](typename InType::scalar_type vin) {
          return vin != 0;
        }); 
      }
    }
  };
  
  ReduceInput(ft, dest, in);
}

/**
 * Find if all values are != 0
 *
 * Returns a boolean value indicating whether all values in the set of inputs
 * are non-zero. The same aggregation rules apply for input vs output tensor
 * size and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @param exec
 *   Executor to use for reduction
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ all(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("all(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  typename detail::exec_type_t<Executor> etype{exec};

  all(dest, permute(in, perm), etype);
#endif  
}

/**
 * Compute a variance reduction
 *
 * Computes the variance of the input according to the output tensor rank and
 * size
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ var(OutType dest, const InType &in, int stream = 0)
{
  var(dest, in, cudaExecutor{stream});
}

/**
 * Compute a variance reduction
 *
 * Computes the variance of the input according to the output tensor rank and
 * size
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Executor type
 */
template <typename OutType, typename InType, typename Executor, std::enable_if_t<is_executor_t<Executor>(), bool> = true>
void __MATX_INLINE__ var(OutType dest, const InType &in, Executor &&exec)
{
  MATX_NVTX_START("var(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  matxMemorySpace_t space;
  using inner_type = typename inner_op_type_t<typename InType::scalar_type>::type;

  if constexpr (is_device_executor_v<Executor>) {
    space = MATX_ASYNC_DEVICE_MEMORY;
  }
  else {
    space = MATX_HOST_MALLOC_MEMORY;
  }

  auto mean_tns = make_tensor<typename InType::scalar_type>(dest.Descriptor(), space);
  mean(mean_tns, in, exec);

  // need to clone along right most dims
  std::array<index_t, InType::Rank()> cdims;
  for(int i = 0; i < OutType::Rank(); i++) {
    cdims[i] = matxKeepDim;
  }
  for(int i = OutType::Rank(); i < InType::Rank(); i++) {
    cdims[i] = in.Size(i);
  }

  auto mean_op = mean_tns.template Clone<InType::Rank()>(cdims);
  sum(dest, pow(abs(in - mean_op), static_cast<inner_type>(2)), exec);

  // The length of what we are taking the variance over is equal to the product
  // of the outer dimensions covering the different in input/output ranks
  index_t N = in.Size(in.Rank() - 1);
  for (int i = 2; i <= in.Rank() - OutType::Rank(); i++) {
    N *= in.Size(in.Rank() - i);
  }

  // Sample variance for an unbiased estimate
  (dest = dest / static_cast<inner_type>(N - 1)).run(exec);
}

/**
 * Compute a variance reduction
 *
 * Computes the variance of the input according to the output tensor rank and
 * size
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @param exec
 *   Executor to use for reduction
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ var(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("var(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  //typename detail::exec_type_t<Executor> etype{exec};

  var(dest, permute(in, perm), std::forward<Executor>(exec));
#endif  
}

/**
 * Compute a standard deviation reduction
 *
 * Computes the standard deviation of the input according to the output tensor
 * rank and size
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Executor type
 */
template <typename OutType, typename InType, typename Executor, std::enable_if_t<is_executor_t<Executor>(), bool> = true>
void __MATX_INLINE__ stdd(OutType dest, const InType &in, Executor &&exec)
{
  MATX_NVTX_START("stdd(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  var(dest, in, exec);
  (dest = sqrt(dest)).run(exec);
}

/**
 * Compute a standard deviation reduction
 *
 * Computes the standard deviation of the input according to the output tensor
 * rank and size
 *
 * @tparam OutType
 *   Output data type
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
template <typename OutType, typename InType>
void __MATX_INLINE__ stdd(OutType dest, const InType &in, int stream = 0)
{
  stdd(dest, in, cudaExecutor{stream});
}

/**
 * Compute a standard deviation reduction
 *
 * Computes the standard deviation of the input according to the output tensor
 * rank and size
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @param exec
 *   Executor to use for reduction
 */
template <typename OutType, typename InType, int D, typename Executor>
void __MATX_INLINE__ stdd(OutType dest, const InType &in, const int (&dims)[D], Executor &&exec)
{
#ifdef __CUDACC__
  MATX_NVTX_START("stdd(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  static_assert(OutType::Rank() + D == InType::Rank(), "reduction output rank must equal input rank minus reduction dims");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);

  stdd(dest, permute(in, perm), std::forward<Executor>(exec));

#endif  
}

/**
 * Computes the trace of a tensor
 *
 * Computes the trace of a square matrix by summing the diagonal
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Executor type
 */
template <typename OutType, typename InType, typename Executor, std::enable_if_t<is_executor_t<Executor>(), bool> = true>
void __MATX_INLINE__ trace(OutType dest, const InType &in, Executor &&exec)
{
  MATX_NVTX_START("trace(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto d = diag(in);
  sum(dest, d, exec);
}

/**
 * Computes the trace of a tensor
 *
 * Computes the trace of a square matrix by summing the diagonal
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ trace(OutType dest, const InType &in, int stream)
{
  return trace(dest, in, cudaExecutor{stream});
}

} // end namespace matx
