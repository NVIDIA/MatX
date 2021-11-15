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

#include <cinttypes>
#include <cstdint>
#include <cuda/std/atomic>
#include <iomanip>
#include <numeric>
#include <type_traits>

#include "matx_allocator.h"
#include "matx_error.h"
#include "matx_shape.h"
#include "matx_type_utils.h"
#include "matx_utility_kernels.cuh"
#include "matx_tensor_utils.h"

static constexpr int MAX_TENSOR_DIM = 4;
static constexpr bool PRINT_ON_DEVICE = false;

// forward declare
namespace matx {
template <typename T, int RANK> class tensor_t;

template <typename T> class BaseOp;
} // namespace matx

/* Special values used to indicate properties of tensors */
namespace matx {
#ifdef INDEX_64_BIT
enum {
  matxKeepDim = LLONG_MAX,
  matxDropDim = LLONG_MAX - 1,
  matxEnd = LLONG_MAX - 2,
  matxKeepStride = LLONG_MAX - 3
};
#else
enum {
  matxKeepDim = INT_MAX,
  matxDropDim = INT_MAX - 1,
  matxEnd = INT_MAX - 2,
  matxKeepStride = INT_MAX - 3
};
#endif

/**
 * Assignment from one operator/View into a View
 *
 * The set function is used in lieu of the assignment operator to avoid
 * ambiguity in certain scenarios. It can be used in the same scenarios
 * as the assignment operator.
 *
 * @tparam T
 *   Type of operator
 * @tparam RANK
 *   Rank of operator
 * @tparam Op
 *   Operator to use as input
 **/
template <class T, int RANK, class Op>
class set : public BaseOp<set<T, RANK, Op>> {
private:
  tensor_t<T, RANK> out_;
  Op op_;
  std::array<index_t, RANK> size_;

public:
  // Type specifier for reflection on class
  using scalar_type = void;

  /**
   * Constructor to assign an operator to a view
   *
   * @param out
   *   Output destination view
   *
   * @param op
   *   Input operator
   */
  inline set(tensor_t<T, RANK> &out, const Op op) : out_(out), op_(op)
  {
    MATX_STATIC_ASSERT(get_rank<Op>() == -1 || Rank() == get_rank<Op>(),
                       matxInvalidDim);
    if constexpr (RANK > 0) {
      for (int i = 0; i < RANK; i++) {
        index_t size = get_expanded_size<Rank()>(op_, i);
        size_[i] = out_.Size(i);
        MATX_ASSERT_STR(
            size == 0 || size == Size(i), matxInvalidSize,
            "Size mismatch in source operator to destination tensor view");        
      }
    }
  }

  set &operator=(const set &) = delete;

  __MATX_DEVICE__ __MATX_HOST__ inline auto operator()() noexcept
  {
    if constexpr (is_matx_half_v<T> &&
                  std::is_integral_v<decltype(get_value(op_))>) {
      out_() = static_cast<float>(get_value(op_));
    }
    else {
      out_() = get_value(op_);
    }

    return out_();
  }

  __MATX_DEVICE__ __MATX_HOST__ inline auto operator()(index_t i) noexcept
  {
    if constexpr (is_matx_half_v<T> &&
                  std::is_integral_v<decltype(get_value(op_, i))>) {
      out_(i) = static_cast<float>(get_value(op_, i));
    }
    else {
      out_(i) = get_value(op_, i);
    }

    return out_(i);
  }

  __MATX_DEVICE__ __MATX_HOST__ inline auto operator()(index_t i, index_t j) noexcept
  {
    if constexpr (is_matx_half_v<T> &&
                  std::is_integral_v<decltype(get_value(op_, i, j))>) {
      out_(i, j) = static_cast<float>(get_value(op_, i, j));
    }
    else {
      out_(i, j) = get_value(op_, i, j);
    }

    return out_(i, j);
  }

  __MATX_DEVICE__ __MATX_HOST__ inline auto operator()(index_t i, index_t j, index_t k) noexcept
  {
    if constexpr (is_matx_half_v<T> &&
                  std::is_integral_v<decltype(get_value(op_, i, j, k))>) {
      out_(i, j, k) = static_cast<float>(get_value(op_, i, j, k));
    }
    else {
      out_(i, j, k) = get_value(op_, i, j, k);
    }

    return out_(i, j, k);
  }

  __MATX_DEVICE__ __MATX_HOST__ inline auto operator()(index_t i, index_t j, index_t k,
                                    index_t l) noexcept
  {
    if constexpr (is_matx_half_v<T> &&
                  std::is_integral_v<decltype(get_value(op_, i, j, k, l))>) {
      out_(i, j, k, l) = static_cast<float>(get_value(op_, i, j, k, l));
    }
    else {
      out_(i, j, k, l) = get_value(op_, i, j, k, l);
    }

    return out_(i, j, k, l);
  }

  /**
   * Get the rank of the operator
   *
   * @return
   *   Rank of the operator
   */
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }

  /**
   * Get the rank of the operator along a single dimension
   *
   * @param dim
   *   Dimension to retrieve size
   * @return
   *   Size of dimension
   */
  template <int M = RANK, std::enable_if_t<M >= 1, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ index_t Size(uint32_t dim) const
  {
    return size_[dim];
  }
};

/**
 * View of an underlying tensor data object
 *
 * The tensor_t class provides multiple ways to view the data inside of a
 * matxTensorData_t object. Views do not modify the underlying data; they simply
 * present a different way to look at the data. This includes where the data
 * begins and ends, the stride, the rank, etc. Views are very lightweight, and
 * any number of views can be generated from the same data object. Since views
 * represent different ways of looking at the same data, it is the
 * responsibility of the user to ensure that proper synchronization is done when
 * using multiple views on the same data. Failure to do so can result in race
 * conditions on the device or host.
 */
template <typename T, int RANK> class tensor_t {
public:
  // Type specifier for reflection on class
  using type = T; // TODO is this necessary
  using scalar_type = T;
  using tensor_view = bool;

  // Type specifier for signaling this is a matx operation
  using matxop = bool;

  // Delete default constructor for ranks higher than 0
  template <int M = RANK, std::enable_if_t<M >= 1, bool> = true>
  tensor_t() = delete;

  __MATX_HOST__ __MATX_DEVICE__ tensor_t<T, RANK>(tensor_t<T, RANK> const &rhs) noexcept
      : data_(rhs.data_), ldata_(rhs.ldata_), shape_(rhs.shape_), s_(rhs.s_),
        refcnt_(rhs.refcnt_)
  {
#ifndef __CUDA_ARCH__
    if (refcnt_ != nullptr) {
      (*refcnt_)++;
    }
#endif
  }

  __MATX_HOST__ __MATX_DEVICE__ tensor_t<T, RANK>(tensor_t<T, RANK> const &&rhs) noexcept
      : data_(rhs.data_), ldata_(rhs.ldata_), shape_(std::move(rhs.shape_)),
        s_(rhs.s_), refcnt_(rhs.refcnt_)
  {
  }

  /** Perform a shallow copy of a tensor view
   *
   * Alternative to operator= since it's used for lazu evaluation. This function
   * is used to perform a shallow copy of a tensor view where the data pointer
   * points to the same location as the right hand side's data. *
   *
   * @param rhs
   *   Tensor to copy from
   */
  __MATX_HOST__ __MATX_DEVICE__ void Shallow(const tensor_t<T, RANK> &rhs) noexcept
  {
    data_ = rhs.data_;
    ldata_ = rhs.ldata_;
    shape_ = rhs.shape_;
    s_ = rhs.s_;
    refcnt_ = rhs.refcnt_;
    if (refcnt_ != nullptr) {
      (*refcnt_)++;
    }
  }

  inline __MATX_HOST__ __MATX_DEVICE__ ~tensor_t()
  {
#ifndef __CUDA_ARCH__
    Free();
#endif
  }

  /**
   * Constructor for a rank-0 tensor (scalar).
   */
  template <int M = RANK, std::enable_if_t<M == 0, bool> = true> tensor_t()
  {
#ifndef __CUDA_ARCH__
    Allocate();
#endif
  }

  /**
   * Constructor for a rank-0 tensor (scalar).
   *
   * @param data
   *   Data pointer
   */
  template <int M = RANK, std::enable_if_t<M == 0, bool> = true>
  tensor_t(T *const data, cuda::std::atomic<uint32_t> *refcnt = nullptr) :
    data_(data), ldata_(data), refcnt_(refcnt)
  {
#ifndef __CUDA_ARCH__
    if (refcnt != nullptr) {
      (*refcnt)++;
    }
#endif    
  }

  /**
   * Constructor for a rank-1 and above tensor.
   *
   * @param shape
   *   Tensor shape
   */
  inline tensor_t(tensorShape_t<RANK> const &shape) : shape_(shape)
  {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(shape.Size(i) > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
    }

    if constexpr (RANK >= 1) {
      s_[RANK - 1] = 1;
    }

#pragma unroll
    for (int i = RANK - 2; i >= 0; i--) {
      s_[i] = s_[i + 1] * shape_.Size(i + 1);
    }

    Allocate();
  }

  /**
   * Constructor for a rank-1 and above tensor.
   *
   * @param shape
   *   Tensor shape
   * @param strides
   *   Tensor strides
   */
  inline tensor_t(tensorShape_t<RANK> const &shape,
                  const index_t (&strides)[RANK])
      : shape_(shape)
  {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(shape.Size(i) > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
    }

    Allocate();
    memcpy((void *)s_.data(), (void *)strides, s_.size() * sizeof(index_t));
  }

  /**
   * Constructor for a rank-1 and above tensor using a shape input
   *
   * @param shape
   *   Sizes for each dimension. Length of sizes must match RANK
   */
  inline tensor_t(const index_t (&shape)[RANK])
      : tensor_t(tensorShape_t<RANK>{static_cast<index_t const *>(shape)})
  {
  }

  /**
   * Constructor for a rank-1 and above tensor using a user pointer and shape
   * input
   *
   * @param data
   *   Base data pointer (allocated address)
   * @param ldata
   *   Offset data pointer (start of view)
   * @param shape
   *   Sizes for each dimension. Length of sizes must match RANK
   * @param refcnt
   *   Reference counter or nullptr if not tracked
   */
  inline tensor_t(T *const data, T *const ldata,
                  const tensorShape_t<RANK> &shape,
                  cuda::std::atomic<uint32_t> *refcnt)
      : data_(data), ldata_(ldata), shape_(shape), refcnt_(refcnt)
  {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(shape.Size(i) > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
    }

    if constexpr (RANK >= 1) {
      s_[RANK - 1] = 1;
    }

#pragma unroll
    for (int i = RANK - 2; i >= 0; i--) {
      s_[i] = s_[i + 1] * shape_.Size(i + 1);
    }

#ifndef __CUDA_ARCH__
    if (refcnt != nullptr) {
      (*refcnt)++;
    }
#endif
  }

  /**
   * Constructor for creating a view with a user-defined data pointer.
   *
   * If not reference counted, it is the caller's responsibility to manage the
   * data pointer, including allocation and freeing.
   *
   * @param data
   *   Pointer to data
   * @param shape
   *   Tensor shape
   */
  inline tensor_t(T *const data, const tensorShape_t<RANK> &shape)
      : tensor_t(data, data, shape, nullptr)
  {
  }

  /**
   * Constructor for creating a view with a user-defined data pointer.
   *
   * If not reference counted, it is the caller's responsibility to manage the
   * data pointer, including allocation and freeing.
   *
   * @param data
   *   Pointer to data
   *
   * @param shape
   *   Sizes for each dimension. Length of sizes must match RANK
   */
  inline tensor_t(T *const data, const index_t (&shape)[RANK]) noexcept
      : tensor_t(data, data,
                 tensorShape_t<RANK>{static_cast<index_t const *>(shape)},
                 nullptr)
  {
  }

  /**
   * Constructor for creating a view with a user-defined data pointer.
   *
   * If not reference counted, it is the caller's responsibility to manage the
   * data pointer, including allocation and freeing.
   *
   * @param data
   *   Base data pointer (allocated address)
   * @param ldata
   *   Offset data pointer (start of view)
   * @param shape
   *   Sizes for each dimension. Length of sizes must match RANK
   * @param strides
   *   Tensor strides
   * @param refcnt
   *   Reference counter or nullptr if not tracked
   */
  inline tensor_t(T *const data, T *const ldata,
                  tensorShape_t<RANK> const &shape,
                  const index_t (&strides)[RANK],
                  cuda::std::atomic<uint32_t> *refcnt = nullptr)
      : data_(data), ldata_(ldata), shape_(shape), refcnt_(refcnt)
  {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(shape.Size(i) > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
    }

    memcpy((void *)s_.data(), (void *)strides, s_.size() * sizeof(index_t));

#ifndef __CUDA_ARCH__
    if (refcnt != nullptr) {
      (*refcnt)++;
    }
#endif
  }

  /**
   * Constructor for creating a view with a user-defined data pointer.
   *
   * If not reference counted, it is the caller's responsibility to manage the
   * data pointer, including allocation and freeing.
   *
   * @param data
   *   Base data pointer (allocated address)
   * @param shape
   *   Sizes for each dimension. Length of sizes must match RANK
   * @param strides
   *   Tensor strides
   */
  inline tensor_t(T *const data, tensorShape_t<RANK> const &shape,
                  const index_t (&strides)[RANK])
      : tensor_t(data, data, shape, strides, nullptr)
  {
  }

  /**
   * Lazy assignment operator=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator=(const tensor_t<T, RANK> &op)
  {
    return set(*this, op);
  }
  /**
   * Lazy assignment operator=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator=(const T2 &op)
  {
    return set(*this, op);
  }

  /**
   * Lazy assignment operator+=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator+=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this + op);
  }

  /**
   * Lazy assignment operator+=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator+=(const T2 &op)
  {
    return set(*this, *this + op);
  }

  /**
   * Lazy assignment operator-=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator-=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this - op);
  }

  /**
   * Lazy assignment operator-=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator-=(const T2 &op)
  {
    return set(*this, *this - op);
  }

  /**
   * Lazy assignment operator*=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator*=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this * op);
  }

  /**
   * Lazy assignment operator*=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator*=(const T2 &op)
  {
    return set(*this, *this * op);
  }

  /**
   * Lazy assignment operator/=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator/=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this / op);
  }

  /**
   * Lazy assignment operator/=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator/=(const T2 &op)
  {
    return set(*this, *this / op);
  }

  /**
   * Lazy assignment operator<<=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator<<=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this << op);
  }

  /**
   * Lazy assignment operator<<=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator<<=(const T2 &op)
  {
    return set(*this, *this << op);
  }

  /**
   * Lazy assignment operator>>=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator>>=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this >> op);
  }

  /**
   * Lazy assignment operator>>=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator>>=(const T2 &op)
  {
    return set(*this, *this >> op);
  }

  /**
   * Lazy assignment operator|=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator|=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this | op);
  }

  /**
   * Lazy assignment operator|=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator|=(const T2 &op)
  {
    return set(*this, *this | op);
  }

  /**
   * Lazy assignment operator&=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator&=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this & op);
  }

  /**
   * Lazy assignment operator&=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator&=(const T2 &op)
  {
    return set(*this, *this & op);
  }

  /**
   * Lazy assignment operator^=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator^=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this ^ op);
  }

  /**
   * Lazy assignment operator^=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator^=(const T2 &op)
  {
    return set(*this, *this ^ op);
  }

  /**
   * Lazy assignment operator%=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] inline __MATX_HOST__ auto operator%=(const tensor_t<T, RANK> &op)
  {
    return set(*this, *this % op);
  }

  /**
   * Lazy assignment operator%=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] inline __MATX_HOST__ auto operator%=(const T2 &op)
  {
    return set(*this, *this % op);
  }

  /**
   * Get the shape the tensor from the underlying data
   *
   * @return
   *    A shape of the data with the appropriate dimensions set
   */
  inline tensorShape_t<RANK> Shape() const noexcept { return shape_; }

  /**
   * Get a view of the tensor from the underlying data using a custom shape
   *
   * Returns a view based on the shape passed in. Both the rank and the
   * dimensions can be increased or decreased from the original data object as
   * long as they fit within the bounds of the memory allocation. This function
   * only allows a contiguous view of memory, regardless of the shape passed in.
   * For example, if the original shape is {8, 2} and a view of {2, 1} is
   * requested, the data in the new view would be the last two elements of the
   * last dimension of the original data.
   *
   * The function is similar to MATLAB and Python's reshape(), except it does
   * NOT make a copy of the data, whereas those languages may, depending on the
   * context. It is up to the user to understand any existing views on the
   * underlying data that may conflict with other views.
   *
   * While this function is similar to Slice(), it does not allow slicing a
   * particular start and end point as slicing does, and slicing also does not
   * allow increasing the rank of a tensor as View(shape) does.
   *
   * Note that the type of the data type of the tensor can also change from the
   * original data. This may be useful in situations where a union of data types
   * could be used in different ways. For example, a complex<float> could be
   * reshaped into a float tensor that has twice as many elements, and
   * operations can be done on floats instead of complex types.
   *
   * @tparam M
   *   New type of tensor
   * @tparam R
   *   New rank of tensor
   * @param shape
   *   New shape of tensor
   *
   * @return
   *    A view of the data with the appropriate strides and dimensions set
   */
  template <typename M = T, int R = RANK>
  inline tensor_t<M, R> View(tensorShape_t<R> const &shape) const
  {
    // Ensure new shape's total size is not larger than the original
    MATX_ASSERT_STR(
        sizeof(M) * shape.TotalSize() <= Bytes(), matxInvalidSize,
        "Total size of new tensor must not be larger than the original");

    // R == 0 will essentially be optimized out and unused in later checks
    index_t strides[R];

    if constexpr (R >= 1) {
      strides[R - 1] = 1;
    }

    for (int i = R - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape.Size(i + 1);
    }

    return tensor_t<M, R>(data_, data_, shape, strides, refcnt_);
  }

  template <typename M = T, int R = RANK>
  inline tensor_t<M, R> View(const index_t (&shape)[R]) const
  {
    return View(tensorShape_t<R>{(const index_t *)shape});
  }

  /**
   * Get a view of the tensor from the underlying data and default shape
   *
   * Returns a view with a shape based on the original shape used to create the
   * data object. The view returned will always occupy the entire underlying
   * data.
   *
   * @return
   *    A view of the data with the appropriate strides and dimensions set
   */
  inline tensor_t<T, RANK> View() const noexcept
  {
    // RANK == 0 will essentially be optimized out and unused in later checks
    index_t strides[RANK];

    if constexpr (RANK >= 1) {
      strides[RANK - 1] = 1;
    }

    for (int i = RANK - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * Size(i + 1);
    }

    return tensor_t<T, RANK>(data_, data_, shape_, strides, refcnt_);
  }

  /**
   * Prefetch the data asynchronously from the host to the device.
   *
   * All copies are done asynchronously in a stream. The order of the copy
   * is predictable within work in the same stream, but not when the transfer
   * will occur.
   *
   * @param stream
   *   The CUDA stream to prefetch within
   */
  inline void PrefetchDevice(cudaStream_t const stream) const noexcept
  {
    int dev;
    cudaGetDevice(&dev);
    cudaMemPrefetchAsync(data_, TotalSize() * sizeof(T), dev, stream);
  }

  /**
   * Prefetch the data asynchronously from the device to the host.
   *
   * All copies are done asynchronously in a stream. The order of the copy
   * is predictable within work in the same stream, but not when the transfer
   * will occur.
   *
   * @param stream
   *   The CUDA stream to prefetch within
   */
  inline void PrefetchHost(cudaStream_t const stream) const noexcept
  {
    cudaMemPrefetchAsync(data_, TotalSize() * sizeof(T), cudaCpuDeviceId,
                         stream);
  }

  /**
   * Create a view of only real-valued components of a complex array
   *
   * Only available on complex data types.
   *
   * @returns tensor view of only real-valued components
   *
   */
#ifdef DOXYGEN_ONLY
  tensor_t RealView() const noexcept
  {
#else
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = true>
  inline tensor_t<typename U::value_type, RANK> RealView() const noexcept
  {
#endif
    using Type = typename U::value_type;
    Type *data = reinterpret_cast<Type *>(data_);
    index_t strides[RANK];
#pragma unroll
    for (int i = 0; i < RANK; i++) {
      strides[i] = s_[i];
    }

    if constexpr (RANK > 0) {
#pragma unroll
      for (int i = 0; i < RANK; i++) {
        strides[i] *= 2;
      }
    }

    return tensor_t<Type, RANK>(reinterpret_cast<Type *>(data_), data, shape_,
                                strides, refcnt_);
  }

  /**
   * Create a view of only imaginary-valued components of a complex array
   *
   * Only available on complex data types.
   *
   * @returns tensor view of only imaginary-valued components
   *
   */
#ifdef DOXYGEN_ONLY
  tensor_t ImagView() const noexcept
  {
#else
  template <typename U = T, std::enable_if_t<is_complex_v<U>, bool> = true>
  inline tensor_t<typename U::value_type, RANK> ImagView() const noexcept
  {
#endif
    using Type = typename U::value_type;
    Type *data = reinterpret_cast<Type *>(data_) + 1;
    index_t strides[RANK];
#pragma unroll
    for (int i = 0; i < RANK; i++) {
      strides[i] = s_[i];
    }

    if constexpr (RANK > 0) {
#pragma unroll
      for (int i = 0; i < RANK; i++) {
        strides[i] *= 2;
      }
    }

    return tensor_t<Type, RANK>(reinterpret_cast<Type *>(data_), data, shape_,
                                strides, refcnt_);
  }

  /**
   * Permute the dimensions of a tensor
   *
   * Accepts any order of permutation. Number of dimensions must match RANK of
   * tensor
   *
   * @tparam M
   *   Rank of tensor to permute. Should not be used directly
   *
   * @param dims
   *   Dimensions of tensor
   *
   * @returns tensor view of only imaginary-valued components
   *
   */
#ifdef DOXYGEN_ONLY
  tensor_t Permute(const uint32_t (&dims)[RANK]) const
  {
#else
  template <int M = RANK, std::enable_if_t<M >= 2, bool> = true>
  inline tensor_t Permute(const uint32_t (&dims)[RANK]) const
  {
#endif
    index_t n[RANK];
    index_t s[RANK];
    [[maybe_unused]] bool done[RANK] = {0};

#pragma unroll
    for (int i = 0; i < RANK; i++) {
      int d = dims[i];
      MATX_ASSERT_STR(d < RANK, matxInvalidDim,
                      "Index to permute is larger than tensor rank");
      MATX_ASSERT_STR(done[d] == false, matxInvalidParameter,
                      "Cannot list the same dimension to permute twice");
      done[d] = true;
      n[i] = Size(d);
      s[i] = s_[d];
    }

    return tensor_t(data_, data_, n, s, refcnt_);
  }

  /**
   * Permute the last two dimensions of a matrix
   *
   * Utility function to permute the last two dimensions of a tensor. This is
   * useful in the numerous operations that take a permuted matrix as input, but
   * we don't want to permute the inner dimensions of a larger tensor.
   *
   * @tparam M
   *  Rank of tensor
   *
   * @param dims
   *  Dimensions of tensors
   *
   * @returns tensor view with last two dims permuted
   *
   */
#ifdef DOXYGEN_ONLY
  tensor_t PermuteMatrix() const
  {
#else
  template <int M = RANK>
  inline std::enable_if_t<M >= 2, tensor_t> PermuteMatrix() const
  {
#endif
    uint32_t tdims[RANK];
    std::iota(std::begin(tdims), std::end(tdims), 0);
    std::swap(tdims[RANK - 2], tdims[RANK - 1]);
    return Permute(tdims);
  }

  /**
   * Get the underlying local data pointer from the view
   *
   * @returns Underlying data pointer of type T
   *
   */
  __MATX_HOST__ __MATX_DEVICE__ inline T *Data() const noexcept { return ldata_; }

  /**
   * Set the underlying data pointer from the view
   *
   * Decrements any reference-counted memory and potentially frees before
   * resetting the data pointer. If refcnt is not nullptr, the count is
   * incremented.
   *
   * @param data
   *   Data pointer to set
   * @param refcnt
   *   Optional reference count for new memory or nullptr if not tracked
   *
   */
  __MATX_HOST__ inline void
  SetData(T *const data, cuda::std::atomic<uint32_t> *refcnt = nullptr) noexcept
  {
    SetData(data, data, refcnt);
  }

  /**
   * Set the underlying data and local data pointer from the view
   *
   * Decrements any reference-counted memory and potentially frees before
   * resetting the data pointer. If refcnt is not nullptr, the count is
   * incremented.
   *
   * @param data
   *   Allocated data pointer
   * @param ldata
   *   Local data pointer offset into allocated
   * @param refcnt
   *   Optional reference count for new memory or nullptr if not tracked
   *
   */
  __MATX_HOST__ inline void
  SetData(T *const data, T *const ldata,
          cuda::std::atomic<uint32_t> *refcnt = nullptr) noexcept
  {
    Free();

    data_ = data;
    ldata_ = ldata;

    refcnt_ = refcnt;
    if (refcnt_ != nullptr) {
      (*refcnt_)++;
    }
  }

  /**
   * Get the rank of the tensor
   *
   * @returns Rank of the tensor
   *
   */
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }

  /**
   * Get the reference count
   *
   * @returns Reference count or 0 if not tracked
   *
   */
  inline __MATX_HOST__ index_t GetRefCount() const noexcept
  {
    if (refcnt_ == nullptr) {
      return 0;
    }

    return refcnt_->load();
  }

  /**
   * Get the size of a single dimension of the tensor
   *
   * @param dim
   *   Desired dimension
   *
   * @returns Number of elements in dimension
   *
   */
  inline __MATX_HOST__ __MATX_DEVICE__ index_t Size(uint32_t dim) const noexcept
  {
    return shape_.Size(dim);
  }

  /**
   * Get the size of the last dimension
   *
   * @return
   *    The size of the dimension
   */
  inline __MATX_HOST__ __MATX_DEVICE__ index_t Lsize() const noexcept
  {
    return shape_.Size(Rank() - 1);
  }

  /**
   * Check if a tensor is linear in memory for all elements in the view
   *
   * @return
   *    The size of the dimension
   */
  inline __MATX_HOST__ __MATX_DEVICE__ bool IsLinear() const noexcept
  {
    index_t ttl = 1;
    for (int i = RANK - 1; i >= 0; i--) {
      if (s_[i] != ttl) {
        return false;
      }

      ttl *= shape_.Size(i);
    }

    return true;
  }

  /**
   * Get the stride of a single dimension of the tensor
   *
   * @param dim
   *   Desired dimension
   *
   * @returns Stride (in elements) in dimension
   *
   */
#ifdef DOXYGEN_ONLY
  index_t Stride(uint32_t const dim) const noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M >= 1, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ index_t Stride(uint32_t dim) const
  {
#endif
    return s_[dim];
  }

  /**
   * Get the total number of elements in the tensor
   *
   *
   * @returns Total number of elements across all dimensions
   *
   */
  inline index_t TotalSize() const noexcept { return shape_.TotalSize(); }

  /**
   * operator() getter with an array index
   *
   * @returns value in tensor
   *
   */
  template <int M = RANK, std::enable_if_t<M >= 1, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ const T &operator()(const std::array<index_t, RANK> &idx) const noexcept
  {
    if constexpr (RANK == 1) {
      return this->operator()(idx[0]);
    }
    else if constexpr (RANK == 2) {
      return this->operator()(idx[0], idx[1]);
    }
    else if constexpr (RANK == 3) {
      return this->operator()(idx[0], idx[1], idx[2]);
    }
    else if constexpr (RANK == 4) {
      return this->operator()(idx[0], idx[1], idx[2], idx[3]);
    }            
  }  

  /**
   * operator() setter with an array index
   *
   * @returns value in tensor
   *
   */
  template <int M = RANK, std::enable_if_t<M >= 1, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ T &operator()(const std::array<index_t, RANK> &idx) noexcept
  {
    if constexpr (RANK == 1) {
      return this->operator()(idx[0]);
    }
    else if constexpr (RANK == 2) {
      return this->operator()(idx[0], idx[1]);
    }
    else if constexpr (RANK == 3) {
      return this->operator()(idx[0], idx[1], idx[2]);
    }
    else if constexpr (RANK == 4) {
      return this->operator()(idx[0], idx[1], idx[2], idx[3]);
    }            
  }    

  /**
   * Rank-0 operator() getter
   *
   * @returns value in tensor
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ const T &operator()() const noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 0, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ const T &operator()() const noexcept
  {
#endif
    return *ldata_;
  }

  /**
   * Rank-0 operator() setter
   *
   * @returns reference to value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ T &operator()() noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 0, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ T &operator()() noexcept
  {
#endif
    return *ldata_;
  }

  /**
   * Rank-1 operator() getter
   *
   * @param id0
   *   Index into first dimension
   *
   * @returns value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0) const noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 1, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0) const noexcept
  {
#endif
    return *(ldata_ + s_[0] * id0);
  }

  /**
   * Rank-1 operator() setter
   *
   * @param id0
   *   Index into first dimension
   *
   * @returns reference to value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0) noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 1, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0) noexcept
  {
#endif
    return *(ldata_ + s_[0] * id0);
  }

  /**
   * Rank-2 operator() getter
   *
   * @param id0
   *   Index into first dimension
   *
   * @param id1
   *   Index into second dimension
   *
   * @returns value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0,
                                          index_t id1) const noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 2, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0,
                                                 index_t id1) const noexcept
  {
#endif
    return *(ldata_ + s_[0] * id0 + s_[1] * id1);
  }

  /**
   * Rank-2 operator() setter
   *
   * @param id0
   *   Index into first dimension
   *
   * @param id1
   *   Index into second dimension
   *
   * @returns reference to value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1) noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 2, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1) noexcept
  {
#endif
    return *(ldata_ + s_[0] * id0 + s_[1] * id1);
  }

  /**
   * Rank-3 operator() getter
   *
   * @param id0
   *   Index into first dimension
   *
   * @param id1
   *   Index into second dimension
   *
   * @param id2
   *   Index into third dimension
   *
   * @returns value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0, index_t id1,
                                          index_t id2) const noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 3, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0, index_t id1,
                                                 index_t id2) const noexcept
  {
#endif
    return *(ldata_ + s_[0] * id0 + s_[1] * id1 + s_[2] * id2);
  }

  /**
   * Rank-3 operator() setter
   *
   * @param id0
   *   Index into first dimension
   *
   * @param id1
   *   Index into second dimension
   *
   * @param id2
   *   Index into third dimension
   *
   * @returns reference to value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1,
                                    index_t id2) noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 3, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1,
                                           index_t id2) noexcept
  {
#endif
    return *(ldata_ + s_[0] * id0 + s_[1] * id1 + s_[2] * id2);
  }

  /**
   * Rank-4 operator() getter
   *
   * @param id0
   *   Index into first dimension
   *
   * @param id1
   *   Index into second dimension
   *
   * @param id2
   *   Index into third dimension
   *
   * @param id3
   *   Index into fourth dimension
   *
   * @returns value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0, index_t id1, index_t id2,
                                          index_t id3) const noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 4, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ const T &
  operator()(index_t id0, index_t id1, index_t id2, index_t id3) const noexcept
  {
#endif
    return *(ldata_ + s_[0] * id0 + s_[1] * id1 + s_[2] * id2 + s_[3] * id3);
  }

  /**
   * Rank-4 operator() setter
   *
   * @param id0
   *   Index into first dimension
   *
   * @param id1
   *   Index into second dimension
   *
   * @param id2
   *   Index into third dimension
   *
   * @param id3
   *   Index into fourth dimension
   *
   * @returns reference to value at given index
   *
   */
#ifdef DOXYGEN_ONLY
  __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1, index_t id2,
                                    index_t id3) noexcept
  {
#else
  template <int M = RANK, std::enable_if_t<M == 4, bool> = true>
  inline __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1,
                                           index_t id2, index_t id3) noexcept
  {
#endif
    return *(ldata_ + s_[0] * id0 + s_[1] * id1 + s_[2] * id2 + s_[3] * id3);
  }

  /**
   * Create an overlapping tensor view
   *
   * Creates and overlapping tensor view where an existing tensor can be
   * repeated into a higher rank with overlapping elements. For example, the
   * following 1D tensor [1 2 3 4 5] could be cloned into a 2d tensor with a
   * window size of 2 and overlap of 1, resulting in:
   *
   * [1 2
   *  2 3
   *  3 4
   *  4 5]
   *
   * Currently this only works on 1D tensors going to 2D, but may be expanded
   * for higher dimensions in the future. Note that if the window size does not
   * divide evenly into the existing column dimension, the view may chop off the
   * end of the data to make the tensor rectangular.
   *
   * @param windows
   *   Window size (columns in output)
   * @param strides
   *   Strides between data elements
   *
   * @returns Overlapping view of data
   *
   */
#ifdef DOXYGEN_ONLY
  tensor_t<T, RANK + 1>
  OverlapView(std::initializer_list<index_t> const &windows,
              std::initializer_list<index_t> const &strides) const
  {
#else
  template <int M = RANK, std::enable_if_t<M == 1, bool> = true>
  inline tensor_t<T, RANK + 1>
  OverlapView(std::initializer_list<index_t> const &windows,
              std::initializer_list<index_t> const &strides) const
  {
#endif
    index_t n[RANK + 1], s[RANK + 1];

    // This only works for 1D tensors going to 2D at the moment. Generalize to
    // higher dims later
    index_t window_size = *(windows.begin());
    index_t stride_size = *(strides.begin());

    MATX_ASSERT(stride_size < window_size, matxInvalidSize);
    MATX_ASSERT(stride_size > 0, matxInvalidSize);

    // Figure out the actual length of the signal we can use. It might be
    // shorter than the original tensor if the window/stride doesn't line up
    // properly to make a rectangular matrix.
    index_t adj_el = Size(0) - window_size;
    while ((adj_el % stride_size) != 0) {
      adj_el--;
    }

    n[1] = window_size;
    s[1] = 1;
    n[0] = adj_el / stride_size + 1;
    s[0] = stride_size;

    return tensor_t<T, RANK + 1>(data_, data_, n, s, refcnt_);
  }

  /**
   * Clone a tensor into a higher-dimension tensor
   *
   * Clone() allows a copy-less method to clone data into a higher dimension
   * tensor. The underlying data does not grow or copy, but instead the indices
   * of the higher-ranked tensor access the original data potentially multiple
   * times. Clone is similar to MATLAB's repmat() function where it's desired
   * to take a tensor of a lower dimension and apply an operation with it to a
   * tensor in a higher dimension by broadcasting the values.
   *
   * For example, in a
   * rank=2 tensor that's MxN, and a rank=1 tensor that's 1xN, Clone() can take
   * the rank=1 tensor and broadcast to an MxN rank=2 tensor, and operations
   * such as the Hadamard product can be performed. In this example, the final
   * operation will benefit heavily from device caching since the same 1xN
   * rank=1 tensor will be accessed M times.
   *
   * @param clones
   *   List of sizes of each dimension to clone. Parameter length must match
   * rank of tensor. A special sentinel value of matxKeepDim should be used when
   * the dimension from the original tensor is to be kept.
   *
   * @returns Cloned view representing the higher-dimension tensor
   *
   */
#ifdef DOXYGEN_ONLY
  tensor_t<T, N> Clone(const index_t (&clones)[N]) const
  {
#else
  template <int N, std::enable_if_t<(N <= 4 && N > RANK), bool> = true>
  inline tensor_t<T, N> Clone(const index_t (&clones)[N]) const
  {
#endif
    index_t n[N], s[N];

    int d = 0;

#pragma unroll
    for (int i = 0; i < N; i++) {
      index_t size = clones[i];

      if (size == matxKeepDim) {
        n[i] = Size(d);
        if constexpr (RANK == 0) {
          s[i] = 1;
        }
        else {
          s[i] = s_[d];
        }
        d++;
      }
      else {
        n[i] = size;
        s[i] = 0;
      }
    }
    MATX_ASSERT_STR(d == RANK, matxInvalidDim,
                    "Must keep as many dimension as the original tensor has");

    return tensor_t<T, N>(data_, data_, n, s, refcnt_);
  }

  /**
   * Rank-0 initializer list setting
   *
   * @param val
   *   0 initializer list value
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<M == 0, bool> = true>
  inline __MATX_HOST__ void SetVals(T const &val) noexcept
  {
    this->operator()() = val;
  }

  /**
   * Rank-1 non-complex or rank-0 initializer list setting
   *
   * @param vals
   *   1D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<(!is_cuda_complex_v<T> && M == 1) ||
                                               (is_cuda_complex_v<T> && M == 0),
                                           bool> = true>
  inline __MATX_HOST__ void SetVals(const std::initializer_list<T> &vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      if constexpr (is_cuda_complex_v<T>) {
        typename T::value_type real = (vals.begin() + i)->real();
        typename T::value_type imag = (vals.begin() + i + 1)->real();
        this->operator()() = {real, imag};
      }
      else {
        this->operator()(i) = *(vals.begin() + i);
      }
    }
  }

  /**
   * Rank-2 non-complex or rank-1 initializer list setting
   *
   * @param vals
   *   1D/2D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<(!is_cuda_complex_v<T> && M == 2) ||
                                               (is_cuda_complex_v<T> && M == 1),
                                           bool> = true>
  inline __MATX_HOST__ void
  SetVals(const std::initializer_list<const std::initializer_list<T>>
              &vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        if constexpr (is_cuda_complex_v<T>) {
          typename T::value_type real =
              ((vals.begin() + i)->begin() + j)->real();
          typename T::value_type imag =
              ((vals.begin() + i)->begin() + j + 1)->real();
          this->operator()(i) = {real, imag};
          j++;
        }
        else {
          this->operator()(i, j) = *((vals.begin() + i)->begin() + j);
        }
      }
    }
  }

  /**
   * Rank-3 non-complex or rank-2 complex initializer list setting
   *
   * @param vals
   *   3D/2D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<(!is_cuda_complex_v<T> && M == 3) ||
                                               (is_cuda_complex_v<T> && M == 2),
                                           bool> = true>
  inline __MATX_HOST__ void
  SetVals(const std::initializer_list<
          const std::initializer_list<const std::initializer_list<T>>>
              vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        for (size_t k = 0; k < ((vals.begin() + i)->begin() + j)->size(); k++) {
          if constexpr (is_cuda_complex_v<T>) {
            typename T::value_type real =
                (((vals.begin() + i)->begin() + j)->begin() + k)->real();
            typename T::value_type imag =
                (((vals.begin() + i)->begin() + j)->begin() + k + 1)->real();
            this->operator()(i, j) = {real, imag};
            k++;
          }
          else {
            this->operator()(i, j, k) =
                *(((vals.begin() + i)->begin() + j)->begin() + k);
          }
        }
      }
    }
  }

  /**
   * Rank-4 non-complex or rank-3 complex initializer list setting
   *
   * @param vals
   *   3D/4D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<(!is_cuda_complex_v<T> && M == 4) ||
                                               (is_cuda_complex_v<T> && M == 3),
                                           bool> = true>
  inline __MATX_HOST__ void
  SetVals(const std::initializer_list<const std::initializer_list<
              const std::initializer_list<const std::initializer_list<T>>>>
              &vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        for (size_t k = 0; k < ((vals.begin() + i)->begin() + j)->size(); k++) {
          for (size_t l = 0;
               l < (((vals.begin() + i)->begin() + j)->begin + k)->size();
               l++) {
            if constexpr (is_cuda_complex_v<T>) {
              typename T::value_type real =
                  ((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                   l)
                      ->real();
              typename T::value_type imag =
                  ((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                   l + 1)
                      ->real();
              this->operator()(i, j, k) = {real, imag};
              l++;
            }
            else {
              this->operator()(i, j, k, l) =
                  *((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                    l);
            }
          }
        }
      }
    }
  }

  /**
   * Rank-4 complex initializer list setting
   *
   * @param vals
   *   4D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK,
            std::enable_if_t<is_cuda_complex_v<T> && M == 4, bool> = true>
  inline __MATX_HOST__ void
  SetVals(const std::initializer_list<
          const std::initializer_list<const std::initializer_list<
              const std::initializer_list<const std::initializer_list<T>>>>>
              &vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        for (size_t k = 0; k < ((vals.begin() + i)->begin() + j)->size(); k++) {
          for (size_t l = 0;
               l < (((vals.begin() + i)->begin() + j)->begin + k)->size();
               l++) {
            for (size_t m = 0;
                 m < ((((vals.begin() + i)->begin() + j)->begin + k)->begin + l)
                         ->size();
                 m++) {
              typename T::value_type real =
                  (((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                    l)
                       ->begin() +
                   m)
                      ->real();
              typename T::value_type imag =
                  (((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                    l)
                       ->begin() +
                   m + 1)
                      ->real();
              this->operator()(i, j, k, l) = {real, imag};
              m++;
            }
          }
        }
      }
    }
  }

  /**
   * Slice a tensor either within the same dimension or to a lower dimension
   *
   * Slice() allows a copy-less method to extract a subset of data from one or
   * more dimensions of a tensor. This includes completely dropping an unwanted
   * dimension, or simply taking a piece of a wanted dimension. Slice() is very
   * similar to indexing operations in both Python and MATLAB.
   *
   * @param firsts
   *   List of starting index into each dimension. Indexing is 0-based
   *
   * @param ends
   *   List of ending index into each dimension. Indexing is 0-based
   *   Two special sentinel values can be used:
   *     1) matxEnd is used to indicate the end of that particular
   *        dimension without specifying the size. This is similar to "end" in
   *        MATLAB and leaving off an end in Python "a[1:]"
   *
   *     2) matxDropDim is used to slice (drop) a dimension entirely. This
   * results in a tensor with a smaller rank than the original
   *
   * @param strides
   *   List of strides for each dimension.
   *   A special sentinel value of matxKeepStride is used to keep the existing
   * stride of the dimension
   *
   * @returns Sliced view of tensor
   *
   */
#ifdef DOXYGEN_ONLY
  tensor_t<T, N> Slice(const index_t (&firsts)[RANK],
                       const index_t (&ends)[RANK],
                       const index_t (&strides)[RANK]) const
  {
#else
  template <int N = RANK,
            std::enable_if_t<(N <= RANK && RANK > 0), bool> = true>
  inline tensor_t<T, N> Slice([[maybe_unused]] const index_t (&firsts)[RANK],
                              [[maybe_unused]] const index_t (&ends)[RANK],
                              [[maybe_unused]] const index_t (&strides)[RANK]) const
  {
#endif
    index_t n[N] = {};
    index_t s[N] = {};
    T *data = data_;
    int d = 0;
    bool def_stride = (strides[0] == -1);

#pragma unroll
    for (int i = 0; i < RANK; i++) {
      index_t first = firsts[i];
      index_t end = ends[i];

      MATX_ASSERT_STR(first < end, matxInvalidSize, "Slice must be at least one element long");

      [[maybe_unused]] index_t stride_mult = (def_stride || strides[i] == matxKeepStride)
                                ? 1
                                : strides[i]; // custom stride

      MATX_ASSERT_STR(first < end, matxInvalidParameter,
                      "Starting slice must be less than end slice");
      MATX_ASSERT_STR(first < Size(i), matxInvalidParameter,
                      "Index to slice is larger than the tensor rank");

      // offset by first
      data += first * s_[i];

      if (end != matxDropDim) {
        if (end == matxEnd) {
          n[d] = Size(i) - first;
        }
        else {
          n[d] = end - first;
        }

        // New length is shorter if we have a non-1 stride
        n[d] = static_cast<index_t>(std::ceil(
            static_cast<double>(n[d]) / static_cast<double>(stride_mult)));

        s[d] = s_[i] * stride_mult;
        d++;
      }
    }

    MATX_ASSERT_STR(d == N, matxInvalidDim,
                    "Number of indices must match the target rank to slice to");

    return tensor_t<T, N>(data_, data, n, s, refcnt_);
  }

  /**
   * Slice a tensor either within the same dimension or to a lower dimension
   *
   * Slice() allows a copy-less method to extract a subset of data from one or
   * more dimensions of a tensor. This includes completely dropping an unwanted
   * dimension, or simply taking a piece of a wanted dimension. Slice() is very
   * similar to indexing operations in both Python and MATLAB.
   *
   * @param firsts
   *   List of starting index into each dimension. Indexing is 0-based
   *
   * @param ends
   *   List of ending index into each dimension. Indexing is 0-based
   *   Two special sentinel values can be used:
   *     1) matxEnd is used to indicate the end of that particular
   *        dimension without specifying the size. This is similar to "end" in
   *        MATLAB and leaving off an end in Python "a[1:]"
   *
   *     2) matxDropDim is used to slice (drop) a dimension entirely. This
   * results in a tensor with a smaller rank than the original
   *
   * @returns Sliced view of tensor
   *
   */
#ifdef DOXYGEN_ONLY
  tensor_t<T, N> Slice(const index_t (&firsts)[RANK],
                       const index_t (&ends)[RANK]) const
  {
#else
  template <int N = RANK, std::enable_if_t<(N <= RANK && RANK > 0), bool> = true>
  inline tensor_t<T, N> Slice(const index_t (&firsts)[RANK],
                              const index_t (&ends)[RANK]) const
  {
#endif
    const index_t strides[RANK] = {-1};
    return Slice<N>(firsts, ends, strides);
  }

  /**
   * Get the total size of the underlying data, including the multiple of the
   * type
   *
   * @return
   *    The size (in bytes) of all dimensions combined
   */
  inline size_t Bytes() const noexcept { return sizeof(T) * TotalSize(); };

  /**
   * Print a value
   *
   * Type-agnostic function to print a value to stdout
   *
   * @param val
   */
  inline __MATX_HOST__ __MATX_DEVICE__ void PrintVal(const T &val) const noexcept
  {
    if constexpr (is_complex_v<T>) {
      printf("%.4f%+.4fj ", static_cast<float>(val.real()),
             static_cast<float>(val.imag()));
    }
    else if constexpr (is_matx_half_v<T> || is_half_v<T>) {
      printf("%.4f ", static_cast<float>(val));
    }
    else if constexpr (std::is_floating_point_v<T>) {
      printf("%.4f ", val);
    }
    else if constexpr (std::is_same_v<T, long long int>) {
      printf("%lld ", val);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
      printf("%" PRId64 " ", val);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
      printf("%" PRId32 " ", val);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
      printf("%" PRId16 " ", val);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
      printf("%" PRId8 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
      printf("%" PRIu64 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
      printf("%" PRIu32 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
      printf("%" PRIu16 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
      printf("%" PRIu8 " ", val);
    }
  }

  /**
   * Print a tensor
   *
   * Type-agnostic function to print a tensor to stdout
   *
   */
  template <typename ... Args>
  __MATX_HOST__ __MATX_DEVICE__ void InternalPrint(Args ...dims) const noexcept
  {
    MATX_STATIC_ASSERT(RANK == sizeof...(Args), "Number of dimensions to print must match tensor rank");

    if constexpr (sizeof...(Args) == 0) {
      PrintVal(this->operator()());
      printf("\n");
    }
    else if constexpr (sizeof...(Args) == 1) {
      auto& k = pp_get<0>(dims...);
      for (index_t _k = 0; _k < ((k == 0) ? Size(0) : k); _k++) {
        printf("%06lld: ", _k);
        PrintVal(this->operator()(_k));
        printf("\n");
      }
    }
    else if constexpr (sizeof...(Args) == 2) {
      auto& k = pp_get<0>(dims...);
      auto& l = pp_get<1>(dims...);
      for (index_t _k = 0; _k < ((k == 0) ? Size(0) : k); _k++) {
        for (index_t _l = 0; _l < ((l == 0) ? Size(1) : l); _l++) {
          if (_l == 0)
            printf("%06lld: ", _k);

          PrintVal(this->operator()(_k, _l));
        }
        printf("\n");
      }
    }
    else if constexpr (sizeof...(Args) == 3) {
      auto& j = pp_get<0>(dims...);
      auto& k = pp_get<1>(dims...);
      auto& l = pp_get<2>(dims...);
      for (index_t _j = 0; _j < ((j == 0) ? Size(0) : j); _j++) {
        printf("[%06lld,:,:]\n", _j);
        for (index_t _k = 0; _k < ((k == 0) ? Size(1) : k); _k++) {
          for (index_t _l = 0; _l < ((l == 0) ? Size(2) : l); _l++) {
            if (_l == 0)
              printf("%06lld: ", _k);

            PrintVal(this->operator()(_j, _k, _l));
          }
          printf("\n");
        }
        printf("\n");
      }      
    }
    else if constexpr (sizeof...(Args) == 4) {
      auto& i = pp_get<0>(dims...);
      auto& j = pp_get<1>(dims...);
      auto& k = pp_get<2>(dims...);
      auto& l = pp_get<3>(dims...); 
      for (index_t _i = 0; _i < ((i == 0) ? Size(0) : i); _i++) {
        for (index_t _j = 0; _j < ((j == 0) ? Size(1) : j); _j++) {
          printf("[%06lld,%06lld,:,:]\n", _i, _j);
          for (index_t _k = 0; _k < ((k == 0) ? Size(2) : k); _k++) {
            for (index_t _l = 0; _l < ((l == 0) ? Size(3) : l); _l++) {
              if (_l == 0)
                printf("%06lld: ", _k);

              PrintVal(this->operator()(_i, _j, _k, _l));
            }
            printf("\n");
          }
          printf("\n");
        }
      }
    }
  }  

  /**
   * Print a tensor
   *
   * Type-agnostic function to print a tensor to stdout
   *
   */
  template <typename ... Args>
  inline void Print(Args ...dims) const
  {
#ifdef __CUDACC__    
    auto kind = GetPointerKind(data_);
    cudaDeviceSynchronize();
    if (HostPrintable(kind)) {
      InternalPrint(dims...);
    }
    else if (DevicePrintable(kind)) {
      if (PRINT_ON_DEVICE) {
        PrintKernel<<<1, 1>>>(*this, dims...);
      }
      else {
        tensor_t<T, RANK> tmpv(this->shape_, (const index_t(&)[RANK])this->s_);
        cudaMemcpy(tmpv.Data(), this->Data(), tmpv.Bytes(),
                   cudaMemcpyDeviceToHost);
        tmpv.Print(dims...);
      }
    }
#else
    InternalPrint(dims...);
#endif    
  }

  /**
   * @brief Returns an N-D coordinate as an array corresponding to the absolute index abs
   * 
   * @param abs Absolute index
   * @return std::array of indices 
   */
  __forceinline__ std::array<index_t, RANK> GetIdxFromAbs(index_t abs) {
    std::array<index_t, RANK> indices;
    std::array sh = shape_.AsArray();
    
    for (int idx = 0; idx < RANK; idx++) {
      if (idx == RANK-1) {
        indices[RANK-1] = abs;
      }
      else {
        index_t prod = std::accumulate(sh.data() + idx + 1, sh.data() + RANK, 1, std::multiplies<index_t>());
        indices[idx] = abs / prod;
        abs -= prod * indices[idx];
      }
    }

    return indices;
  }

private:
  /**
   * Allocate managed memory backing the view
   *
   * Used when no user-defined pointer is passed in
   **/
  inline void Allocate()
  {
    matxAlloc((void **)&data_, Bytes());
    MATX_ASSERT(data_ != NULL, matxOutOfMemory);
    ldata_ = data_;
    refcnt_ = new cuda::std::atomic<uint32_t>{1};
  }

  /**
   * Free managed memory backing the view
   *
   * Used when no user-defined pointer is passed in
   **/
  inline void Free()
  {
    if (refcnt_ == nullptr) {
      return;
    }

    if (--(*refcnt_) == 0) {
      MATX_ASSERT(data_ != nullptr, matxAssertError);
      matxFree(data_);
      data_ = nullptr;
      delete refcnt_;
    }
  };

  T *data_; // Local data pointer to this tensor view
  T *ldata_;
  cuda::std::atomic<uint32_t> *refcnt_;
  tensorShape_t<RANK> shape_;
  std::array<index_t, RANK> s_; // +1 to avoid zero sized array
};




// make_tensor helpers
/**
 * Create a 0D tensor with managed memory
 *
 **/
template <typename T>
tensor_t<T,0> make_tensor() {
  return tensor_t<T,0>{};
}

/**
 * Create a 0D tensor with user memory
 *
 * @param data
 *   Pointer to device data
 **/
template <typename T>
tensor_t<T,0> make_tensor(T *const data) {
  return tensor_t<T,0>{data};
}

/**
 * Create a tensor with managed memory
 *
 * @param shape
 *   Shape of tensor
 **/
template <typename T, int RANK>
tensor_t<T,RANK> make_tensor(const index_t (&shape)[RANK]) {
  return tensor_t<T,RANK>{shape};
}

/**
 * Create a tensor with managed memory
 *
 * @param shape
 *   Shape of tensor
 * @param strides
 *   Strides of tensor
 **/
template <typename T, int RANK>
tensor_t<T,RANK> make_tensor(const index_t (&shape)[RANK], const index_t (&strides)[RANK]) {
  return tensor_t<T,RANK>{shape, strides};
}


/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, int RANK>
tensor_t<T,RANK> make_tensor(T *const data, const index_t (&shape)[RANK]) {
  return tensor_t<T,RANK>{data, shape};
}


/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param strides
 *   Strides of tensor
 **/
template <typename T, int RANK>
tensor_t<T,RANK> make_tensor(T *const data, const index_t (&shape)[RANK], const index_t (&strides)[RANK]) {
  return tensor_t<T,RANK>{data, shape, strides};
}

} // end namespace matx
