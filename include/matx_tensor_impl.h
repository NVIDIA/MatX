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

#include <type_traits>
#include "matx_error.h"
#include "matx_shape.h"
#include "matx_set.h"
#include "matx_defines.h"
#include "matx_type_utils.h"
#include "matx_tensor_utils.h"
#include "matx_exec_kernel.h"

namespace matx {

  
template <typename T>
class BaseOp
{
public:
  using matxop = bool;

  // Launch work in the stream
  void run(cudaStream_t stream = 0) noexcept
  {
    exec(*static_cast<T *>(this), CUDADeviceExecutor{stream});
  }

  // Record an event after the work
  void run(cudaEvent_t ev, cudaStream_t stream = 0) noexcept
  {
    exec(*static_cast<T *>(this), CUDADeviceExecutor{stream});
    cudaEventRecord(ev, stream);
  }

  template <typename Ex, std::enable_if_t<is_executor_t<Ex>(), bool> = true>
  void run (Ex ex) {
    exec(*static_cast<T *>(this), ex);
  }
};



/**
 * @brief Bare implementation of tensor class
 * 
 * Defines the minimum operations needed for a tensor to be useful. This class
 * should be limited to primitive types. The bare class contains operations that
 * are necessary to function on a device (getters/setters), but does not contain
 * any ownership semantics.
 * 
 * This class is designed to be inherited from more capable tensor classes like
 * tensor_t below. Because of the asynchronous nature of GPU devices, derived classes
 * are responsible for handling ownership and lifetime of the pointer in a 
 * tensor_impl_t. 
 * 
 * @tparam T Type of tensor
 * @tparam RANK Rank of tensor
 */
template <typename T, int RANK> 
class tensor_impl_t {
  public:
    // Type specifier for reflection on class
    using type = T; // TODO is this necessary
    using scalar_type = T;
    using tensor_view = bool;

    // Type specifier for signaling this is a matx operation
    using matxop = bool;

    __MATX_HOST__ __MATX_DEVICE__ tensor_impl_t<T, RANK>(tensor_impl_t<T, RANK> const &rhs) noexcept
        : ldata_(rhs.ldata_), shape_(rhs.shape_), s_(rhs.s_) { }

    __MATX_HOST__ __MATX_DEVICE__ tensor_impl_t<T, RANK>(tensor_impl_t<T, RANK> const &&rhs) noexcept
        : ldata_(rhs.ldata_), shape_(rhs.shape_), s_(rhs.s_) { }

    __MATX_HOST__ __MATX_DEVICE__ tensor_impl_t<T, RANK>(T *const ldata,
                  const tensorShape_t<RANK> &shape, const std::array<index_t, RANK> strides) noexcept
        : ldata_(ldata), shape_(shape), s_(strides) { }        


    __MATX_INLINE__ ~tensor_impl_t() = default;

    /**
     * Constructor for a rank-0 tensor (scalar).
     */
    template <int M = RANK, std::enable_if_t<M == 0, bool> = true> tensor_impl_t() {};

    /**
     * Constructor for a rank-0 tensor (scalar).
     *
     * @param data
     *   Data pointer
     */
    template <int M = RANK, std::enable_if_t<M == 0, bool> = true>
    tensor_impl_t(T *const data) : ldata_(data) { }

    /**
     * Constructor for a rank-1 and above tensor.
     *
     * @param shape
     *   Tensor shape
     */
    __MATX_INLINE__ tensor_impl_t(tensorShape_t<RANK> const &shape) : shape_(shape)
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
    }

    /**
     * Constructor for a rank-1 and above tensor.
     *
     * @param shape
     *   Tensor shape
     * @param strides
     *   Tensor strides
     */
    __MATX_INLINE__ tensor_impl_t(tensorShape_t<RANK> const &shape, const index_t (&strides)[RANK])
        : shape_(shape)
    {
      for (int i = 0; i < RANK; i++) {
        MATX_ASSERT_STR(shape.Size(i) > 0, matxInvalidSize,
                        "Must specify size larger than 0 for each dimension");
      }

      memcpy((void *)s_.data(), (void *)strides, s_.size() * sizeof(index_t));
    }

    /**
     * Constructor for a rank-1 and above tensor using a shape input
     *
     * @param shape
     *   Sizes for each dimension. Length of sizes must match RANK
     */
    __MATX_INLINE__ tensor_impl_t(const index_t (&shape)[RANK])
        : tensor_impl_t(tensorShape_t<RANK>{static_cast<index_t const *>(shape)})
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
    __MATX_INLINE__ tensor_impl_t(T *const ldata, const tensorShape_t<RANK> &shape)
        : ldata_(ldata), shape_(shape)
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
    __MATX_INLINE__ tensor_impl_t(T *const data, const index_t (&shape)[RANK]) noexcept
        : tensor_impl_t(data, tensorShape_t<RANK>{static_cast<index_t const *>(shape)})
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
    __MATX_INLINE__ tensor_impl_t(T *const ldata,
                    tensorShape_t<RANK> const &shape,
                    const index_t (&strides)[RANK])
        : ldata_(ldata), shape_(shape)
    {
      for (int i = 0; i < RANK; i++) {
        MATX_ASSERT_STR(shape.Size(i) > 0, matxInvalidSize,
                        "Must specify size larger than 0 for each dimension");
      }

      memcpy((void *)s_.data(), (void *)strides, s_.size() * sizeof(index_t));
    }

    // Lazy operators
    
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator+=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator+=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this + op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator-=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator-=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this - op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator*=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator*=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this * op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator/=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator/=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this / op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator<<=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator<<=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this << op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator>>=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator>>=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this >> op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator|=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator|=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this | op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator&=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator&=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this & op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator^=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator^=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
      return set(*this, *this ^ op_base);
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator%=(const tensor_impl_t<T, RANK> &op)
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator%=(const T2 &op)
    {
      const typename base_type<T2>::type &op_base = op;
        return set(*this, *this % op_base);
    }

    /**
     * Get the shape the tensor from the underlying data
     *
     * @return
     *    A shape of the data with the appropriate dimensions set
     */
    __MATX_INLINE__ tensorShape_t<RANK> Shape() const noexcept { return this->shape_; }    

  /**
   * Check if a tensor is linear in memory for all elements in the view
   *
   * @return
   *    The size of the dimension
   */
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ bool IsLinear() const noexcept
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
     * operator() getter with an array index
     *
     * @returns value in tensor
     *
     */
    template <int M = RANK, std::enable_if_t<M >= 1, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T &operator()(const std::array<index_t, RANK> &idx) const noexcept
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
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T &operator()(const std::array<index_t, RANK> &idx) noexcept
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
    template <int M = RANK, std::enable_if_t<M == 0, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T &operator()() const noexcept
    {
      return *ldata_;
    }

    /**
     * Rank-0 operator() setter
     *
     * @returns reference to value at given index
     *
     */
    template <int M = RANK, std::enable_if_t<M == 0, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T &operator()() noexcept
    {
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
    template <int M = RANK, std::enable_if_t<M == 1, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0) const noexcept
    {
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
    template <int M = RANK, std::enable_if_t<M == 1, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0) noexcept
    {
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
    template <int M = RANK, std::enable_if_t<M == 2, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0,
                                                  index_t id1) const noexcept
    {
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
    template <int M = RANK, std::enable_if_t<M == 2, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1) noexcept
    {
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
    template <int M = RANK, std::enable_if_t<M == 3, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T &operator()(index_t id0, index_t id1,
                                                  index_t id2) const noexcept
    {
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
    template <int M = RANK, std::enable_if_t<M == 3, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1,
                                            index_t id2) noexcept
    {
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
    template <int M = RANK, std::enable_if_t<M == 4, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T &
    operator()(index_t id0, index_t id1, index_t id2, index_t id3) const noexcept
    {
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
    template <int M = RANK, std::enable_if_t<M == 4, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T &operator()(index_t id0, index_t id1,
                                            index_t id2, index_t id3) noexcept
    {
      return *(ldata_ + s_[0] * id0 + s_[1] * id1 + s_[2] * id2 + s_[3] * id3);
    }  

  /**
   * Get the rank of the tensor
   *
   * @returns Rank of the tensor
   *
   */
  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }


  /**
   * Get the total number of elements in the tensor
   *
   *
   * @returns Total number of elements across all dimensions
   *
   */
  __MATX_INLINE__ index_t __MATX_HOST__ __MATX_DEVICE__ TotalSize() const noexcept { return shape_.TotalSize(); }  

  /**
   * Get the size of a single dimension of the tensor
   *
   * @param dim
   *   Desired dimension
   *
   * @returns Number of elements in dimension
   *
   */
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(uint32_t dim) const noexcept
  {
    return shape_.Size(dim);
  }

  /**
   * Get the size of the last dimension
   *
   * @return
   *    The size of the dimension
   */
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Lsize() const noexcept
  {
    return shape_.Size(Rank() - 1);
  }

  protected:
    T *ldata_;
    tensorShape_t<RANK> shape_;
    std::array<index_t, RANK> s_; // +1 to avoid zero sized array    
};


};