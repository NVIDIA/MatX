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
#include <cuda/std/functional>
#include "matx_error.h"
#include "matx_set.h"
#include "matx_defines.h"
#include "matx_tensor_desc.h"
#include "matx_type_utils.h"
#include "matx_tensor_utils.h"
#include "matx_exec_kernel.h"
#include "matx_make.h"

namespace matx {

/**
 * @brief Provides a base class with functions common to all operators
 * 
 * @tparam T Type of operator
 */
template <typename T>
class BaseOp
{
public:
  using matxop = bool;  ///< Is a MatX custom operator
  using value_type = T; ///< Value type for type traits

  /**
   * @brief Launch kernel in a GPU stream
   * 
   * @param stream CUDA stream
   */
  void run(cudaStream_t stream = 0) noexcept
  {
    exec(*static_cast<T *>(this), CUDADeviceExecutor{stream});
  }

  /**
   * @brief Launch work in a GPU stream and record an event
   * 
   * @param ev CUDA event
   * @param stream CUDA stream
   */
  void run(cudaEvent_t ev, cudaStream_t stream = 0) noexcept
  {
    exec(*static_cast<T *>(this), CUDADeviceExecutor{stream});
    cudaEventRecord(ev, stream);
  }

  /**
   * @brief Launch work in an arbitrary executor
   * 
   * @tparam Ex Executor type
   * @param ex Executor
   */
  template <typename Ex>
  void run (Ex ex) {
    static_assert(is_executor_t<Ex>(), "Ex must be a MatX executor type");
    exec(*static_cast<T *>(this), ex);
  }

};

/**
 * @brief Pprovides a base class for reducing the boilerplate code needed
 * when defining an operator. This is particularly useful in user code with
 * many custom operators that don't want to repeat the Size and Rank functions.
 * 
 * @tparam T Type of operator
 * @tparam RankOp Type of operator providing rank information
 */
template <typename T, typename RankOp>
class BaseOpCustom : public BaseOp<T>
{
public:
  using matxop = bool;  ///< Type trait to indicate this is an operator
  using value_type = T; ///< Type trait of operator type
  std::array<index_t, RankOp::Rank()> size_; ///< Size of each dimension

  BaseOpCustom() = delete;

  /**
   * @brief Construct a new Base Op Custom object
   * 
   * @param size Size of each dimension
   */
  BaseOpCustom(const std::array<index_t, RankOp::Rank()> &size) :
    size_(size) {}

  /**
   * @brief Return rank of operator
   * 
   * @return Operator rank
   */
  static inline constexpr int32_t Rank()
  {
    return RankOp::Rank();
  }  

  /**
   * @brief Return size of operator on a given dimension
   * 
   * @return Operator size on dimension dim
   */
  index_t inline __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const
  {
    return size_[dim];
  }
};


namespace detail {

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
template <typename T, int RANK, typename Desc = tensor_desc_cr_ds_64_64_t<RANK>> 
class tensor_impl_t {
  public:
    // Type specifier for reflection on class
    using type = T; // TODO is this necessary
    using scalar_type = T;
    using value_type = T;
    using tensor_view = bool;
    using desc_type = Desc;
    using shape_type = typename Desc::shape_type;
    using stride_type = typename Desc::stride_type;

    static constexpr bool PRINT_ON_DEVICE = false;    

    // Type specifier for signaling this is a matx operation
    using matxop = bool;

    __MATX_HOST__ tensor_impl_t(const tensor_impl_t &) = default;
    __MATX_HOST__ tensor_impl_t(tensor_impl_t &&) = default;
    __MATX_HOST__ tensor_impl_t& operator=(tensor_impl_t &&) = default;


    __MATX_INLINE__ ~tensor_impl_t() = default;

    /**
     * Constructor for a rank-0 tensor (scalar).
     */
    tensor_impl_t() {
      static_assert(RANK == 0, "Default constructor is only for rank 0 tensors.");
    }

    /**
     * Constructor for a rank-0 tensor (scalar).
     *
     * @param data
     *   Data pointer
     */
    tensor_impl_t(T *const data) : ldata_(data) { 
      static_assert(RANK == 0, "tensor_impl_t with single pointer parameter must be a rank 0 tensor");
    }

    /**
     * Constructor for a rank-1 and above tensor.
     *
     * @param shape
     *   Tensor shape
     */
    template <typename ShapeType, 
              std::enable_if_t<!is_tensor_view_v<typename remove_cvref<ShapeType>::type> && !is_matx_descriptor_v<typename remove_cvref<ShapeType>::type>, bool> = true>
    __MATX_INLINE__ tensor_impl_t(ShapeType &&shape) : desc_(std::forward<ShapeType>(shape))
    {
    }

    /**
     * Constructor for a rank-1 and above tensor.
     *
     * @param shape
     *   Tensor shape
     * @param strides
     *   Tensor strides
     */
    template <typename ShapeType, typename StrideType>
    __MATX_INLINE__ tensor_impl_t(ShapeType &&shape, StrideType &&strides)
        : desc_(std::forward<ShapeType>(shape), std::forward<StrideType>(strides))
    {
    }    

    /**
     * Constructor for a rank-1 and above tensor using a user pointer and shape
     * input
     *
     * @tparam ShapeType
     *   Type of shape
     * @param ldata
     *   Offset data pointer (start of view)
     * @param shape
     *   Sizes for each dimension. Length of sizes must match RANK
     */
    template <typename ShapeType, std::enable_if_t<!is_matx_descriptor_v<typename remove_cvref<ShapeType>::type>, bool> = true>
    __MATX_INLINE__ tensor_impl_t(T *const ldata, ShapeType &&shape)
        : ldata_(ldata), desc_(std::forward<ShapeType>(shape))
    {
    }


    /**
     * Constructor for creating a view with a user-defined data pointer.
     *
     * @tparam ShapeType
     *   Type of shape
     * @tparam StrideType
     *   Type of stride
     * @param ldata
     *   Offset data pointer (start of view)
     * @param shape
     *   Sizes for each dimension. 
     * @param strides
     *   Tensor strides
     */
    template <typename ShapeType, typename StrideType>    
    __MATX_INLINE__ tensor_impl_t(T *const ldata,
                    ShapeType &&shape,
                    StrideType &&strides)
        : ldata_(ldata), desc_(std::forward<ShapeType>(shape), std::forward<StrideType>(strides))
    {
    }


    /**
     * Constructor for creating a view with a descriptor and user-provided pointer
     *
     * If not reference counted, it is the caller's responsibility to manage the
     * data pointer, including allocation and freeing.
     *
     * @tparam DescriptorType
     *   Descriptor type
     * @param desc
     *   Tensor descriptor
     * @param ldata
     *   Data type
     */
    template <typename DescriptorType, std::enable_if_t<is_matx_descriptor_v<typename remove_cvref<DescriptorType>::type>, bool> = true>
    __MATX_INLINE__ tensor_impl_t(T *const ldata,
                    DescriptorType &&desc)
        : ldata_(ldata), desc_{std::forward<DescriptorType>(desc)}
    {
    }

    /**
     * Constructor for creating a view with only a descriptor
     *
     * Descriptor must confirm to all descriptor semantics. See documentation for details
     *
     * @tparam DescriptorType
     *   Descriptor type
     * @param desc
     *   Tensor descriptor
     */
    template <typename DescriptorType, std::enable_if_t<is_matx_descriptor_v<typename remove_cvref<DescriptorType>::type>, bool> = true>
    __MATX_INLINE__ tensor_impl_t(DescriptorType &&desc)
        : desc_{std::forward<DescriptorType>(desc)}
    {
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
    __MATX_INLINE__ auto Shape() const noexcept { return this->desc_.Shape(); }

    /**
     * Set the size of a dimension
     *
     * @return
     *    A shape of the data with the appropriate dimensions set
     */
    __MATX_INLINE__ auto Descriptor() const noexcept { return this->desc_; }     

    template <typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T* GetPointer(Is... indices) const noexcept
    {
      return ldata_ + GetValC<0, Is...>(std::make_tuple(indices...));
    }    

    /**
     * Check if a tensor is linear in memory for all elements in the view
     *
     * @return
     *    The size of the dimension
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ bool IsLinear() const noexcept
    {
      return desc_.IsLinear();
    }

    template <int I = 0, typename ...Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetVal([[maybe_unused]] std::tuple<Is...> tup)  {
      if constexpr (I < sizeof...(Is)) {
        return GetVal<I+1, Is...>(tup) + std::get<I>(tup)*this->desc_.Stride(I);
      }
      else {
        return 0;
      }
    }

    template <int I = 0, typename ...Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetValC([[maybe_unused]] const std::tuple<Is...> tup) const {
      if constexpr (I < sizeof...(Is)) {
        return GetValC<I+1, Is...>(tup) + std::get<I>(tup)*this->desc_.Stride(I);
      }
      else {
        return 0;
      }      
    }    

    /**
     * operator() getter
     *
     * @param indices
     *   Indices of tensor
     *
     * @returns value at given index
     *
     */
    template <int M = RANK, typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T &operator()(Is... indices) const noexcept
    {
      return *(ldata_ + GetValC<0, Is...>(std::make_tuple(indices...)));
    }

    /**
     * operator() getter
     *
     * @param indices
     *   Indices of tensor
     *
     * @returns value at given index
     *
     */
    template <int M = RANK, typename... Is, 
      std::enable_if_t<std::conjunction_v<std::is_integral<Is>...>, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T &operator()(Is... indices) noexcept
    {
      return *(ldata_ + GetVal<0, Is...>(std::make_tuple(indices...)));
    }    

    /**
     * operator() getter with an array index
     *
     * @returns value in tensor
     *
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T operator()(const std::array<index_t, RANK> &idx) const noexcept
    {
      return std::apply([&](auto &&...args) -> T {
          return this->operator()(args...);
        }, idx);      
    }  

    /**
     * operator() setter with an array index
     *
     * @returns value in tensor
     *
     */
    // template <int M = RANK, std::enable_if_t<M >= 1, bool> = true>
    // __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T &operator()(const std::array<index_t, RANK> &idx) noexcept
    // {
    //   if constexpr (RANK == 1) {
    //     return this->operator()(idx[0]);
    //   }
    //   else if constexpr (RANK == 2) {
    //     return this->operator()(idx[0], idx[1]);
    //   }
    //   else if constexpr (RANK == 3) {
    //     return this->operator()(idx[0], idx[1], idx[2]);
    //   }
    //   else {
    //     return this->operator()(idx[0], idx[1], idx[2], idx[3]);
    //   }
    // }      

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
    __MATX_INLINE__ stride_type __MATX_HOST__ __MATX_DEVICE__ TotalSize() const noexcept { return desc_.TotalSize(); }  

    /**
     * Get the size of a single dimension of the tensor
     *
     * @param dim
     *   Desired dimension
     *
     * @returns Number of elements in dimension
     *
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(uint32_t dim) const noexcept
    {
      return desc_.Size(dim);
    }

    /**
     * Get the stride of a single dimension of the tensor
     *
     * @param dim
     *   Desired dimension
     *
     * @returns Stride of dimension
     *
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Stride(uint32_t dim) const noexcept
    {
      return desc_.Stride(dim);
    }


    /**
     * Get the size of the last dimension
     *
     * @return
     *    The size of the dimension
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Lsize() const noexcept
    {
      return desc_.Size(Rank() - 1);
    }

    __MATX_INLINE__ __MATX_HOST__  auto Bytes() const noexcept
    {
      return TotalSize() * sizeof(*ldata_);
    }   

    /**
     * @brief Get data pointer
     * 
     * @return data pointer 
     */
    auto Data() const noexcept {
      return ldata_;
    } 


    /**
     * Print a value
     *
     * Type-agnostic function to print a value to stdout
     *
     * @param val
     */
    __MATX_INLINE__ __MATX_HOST__ void PrintVal(const T &val) const noexcept
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
    __MATX_HOST__ void InternalPrint(Args ...dims) const noexcept
    {
      MATX_STATIC_ASSERT(RANK == sizeof...(Args), "Number of dimensions to print must match tensor rank");
      MATX_STATIC_ASSERT(RANK <= 4, "Printing is only supported on tensors of rank 4 or lower currently");
      if constexpr (sizeof...(Args) == 0) {
        PrintVal(this->operator()());
        printf("\n");
      }
      else if constexpr (sizeof...(Args) == 1) {
        auto& k =detail:: pp_get<0>(dims...);
        for (typename Desc::shape_type _k = 0; _k < ((k == 0) ? this->Size(0) : k); _k++) {
          printf("%06lld: ", _k);
          PrintVal(this->operator()(_k));
          printf("\n");
        }
      }
      else if constexpr (sizeof...(Args) == 2) {
        auto& k = detail::pp_get<0>(dims...);
        auto& l = detail::pp_get<1>(dims...);
        for (typename Desc::shape_type _k = 0; _k < ((k == 0) ? this->Size(0) : k); _k++) {
          for (typename Desc::shape_type _l = 0; _l < ((l == 0) ? this->Size(1) : l); _l++) {
            if (_l == 0)
              printf("%06lld: ", _k);

            PrintVal(this->operator()(_k, _l));
          }
          printf("\n");
        }
      }
      else if constexpr (sizeof...(Args) == 3) {
        auto& j = detail::pp_get<0>(dims...);
        auto& k = detail::pp_get<1>(dims...);
        auto& l = detail::pp_get<2>(dims...);
        for (typename Desc::shape_type _j = 0; _j < ((j == 0) ? this->Size(0) : j); _j++) {
          printf("[%06lld,:,:]\n", _j);
          for (typename Desc::shape_type _k = 0; _k < ((k == 0) ? this->Size(1) : k); _k++) {
            for (typename Desc::shape_type _l = 0; _l < ((l == 0) ? this->Size(2) : l); _l++) {
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
        auto& i = detail::pp_get<0>(dims...);
        auto& j = detail::pp_get<1>(dims...);
        auto& k = detail::pp_get<2>(dims...);
        auto& l = detail::pp_get<3>(dims...); 
        for (typename Desc::shape_type _i = 0; _i < ((i == 0) ? this->Size(0) : i); _i++) {
          for (typename Desc::shape_type _j = 0; _j < ((j == 0) ? this->Size(1) : j); _j++) {
            printf("[%06lld,%06lld,:,:]\n", _i, _j);
            for (typename Desc::shape_type _k = 0; _k < ((k == 0) ? this->Size(2) : k); _k++) {
              for (typename Desc::shape_type _l = 0; _l < ((l == 0) ? this->Size(3) : l); _l++) {
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
    void Print(Args ...dims) const
    {
  #ifdef __CUDACC__    
      auto kind = GetPointerKind(ldata_);
      cudaDeviceSynchronize();
      if (HostPrintable(kind)) {
        InternalPrint(dims...);
      }
      else if (DevicePrintable(kind)) {
        if constexpr (PRINT_ON_DEVICE) {
          PrintKernel<<<1, 1>>>(*this, dims...);
        }
        else {
          auto tmpv = make_tensor<T>(desc_);
          typename base_type<decltype(tmpv)>::type tmpd = tmpv;

          cudaMemcpy(tmpd.Data(), Data(), tmpd.Bytes(),
                    cudaMemcpyDeviceToHost);
          tmpd.Print(dims...);
        }
      }
  #else
      InternalPrint(dims...);
  #endif    
    }    

    /**
     * @brief Set local data pointer
     * 
     * @param data
     *   Data pointer to set
     */
    void SetLocalData(T* data) {
      ldata_ = data;
    }

  protected:
    T *ldata_;
    Desc desc_;
};

}
};