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

#include <cassert>
#include <type_traits>
#include <cuda/std/functional>
#include "matx/core/error.h"
#include "matx/core/defines.h"
#include "matx/core/tensor_desc.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"
#include "matx/operators/set.h"
#include "matx/core/vector.h"
#include "iterator.h"
#include "matx/core/make_tensor.h"

namespace matx {

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
template <typename T, int RANK, typename Desc = DefaultDescriptor<RANK>>
class tensor_impl_t {
  public:
    // Type specifier for reflection on class
    using type = T; // TODO is this necessary
    using value_type = T;
    using tensor_view = bool;
    using tensor_impl = bool;
    using desc_type = Desc;
    using shape_type = typename Desc::shape_type;
    using stride_type = typename Desc::stride_type;
    using matxoplvalue = bool;
    using matx_width = bool; ///< Signal we can do vector types from this operator
    using self_type = tensor_impl_t<T, RANK, Desc>;

    // Type specifier for signaling this is a matx operation
    using matxop = bool;

    __MATX_INLINE__ tensor_impl_t(const tensor_impl_t &) = default;
    __MATX_INLINE__ tensor_impl_t(tensor_impl_t &&) = default;
    __MATX_INLINE__ tensor_impl_t& operator=(tensor_impl_t &&) = default;


    __MATX_INLINE__ ~tensor_impl_t() = default;


    const std::string str() const {
      return std::string("T") + std::to_string(RANK) + "_" + to_short_str<T>();
    }

    /** Swaps two raw_pointer_buffers
     *
     * Swaps members of two raw_pointer_buffers
     *
     * @param lhs
     *   Left argument
     * @param rhs
     *   Right argument
     */
    friend void swap(tensor_impl_t<T, RANK, Desc> &lhs, tensor_impl_t<T, RANK, Desc> &rhs) noexcept
    {
      using std::swap;

      swap(lhs.ldata_, rhs.ldata_);
      swap(lhs.desc_, rhs.desc_);
    }

    /**
     * Constructor for a rank-0 tensor (scalar).
     */
    tensor_impl_t() {

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
              std::enable_if_t<!is_tensor_view_v<remove_cvref_t<ShapeType>> && !is_matx_descriptor_v<remove_cvref_t<ShapeType>>, bool> = true>
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
// gcc 14.1 incorrectly reports desc as uninitialized in some contexts
IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
    template <typename DescriptorType, std::enable_if_t<is_matx_descriptor_v<typename remove_cvref<DescriptorType>::type>, bool> = true>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ tensor_impl_t(T *const ldata,  
                    DescriptorType &&desc)                
        : ldata_(ldata), desc_{std::forward<DescriptorType>(desc)}
    {
    }
IGNORE_WARNING_POP_GCC

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

    __MATX_HOST__ void Shallow(const self_type &rhs) noexcept
    {
      ldata_ = rhs.ldata_;
      desc_ = rhs.desc_;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto copy(const tensor_impl_t<T, RANK> &op)
    {
      ldata_ = op.ldata_;
      desc_ = op.desc_;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
      const typename detail::base_type_t<T2> &op_base = op;
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
     * Get the strides the tensor from the underlying data
     *
     * @return
     *    A shape of the data with the appropriate strides set
     */
    __MATX_INLINE__ auto Strides() const noexcept { return this->desc_.Strides(); }

    template <int N = RANK, typename StrideType>
    __MATX_INLINE__ auto SliceImpl([[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &firsts,
                              [[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &ends,
                              [[maybe_unused]] StrideType strides) const
    {
      static_assert(N <= RANK && RANK > 0, "Must slice to a rank the same or less than current rank.");

      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      cuda::std::array<typename Desc::shape_type, N> n = {};
      cuda::std::array<typename Desc::stride_type, N> s = {};

      T *data = ldata_;
      int d = 0;

      [[maybe_unused]] int end_count = 0;
      for (int i = 0; i < RANK; i++) {
        if (ends[i] == matxDropDim) {
          end_count++;
        }
      }

      MATX_ASSERT_STR(((RANK - end_count) == N), matxInvalidSize,
              "Number of matxDropDim specifiers must match the output rank");

  #pragma unroll
      for (int i = 0; i < RANK; i++) {
        typename Desc::shape_type first = firsts[i] < 0 ? this->Size(i) + firsts[i] : firsts[i];
        typename Desc::shape_type end = ends[i]   < 0 ? this->Size(i) + ends[i]   : ends[i];

        MATX_ASSERT_STR((end > matxIdxSentinel) || (end <= this->Size(i)), matxInvalidDim,
          "Slice end index out of range of operator");

        MATX_ASSERT_STR(first < end, matxInvalidSize, "Slice must be at least one element long");

        [[maybe_unused]] typename Desc::stride_type stride_mult;
        
        if constexpr (std::is_same_v<StrideType, detail::NoStride>) {
          stride_mult = 1;
        }
        else {
          stride_mult = (strides[i] == matxKeepStride) ? 1 : strides[i];
        }

        MATX_ASSERT_STR(first < end, matxInvalidParameter,
                        "Starting slice must be less than end slice");
        MATX_ASSERT_STR(first < this->desc_.Size(i), matxInvalidParameter,
                        "Requested slice start index out of bounds");

        // offset by first
        data += first * Stride(i);

        if constexpr (N > 0) {
          if (end != matxDropDim) {
            MATX_ASSERT_STR(end != matxKeepDim, matxInvalidParameter, "matxKeepDim only valid for clone(), not slice()");
            if (end == matxEnd) {
              n[d] = this->Size(i) - first;
            }
            else {
              n[d] = end - first;
            }

            // New length is shorter if we have a non-1 stride
            n[d] = static_cast<typename Desc::shape_type>(std::ceil(
                static_cast<double>(n[d]) / static_cast<double>(stride_mult)));

            s[d] = Stride(i) * stride_mult;
            d++;
          }
        }
      }

      MATX_ASSERT_STR(d == N, matxInvalidDim,
                      "Number of indices must match the target rank to slice to");

      return cuda::std::make_tuple(tensor_desc_t<decltype(n), decltype(s), N>{std::move(n), std::move(s)}, data);
    }


    template <int N = RANK, typename StrideType>
    __MATX_INLINE__ auto Slice([[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &firsts,
                               [[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &ends,
                               [[maybe_unused]] StrideType strides) const
    {
      auto [new_desc, data] = this->SliceImpl<N, StrideType>(firsts, ends, strides);
      return tensor_impl_t<T, N, decltype(new_desc)>{data, std::move(new_desc)};
    }

    template <int N = RANK>
    __MATX_INLINE__ auto Slice(const cuda::std::array<typename Desc::shape_type, RANK> &firsts,
                              const cuda::std::array<typename Desc::shape_type, RANK> &ends) const
    {
      static_assert(N <= RANK && RANK > 0, "Must slice to a rank the same or less than current rank.");

      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      return Slice<N, detail::NoStride>(firsts, ends, detail::NoStride{});
    }  


    template <int N>
    __MATX_INLINE__ auto CloneImpl(const cuda::std::array<index_t, N> &clones) const
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      cuda::std::array<index_t, N> n;
      cuda::std::array<typename Desc::stride_type, N> s;

      int d = 0;

      #pragma unroll
      for (int i = 0; i < N; i++) {
        index_t size = clones[i];

        if (size == matxKeepDim) {
          n[i] = this->desc_.Size(d);
          if constexpr (RANK == 0) {
            s[i] = 1;
          }
          else {
            s[i] = this->desc_.Stride(d);
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
      tensor_desc_t<decltype(n), decltype(s), N> new_desc{std::move(n), std::move(s)};
      return new_desc;
    }
  

    template <int N>
    __MATX_INLINE__ auto Clone(const cuda::std::array<index_t, N> &clones) const
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      auto new_desc = CloneImpl<N>(clones);

      return tensor_impl_t<T, N, decltype(new_desc)>{this->ldata_, std::move(new_desc)};
    }

    __MATX_INLINE__ auto PermuteImpl(const cuda::std::array<int32_t, RANK> &dims) const
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      static_assert(RANK >= 1, "Only tensors of rank 1 and higher can be permuted.");
      cuda::std::array<shape_type, RANK> n;
      cuda::std::array<stride_type, RANK> s;
      [[maybe_unused]] bool done[RANK] = {0};

  #pragma unroll
      for (int i = 0; i < RANK; i++) {
        int d = dims[i];
        MATX_ASSERT_STR(d < RANK, matxInvalidDim,
                        "Index to permute is larger than tensor rank");
        MATX_ASSERT_STR(done[d] == false, matxInvalidParameter,
                        "Cannot list the same dimension to permute twice");
        done[d] = true;
        n[i] = this->Size(d);
        s[i] = this->Stride(d);
      }

      return Desc{std::move(n), std::move(s)};
    }

    __MATX_INLINE__ auto Permute(const cuda::std::array<int32_t, RANK> &dims) const
    {
      auto new_desc = PermuteImpl(dims);
      return tensor_impl_t<T, RANK, decltype(new_desc)>{this->ldata_, std::move(new_desc)};
    }

    template <int N>
    __MATX_INLINE__ auto
    OverlapViewImpl(const cuda::std::array<typename Desc::shape_type, N> &windows,
                const cuda::std::array<typename Desc::stride_type, N> &strides) const
    {
      static_assert(RANK == 1, "Overlapped views only supported on 1D tensors.");

      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      cuda::std::array<typename Desc::shape_type, RANK+1> n;
      cuda::std::array<typename Desc::stride_type, RANK+1> s;

      // This only works for 1D tensors going to 2D at the moment. Generalize to
      // higher dims later
      typename Desc::stride_type window_size = *(windows.begin());
      typename Desc::stride_type stride_size = *(strides.begin());

      MATX_ASSERT(stride_size < window_size, matxInvalidSize);
      MATX_ASSERT(stride_size > 0, matxInvalidSize);

      // Figure out the actual length of the signal we can use. It might be
      // shorter than the original tensor if the window/stride doesn't line up
      // properly to make a rectangular matrix.
      typename Desc::shape_type adj_el = this->desc_.Size(0) - window_size;
      while ((adj_el % stride_size) != 0) {
        adj_el--;
      }

      n[1] = window_size;
      s[1] = 1;
      n[0] = adj_el / stride_size + 1;
      s[0] = stride_size;

      tensor_desc_t<decltype(n), decltype(s), RANK+1> new_desc{std::move(n), std::move(s)};
      return new_desc;
    }

    template <int N>
    __MATX_INLINE__ auto
    OverlapView(const cuda::std::array<typename Desc::shape_type, N> &windows,
                const cuda::std::array<typename Desc::stride_type, N> &strides) const {
      auto new_desc = OverlapViewImpl<N>(windows, strides);
      return tensor_impl_t<T, RANK + 1, decltype(new_desc)>{this->ldata_, std::move(new_desc)};
    }

    template <typename O>
    __MATX_INLINE__ bool isSameView(const O &o) {
      if constexpr (is_tensor_view_v<O> && RANK == O::Rank()) {
        return Data() == o.Data() &&
          this->Shape() == o.Shape() &&
          this->Strides() == o.Strides();
      } else {
        return false;
      }
    }    

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
      return ldata_ + GetValC<0, Is...>(cuda::std::make_tuple(indices...));
    }

    /**
     * Check if a tensor is linear in memory for all elements in the view
     *
     * @return
     *    The size of the dimension
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr bool IsContiguous() const noexcept
    {
      return desc_.IsContiguous();
    }

    template <int I = 0, typename ...Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetVal([[maybe_unused]] cuda::std::tuple<Is...> tup)  {
      if constexpr (I < sizeof...(Is)) {
IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
        return GetVal<I+1, Is...>(tup) + cuda::std::get<I>(tup)*this->desc_.Stride(I);
IGNORE_WARNING_POP_GCC
      }
      else {
        return 0;
      }
    }

    template <int I = 0, typename ...Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetValC([[maybe_unused]] const cuda::std::tuple<Is...> tup) const {
      if constexpr (I < sizeof...(Is)) {
IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
        return GetValC<I+1, Is...>(tup) + cuda::std::get<I>(tup)*this->desc_.Stride(I);
IGNORE_WARNING_POP_GCC
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
    template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const noexcept
    {
      static_assert(sizeof...(Is) == RANK, "Number of indices of operator() must match rank of tensor");
#ifndef NDEBUG
      assert(ldata_ != nullptr);
#endif
      if constexpr (OutWidth != VecWidth::SCALAR) {
        using vec_type = Vector<T, static_cast<size_t>(InWidth)>;
        return *(reinterpret_cast<vec_type*>(ldata_) + GetValC<0, Is...>(cuda::std::make_tuple(indices...)));
      }
      else {
        return *(ldata_ + GetValC<0, Is...>(cuda::std::make_tuple(indices...)));
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
    template <VecWidth InWidth, VecWidth OutWidth, typename... Is,
      std::enable_if_t<std::conjunction_v<std::is_integral<Is>...>, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) noexcept
    {
      static_assert(sizeof...(Is) == RANK, "Number of indices of operator() must match rank of tensor");
#ifndef NDEBUG
      assert(ldata_ != nullptr);
#endif
      if constexpr (OutWidth != VecWidth::SCALAR) {
        using vec_type = Vector<T, static_cast<size_t>(InWidth)>;
        return *(reinterpret_cast<vec_type*>(ldata_) + GetVal<0, Is...>(cuda::std::make_tuple(indices...)));
      }
      else {
        return *(ldata_ + GetVal<0, Is...>(cuda::std::make_tuple(indices...)));
      }
    }

    // These operator() are called from a user-facing context to set/get individual values
    /**
     * operator() getter
     *
     * @param indices
     *   Indices of tensor
     *
     * @returns value at given index
     *
     */
    template <typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const noexcept
    {
      static_assert(sizeof...(Is) == RANK, "Number of indices of operator() must match rank of tensor");
#ifndef NDEBUG
      assert(ldata_ != nullptr);
#endif
      return operator()<VecWidth::SCALAR, VecWidth::SCALAR, Is...>(indices...);
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
    template <typename... Is,
      std::enable_if_t<std::conjunction_v<std::is_integral<Is>...>, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) noexcept
    {
      static_assert(sizeof...(Is) == RANK, "Number of indices of operator() must match rank of tensor");
#ifndef NDEBUG
      assert(ldata_ != nullptr);
#endif
      return operator()<VecWidth::SCALAR, VecWidth::SCALAR, Is...>(indices...);
    }    

     template <VecWidth InWidth, VecWidth OutWidth>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) const noexcept
    {
      return cuda::std::apply([&](auto &&...args) -> T {
          return this->operator()<InWidth, OutWidth>(args...);
        }, idx);
    }

    /**
     * operator() getter with an array index
     *
     * @returns value in tensor
     *
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) const noexcept
    {
      return cuda::std::apply([&](auto &&...args) -> T {
          return this->operator()<VecWidth::SCALAR, VecWidth::SCALAR>(args...);
        }, idx);
    }

    /**
     * operator() getter with an array index
     *
     * @returns value in tensor
     *
     */
    template <VecWidth InWidth, VecWidth OutWidth>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) noexcept
    {
      return cuda::std::apply([&](auto &&...args) -> T& {
          return this->operator()<InWidth, OutWidth>(args...);
        }, idx);
    }

    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) noexcept
    {
      return cuda::std::apply([&](auto &&...args) -> T& {
          return this->operator()<VecWidth::SCALAR, VecWidth::SCALAR>(args...);
        }, idx);
    }    


    VecWidth GetMaxWidth() const {
      constexpr int MAX_VEC_WIDTH = 16; // 16B loads and stores
      //if (IsContiguous()) {
        uint32_t width = 4;
        while (width > 1) {
printf("ret width=%u size=%zu (Lsize() %% width)==0=%d (Stride(Rank() - 1) == 1)=%d  (sizeof(T) * width)<= MAX_VEC_WIDTH=%d     (reinterpret_cast<uintptr_t>(ldata_) %% (sizeof(T) * width))== 0  =%d\n", 
          width, sizeof(T), (Lsize() % width)==0, (Stride(Rank() - 1) == 1), (sizeof(T) * width)<= MAX_VEC_WIDTH, (reinterpret_cast<uintptr_t>(ldata_) % (sizeof(T) * width))== 0);          
          if (((Lsize() % width) == 0) &&
            (Stride(Rank() - 1) == 1) &&
            ((sizeof(T) * width) <= MAX_VEC_WIDTH) &&
            (reinterpret_cast<uintptr_t>(ldata_) % (sizeof(T) * width)) == 0) {

            if constexpr (Rank() > 1) {
              if (((Stride(Rank() - 2) % width) == 0)) {
                break;
              }
            }
            else {
              break;
            }
          }

          width /= 2;
          
        }

        return static_cast<VecWidth>(width);
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
    constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto TotalSize() const noexcept { return desc_.TotalSize(); }

    /**
     * Get the size of a single dimension of the tensor
     *
     * @param dim
     *   Desired dimension
     *
     * @returns Number of elements in dimension
     *
     */
    constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
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
    constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Lsize() const noexcept
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
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  auto Data() const noexcept {
      return ldata_;
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

    template <typename ShapeType, typename Executor>
    __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
    {
    }

    template <typename ShapeType, typename Executor>
    __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
    {
    }


  protected:
    T *ldata_;
    Desc desc_;
};

}
};
