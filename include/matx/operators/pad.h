////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#include <array>

#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"
#include "matx/operators/base_operator.h"

namespace matx
{
  /**
   * @brief Padding mode
   *
   * The padding mode to use for the pad operator. The default value is MATX_PAD_MODE_CONSTANT.
   */
  enum PadMode {
    MATX_PAD_MODE_CONSTANT, ///<Constant padding mode. All padding elements will be set to the user-provided pad_value.
    MATX_PAD_MODE_EDGE ///<Edge padding mode. All padding elements will be set to the edge values of the original operator.
  };

  namespace detail {
    /**
   * PadOp operator
   *
   * Class for padding operators along a single dimension. Sizes of the operator in the 
   * dimensions not being padded will be the same as the original operator. The size of the
   * operator in the dimension being padded will increase by the size of the padding.
   *
   */
   template <typename T>
      class PadOp : public BaseOp<PadOp<T>>
    {
      using self_type = PadOp<T>;

      static constexpr int RANK = T::Rank();

      public:
      using matxop = bool;

      // Scalar type of operation
      using value_type = typename T::value_type;

      __MATX_INLINE__ std::string str() const {
        return "pad(" + op_.str() + ")";
      }

      // Constructor for tuple/array-based padding
      template<typename PadSizeType>
      __MATX_INLINE__ PadOp(const T& op, int axis, const PadSizeType& pad_sizes, const value_type& pad_value, PadMode mode = MATX_PAD_MODE_CONSTANT) 
        : op_(op), axis_(axis), pad_value_(pad_value), mode_(mode)
      {
        static_assert(RANK > 0, "Cannot pad rank-0 tensors");
        MATX_ASSERT_STR(axis >= 0 && axis < RANK, matxInvalidDim, "pad axis must be >= 0 and less than the rank of the operator");
        MATX_ASSERT_STR(pad_sizes.size() == 2, matxInvalidParameter, "pad_sizes must contain exactly 2 elements [before, after]");
        
        before_ = pad_sizes[0];
        after_ = pad_sizes[1];
        
        MATX_ASSERT_STR(before_ >= 0, matxInvalidParameter, "pad before size must be non-negative");
        MATX_ASSERT_STR(after_ >= 0, matxInvalidParameter, "pad after size must be non-negative");
      }

      template <ElementsPerThread EPT, typename Op, typename... Is>
      static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(
          Op&& op, int axis, index_t before, const value_type& pad_value, PadMode mode, Is... indices) {
        if constexpr (EPT == ElementsPerThread::ONE) {
          cuda::std::array<index_t, sizeof...(Is)> ind_array = {{indices...}};
          index_t idx = ind_array[axis];
          index_t op_size = op.Size(axis);
          
          // Check if we're in the padding region
          if (idx < before || idx >= before + op_size) {
            if (mode == MATX_PAD_MODE_EDGE) {
              // Edge padding - replicate edge values
              ind_array[axis] = (idx < before) ? 0 : (op_size - 1);
            } else {
              // Default to constant padding
              return value_type(pad_value);
            }
          } else {
            // Original tensor region - adjust index to remove padding offset
            ind_array[axis] = idx - before;
          }
          return value_type(get_value<EPT>(cuda::std::forward<Op>(op), ind_array));
        } else {
          return Vector<value_type, static_cast<index_t>(EPT)>{};
        }
      }

      template <ElementsPerThread EPT, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return get_impl<EPT>(cuda::std::as_const(op_), axis_, before_, pad_value_, mode_, is...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return this->operator()<detail::ElementsPerThread::ONE>(is...);
      }

      template <ElementsPerThread EPT, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return get_impl<EPT>(cuda::std::forward<decltype(op_)>(op_), axis_, before_, pad_value_, mode_, is...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return this->operator()<detail::ElementsPerThread::ONE>(is...);
      }

      template <OperatorCapability Cap>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
        if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          // With padding, vectorized access would be problematic in cases where the padding is
          // not a multiple of the vector size.
          return ElementsPerThread::ONE;
        } else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_));
        }
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() noexcept
      {
        return RANK;
      }

      constexpr index_t __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const noexcept
      {
        if (dim == axis_) {
          return before_ + op_.Size(dim) + after_;
        } else {
          return op_.Size(dim);
        }
      }

      ~PadOp() = default;
      PadOp(const PadOp &rhs) = default;

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T>()) {
          op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T>()) {
          op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      private:
      typename detail::base_type_t<T> op_;
      int axis_;
      index_t before_;
      index_t after_;
      value_type pad_value_;
      PadMode mode_;
    }; // end class PadOp
  } // end namespace detail

  /**
   * @brief Pad an operator along a single dimension
   *
   * Creates a new operator that pads the input operator along the specified dimension
   * with a constant value or edge replication.
   *
   * @tparam T Input operator type
   * @param op Input operator to pad
   * @param axis Dimension along which to pad
   * @param pad_sizes std::array containing {before, after} padding sizes. This operator will add before elements before
   * the original operator and after elements after the original operator.
   * @param pad_value Value to use for padding (constant padding mode only)
   * @param mode Padding mode. Defaults to MATX_PAD_MODE_CONSTANT if not provided.
   * @return Padded operator
   */
  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ auto pad(const T& op, int axis, const std::array<index_t, 2>& pad_sizes, const typename T::value_type& pad_value, PadMode mode = MATX_PAD_MODE_CONSTANT)
  {
    return detail::PadOp<T>{op, axis, pad_sizes, pad_value, mode};
  }

  /**
   * @brief Pad an operator along a single dimension
   *
   * Creates a new operator that pads the input operator along the specified dimension
   * with a constant value or edge replication.
   *
   * @tparam T Input operator type
   * @param op Input operator to pad
   * @param axis Dimension along which to pad
   * @param pad_sizes C-style array containing {before, after} padding sizes. This operator will add before elements before
   * the original operator and after elements after the original operator.
   * @param pad_value Value to use for padding (constant padding mode only)
   * @param mode Padding mode. Defaults to MATX_PAD_MODE_CONSTANT if not provided.
   * @return Padded operator
   */
  template <typename T>
  __MATX_INLINE__ __MATX_HOST__ auto pad(const T& op, int axis, const index_t (&pad_sizes)[2], const typename T::value_type& pad_value, PadMode mode = MATX_PAD_MODE_CONSTANT)
  {
    return detail::PadOp<T>{op, axis, std::array<index_t, 2>{pad_sizes[0], pad_sizes[1]}, pad_value, mode};
  }
} // end namespace matx
