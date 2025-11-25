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

#ifdef MATX_EN_JIT
      struct JIT_Storage {
        typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;
        value_type pad_value_;
      };

      JIT_Storage ToJITStorage() const {
        return JIT_Storage{detail::to_jit_storage(op_), pad_value_};
      }

      __MATX_INLINE__ std::string get_jit_class_name() const {
        return std::format("JITPad_axis{}_before{}_after{}_mode{}", 
                          axis_, before_, after_, static_cast<int>(mode_));
      }

      __MATX_INLINE__ auto get_jit_op_str() const {
        std::string func_name = get_jit_class_name();
        cuda::std::array<index_t, RANK> out_dims_;
        for (int i = 0; i < RANK; ++i) {
          out_dims_[i] = Size(i);
        }
        
        return cuda::std::make_tuple(
          func_name,
          std::format("template <typename T> struct {} {{\n"
              "  using value_type = typename T::value_type;\n"
              "  using matxop = bool;\n"
              "  constexpr static int axis_ = {};\n"
              "  constexpr static index_t before_ = {};\n"
              "  constexpr static index_t after_ = {};\n"
              "  constexpr static PadMode mode_ = static_cast<PadMode>({});\n"
              "  constexpr static int Rank_ = {};\n"
              "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
              "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n"
              "  value_type pad_value_;\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n"
              "  {{\n"
              "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
              "      cuda::std::array<index_t, sizeof...(Is)> ind_array = {{{{indices...}}}};\n"
              "      index_t idx = ind_array[axis_];\n"
              "      index_t op_size = out_dims_[axis_] - before_ - after_;\n"
              "      if (idx < before_ || idx >= before_ + op_size) {{\n"
              "        if (mode_ == MATX_PAD_MODE_EDGE) {{\n"
              "          ind_array[axis_] = (idx < before_) ? 0 : (op_size - 1);\n"
              "        }} else {{\n"
              "          return value_type(pad_value_);\n"
              "        }}\n"
              "      }} else {{\n"
              "        ind_array[axis_] = idx - before_;\n"
              "      }}\n"
              "      return value_type(get_value<CapType>(op_, ind_array));\n"
              "    }} else {{\n"
              "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
              "    }}\n"
              "  }}\n"
              "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
              "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
              "}};\n",
              func_name, axis_, before_, after_, static_cast<int>(mode_), RANK, detail::array_to_string(out_dims_))
        );
      }
#endif

      __MATX_INLINE__ std::string str() const {
        return "pad(" + op_.str() + ")";
      }

      // Constructor for tuple/array-based padding
      template<typename PadSizeType>
      __MATX_INLINE__ PadOp(const T& op, int axis, const PadSizeType& pad_sizes, const value_type& pad_value, PadMode mode = MATX_PAD_MODE_CONSTANT) 
        : op_(op), axis_(axis), pad_value_(pad_value), mode_(mode)
      {
        MATX_LOG_TRACE("{} constructor: axis={}, mode={}", str(), axis, static_cast<int>(mode));
        static_assert(RANK > 0, "Cannot pad rank-0 tensors");
        MATX_ASSERT_STR(axis >= 0 && axis < RANK, matxInvalidDim, "pad axis must be >= 0 and less than the rank of the operator");
        MATX_ASSERT_STR(pad_sizes.size() == 2, matxInvalidParameter, "pad_sizes must contain exactly 2 elements [before, after]");
        
        before_ = pad_sizes[0];
        after_ = pad_sizes[1];
        
        MATX_ASSERT_STR(before_ >= 0, matxInvalidParameter, "pad before size must be non-negative");
        MATX_ASSERT_STR(after_ >= 0, matxInvalidParameter, "pad after size must be non-negative");
      }

      template <typename CapType, typename Op, typename... Is>
      static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(
          Op&& op, int axis, index_t before, const value_type& pad_value, PadMode mode, Is... indices) {
        if constexpr (CapType::ept == ElementsPerThread::ONE) {
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
          return value_type(get_value<CapType>(cuda::std::forward<Op>(op), ind_array));
        } else {
          return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
        }
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return get_impl<CapType>(cuda::std::as_const(op_), axis_, before_, pad_value_, mode_, is...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return this->operator()<detail::DefaultCapabilities>(is...);
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), axis_, before_, pad_value_, mode_, is...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return this->operator()<detail::DefaultCapabilities>(is...);
      }

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
        if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
          const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
          return std::format("{}<{}>", get_jit_class_name(), op_jit_name);
#else
          return "";
#endif
        }
        else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return combine_capabilities<Cap>(true, detail::get_operator_capability<Cap>(op_, in));
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
          const auto [key, value] = get_jit_op_str();
          if (in.find(key) == in.end()) {
            in[key] = value;
          }
          detail::get_operator_capability<Cap>(op_, in);
          return true;
#else
          return false;
#endif
        }
        else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
          return detail::get_operator_capability<Cap>(op_, in);
        }
        else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          // With padding, vectorized access would be problematic in cases where the padding is
          // not a multiple of the vector size.
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
        } else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
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
