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


#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx
{
  /**
   * Shifts the indexing of an operator to move the array forward or backward by the
   * shift amount.
   *
   * ShiftOp allows adjusting the relative view of a tensor to start at a
   * new offset. This may be useful to cut off part of a tensor that is
   * meaningless, while maintaining a 0-based offset from the new location. A
   * modulo is applied to the new index to allow wrapping around to the beginning.
   * Negative shifts are allowed, and have the effect of moving back from the end
   * of the tensor.
   */
  namespace detail {
    template <int DIM, typename T1, typename T2>
      class ShiftOp : public BaseOp<ShiftOp<DIM, T1, T2>>
    {
      public:
        using matxop = bool;
        using matxoplvalue = bool;
        using value_type = typename T1::value_type;
        using self_type = ShiftOp<DIM, T1, T2>;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> shift_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_), detail::to_jit_storage(shift_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITShift_dim{}", DIM);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, Rank()> out_dims_;
          for (int i = 0; i < Rank(); ++i) {
            out_dims_[i] = Size(i);
          }
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T1, typename T2> struct {} {{\n"
                "  using value_type = typename T1::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int DIM_ = {};\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> sizes_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op_;\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> shift_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      cuda::std::array idx{{indices...}};\n"
                "      index_t shift = -get_value<CapType>(shift_, indices...);\n"
                "      shift = (shift + idx[DIM_]) % sizes_[DIM_];\n"
                "      if (shift < 0) shift += sizes_[DIM_];\n"
                "      idx[DIM_] = shift;\n"
                "      return get_value<CapType>(op_, idx);\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<size_t>(CapType::ept)>();\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return sizes_[dim]; }}\n"
                "}};\n",
                func_name, DIM, Rank(), detail::array_to_string(out_dims_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "shift(" + op_.str() + ")"; }

        __MATX_INLINE__ ShiftOp(const T1 &op, T2 shift) : op_(op), shift_(shift)
        {
          MATX_LOG_TRACE("{} constructor: dim={}, rank={}", str(), DIM, Rank());
          static_assert(DIM < Rank(), "Dimension to shift must be less than rank of tensor");

          MATX_LOOP_UNROLL
          for (int i = 0; i < Rank(); i++) {
            index_t size1 = detail::get_expanded_size<Rank()>(op_, i);
            index_t size2 = detail::get_expanded_size<Rank()>(shift_, i);
            sizes_[i] = detail::matx_max(size1,size2);
          }

          MATX_ASSERT_COMPATIBLE_OP_SIZES(shift_);
          MATX_ASSERT_COMPATIBLE_OP_SIZES(op_);
        }

        template <typename CapType, typename Op, typename Sizes, typename ShiftType, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(
            Op&& op,
            const Sizes &sizes,
            ShiftType shiftin,
            Is... indices)
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            cuda::std::array idx{indices...};
            index_t shift = -get_value<CapType>(shiftin, indices...);

            shift = (shift + idx[DIM]) % sizes[DIM];

            if (shift < 0) {
              shift += sizes[DIM];
            }

            idx[DIM] = shift;

            return get_value<CapType>(cuda::std::forward<Op>(op), idx);
          } else {
            return Vector<value_type, static_cast<size_t>(CapType::ept)>();
          }
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<CapType>(cuda::std::as_const(op_), sizes_, shift_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), sizes_, shift_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
            const auto shift_jit_name = detail::get_operator_capability<Cap>(shift_, in);
            return std::format("{}<{},{}>", get_jit_class_name(), op_jit_name, shift_jit_name);
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
            const auto [key, value] = get_jit_op_str();
            if (in.find(key) == in.end()) {
              in[key] = value;
            }
            detail::get_operator_capability<Cap>(op_, in);
            detail::get_operator_capability<Cap>(shift_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(op_, in) + 
                   detail::get_operator_capability<Cap>(shift_, in);
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(
              my_cap,
            detail::get_operator_capability<Cap>(op_, in),
              detail::get_operator_capability<Cap>(shift_, in)
            );
          } else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(op_, in_copy), detail::get_operator_capability<Cap>(shift_, in_copy));
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap, detail::get_operator_capability<Cap>(op_, in), detail::get_operator_capability<Cap>(shift_, in)
            );
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::matx_max(detail::get_rank<T1>(), detail::get_rank<T2>());
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
          return sizes_[dim];
        }

        ~ShiftOp() = default;
        ShiftOp(const ShiftOp &rhs) = default;

        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R>
        __MATX_INLINE__ auto operator=(const R &rhs) {
          return set(*this, rhs);
        }

      private:
        typename detail::base_type_t<T1> op_;
        cuda::std::array<index_t, Rank()> sizes_;
        typename detail::base_type_t<T2> shift_;
    };
  }
  /**
   * Operator to shift dimension by a given amount
   *
   * @tparam DIM
   *   The dimension to be shifted
   *
   * @tparam OpT
   *   Type of operator or view
   *
   * @tparam ShiftOpT
   *   Type of the operator for the shift
   *
   * @param op
   *   Operator or view to shift
   *
   * @param s
   *   Operator which returns the shift
   *
   * @returns
   *   New operator with shifted indices
   */
  template <int DIM, typename OpT, typename ShiftOpT>
    auto __MATX_INLINE__ shift(const OpT &op, ShiftOpT s)
    {
      return detail::ShiftOp<DIM, OpT, ShiftOpT>(op, s);
    };


  /**
   * Operator to shift dimension by a given amount.
   * This version allows multiple dimensions.
   *
   * @tparam DIM
   *   The dimension to be shifted
   *
   * @tparam DIMS...
   *   The dimensions targeted for shifts
   *
   * @tparam OpT
   *   Type of operator or view
   *
   * @tparam ShiftsT
   *   Type of the shift operators
   *
   * @param op
   *   Operator or view to shift
   *
   * @param s
   *   Amount to shift forward
   *
   * @param shifts
   *    list of shift amounts
   * @returns
   *   New operator with shifted indices
   */
  template <int DIM, int... DIMS,  typename OpT, typename ShiftT,  typename... ShiftsT>
    auto __MATX_INLINE__ shift(const OpT &op, ShiftT s, ShiftsT... shifts)
    {
      static_assert(sizeof...(DIMS) == sizeof...(shifts), "shift: number of DIMs must match number of shifts");

      // recursively call shift  on remaining bits
      auto rop = shift<DIMS...>(op, shifts...);

      // construct shift op
      return detail::ShiftOp<DIM, decltype(rop), decltype(s)>(rop, s);
    };

} // end namespace matx
