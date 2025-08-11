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

        __MATX_INLINE__ std::string str() const { return "shift(" + op_.str() + ")"; }

        __MATX_INLINE__ ShiftOp(const T1 &op, T2 shift) : op_(op), shift_(shift)
        {
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
#ifdef __CUDA_ARCH__
        if constexpr (CapType::jit) {
          if ((threadIdx.x * CapType::ept) >= op.Size(op.Rank() - 1)) {
            return detail::GetJitSentinelValue<CapType, value_type>();
          }
        }
#endif
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
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] const InType& in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(
              my_cap,
            detail::get_operator_capability<Cap>(op_, in),
              detail::get_operator_capability<Cap>(shift_, in)
            );
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap,
            detail::get_operator_capability<Cap>(op_, in),
              detail::get_operator_capability<Cap>(shift_, in)
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
          if constexpr (is_matx_transform_op<R>()) {
            return mtie(*this, rhs);
          }
          else {
            return set(*this, rhs);
          }
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
