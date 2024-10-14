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
#include "matx/operators/scalar_ops.h"
#include "matx/operators/base_operator.h"

namespace matx
{

  namespace detail {
    template <typename Op1, typename Op2> 
    class IsCloseOp : public BaseOp<IsCloseOp<Op1, Op2>>
    {
      public:
        using matxop = bool;
        using value_type = int;
        using op_type = typename remove_cvref_t<Op2>::value_type;
        using inner_type = typename inner_op_type_t<op_type>::type;

        __MATX_INLINE__ std::string str() const { return "isclose()"; }

        __MATX_INLINE__ IsCloseOp(const Op1 &op1, const Op2 &op2, double rtol, double atol) : 
          op1_(op1), op2_(op2), rtol_(static_cast<inner_type>(rtol)), atol_(static_cast<inner_type>(atol)) 
        {
          static_assert(op1.Rank() == op2.Rank(), "Operator ranks must match in isclose()");
          ASSERT_COMPATIBLE_OP_SIZES(op1); 
          ASSERT_COMPATIBLE_OP_SIZES(op2);
        }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ int operator()([[maybe_unused]] Is... indices) const 
        {

          return static_cast<int>(detail::_internal_abs(op1_(indices...) - op2_(indices...)) <= 
              static_cast<inner_type>(atol_) + static_cast<inner_type>(rtol_) * detail::_internal_abs(op2_(indices...)));
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::matx_max(detail::get_rank<Op1>(), detail::get_rank<Op2>());
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          index_t size1 = detail::get_expanded_size<Rank()>(op1_, dim);
          index_t size2 = detail::get_expanded_size<Rank()>(op2_, dim);
          return detail::matx_max(size1,size2);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<Op1>()) {
            op1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<Op2>()) {
            op2_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }          
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<Op1>()) {
            op1_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<Op2>()) {
            op2_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }          

      private:
        typename detail::base_type_t<Op1> op1_;
        typename detail::base_type_t<Op2> op2_;
        inner_type rtol_;
        inner_type atol_;

    };
  }

  /**
   * @brief Returns an integer tensor where an element is 1 if:
   *    abs(op1 - op2) <= atol + rtol * abs(op2)
   * 
   * or 0 otherwise
   * 
   * @tparam Op1 First operator type
   * @tparam Op2 Second operator type
   * @param op1 First operator
   * @param op2 Second operator
   * @param rtol Relative tolerance
   * @param atol Absolute tolerance
   * @return IsClose operator
   */
  template <typename Op1, typename Op2>
  __MATX_INLINE__ auto isclose(const Op1 &op1, const Op2 &op2, double rtol = 1e-5, double atol = 1e-8) {
    return detail::IsCloseOp<Op1, Op2>(op1, op2, rtol, atol);
  }
} // end namespace matx
