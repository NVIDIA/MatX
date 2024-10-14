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
   * Chain multiple operator statements
   *
   * Takes a variable list of operator statements to execute concurrently.
   * Chaining may improve performance over executing each operation separately.
   */
  namespace detail {
    template<class Op1, class Op2>
      class CommaOp : public BaseOp<CommaOp<Op1, Op2>>{
        public:
          __MATX_HOST__ __MATX_INLINE__  CommaOp(const Op1 &op1, const Op2 &op2) : op1_(op1), op2_(op2) {
            MATX_STATIC_ASSERT_STR(Op1::Rank() == Op2::Rank(), matxInvalidSize, 
                "Chained expressions using the comma operator must match in rank");
            if constexpr ( Rank() > 0) {
              for(int i = 0; i < Rank(); i++) {
                MATX_ASSERT_STR(op1_.Size(i) == op2_.Size(i), matxInvalidSize, "comma operators sizes of operators must match");
              }
            }
          }

	        __MATX_INLINE__ std::string str() const { return op1_.str() + ", " + op2_.str(); }

          template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
          auto __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ operator()(Is... indices) const {
            op1_(indices...);
            return op2_(indices...);
          }                       

          static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() noexcept
          {
            return Op2::Rank();
          }

          constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
          {
            return op2_.Size(dim);
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
      };  
  }

  /**
   * @brief Comma operator for evaluating multiple operators
   * 
   * @tparam T Left operator type
   * @tparam S Right operator type
   * @param l Left operator value
   * @param r Right operator value
   * @return Result of comma operator
   */
  template <typename T, typename S, std::enable_if_t<is_matx_op<T>() && is_matx_op<S>(), bool> = true>
    __MATX_INLINE__ __MATX_HOST__ auto operator,(const T &l, const S &r)
    {
      return detail::CommaOp(l, r);
    }
} // end namespace matx
