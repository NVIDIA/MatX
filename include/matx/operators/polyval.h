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
   * Returns the polynomial evaluated at each point
   */
  namespace detail {
    template <typename Op, typename Coeffs>
    class PolyvalOp : public BaseOp<PolyvalOp<Op, Coeffs>>
    {
      private:
        Op op_;
        Coeffs coeffs_;

      public:
        using matxop = bool;
        using value_type = typename Op::value_type;

        __MATX_INLINE__ std::string str() const { return "polyval()"; }
        __MATX_INLINE__ PolyvalOp(const Op &op, const Coeffs &coeffs) : op_(op), coeffs_(coeffs) {
          MATX_STATIC_ASSERT_STR(Coeffs::Rank() == 1, matxInvalidDim, "Coefficient must be rank 1");
          MATX_STATIC_ASSERT_STR(Op::Rank() == 1, matxInvalidDim, "Input operator must be rank 1");
        };

        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ value_type operator()(index_t idx) const
        {
          // Horner's method for computing polynomial
          value_type ttl{coeffs_(0)};
          for(int i = 1; i < coeffs_.Size(0); i++) {
              ttl = ttl * op_(idx) + coeffs_(i);
          }

          return ttl;
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return 1;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          return op_.Size(0);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<Op>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<Op>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }


  /**
   * @brief Evaluate a polynomial
   * 
   * Currently only allows 1D input and coefficients
   * 
   * @tparam Op Type of input values to evaluate
   * @tparam Coeffs Type of coefficients
   * @param op Input values to evaluate
   * @param coeffs Coefficient values
   * @return polyval operator 
   */
  template <typename Op, typename Coeffs>
  __MATX_INLINE__ auto polyval(const Op &op, const Coeffs &coeffs) {
    return detail::PolyvalOp(op, coeffs);
  }
} // end namespace matx
