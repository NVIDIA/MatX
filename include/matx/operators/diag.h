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
   * Returns elements on the diagonal
   *
   * Returns elements on the diagonal of a 2D tensor. Any dimensions above 2 will
   * be considered batch dimension and the size of those match the size of the
   * input operator. The last dimension is always sized to be the minimum of the
   * last two dimension of the input operator
   */
  namespace detail {
    template <typename T1, int RANK>
      class DiagOp : public BaseOp<DiagOp<T1, RANK>>
    {
      private:
        typename base_type<T1>::type op_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ std::string str() const { return "diag(" + op_.str() + ")"; }
 
        __MATX_INLINE__ DiagOp(T1 op) : op_(op) {}

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
          {
            static_assert(sizeof...(Is) == RANK - 1, "Diagonal operator must have one fewer index than rank of operator");
            static_assert(RANK > 1, "Cannot make get diagonals from 0D tensor");

            using tt = cuda::std::tuple_element_t<0, cuda::std::tuple<Is...>>;
            auto tup = cuda::std::make_tuple(indices..., static_cast<tt>(0));
            cuda::std::get<RANK - 1>(tup) = pp_get<RANK-2>(indices...);
            return cuda::std::apply(op_, tup);
          }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return RANK - 1;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          if (dim < RANK - 2) {
            return op_.Size(dim);
          }
          else {
            return cuda::std::min(op_.Size(RANK - 1), op_.Size(RANK-2));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }

  /**
   * Get the elements on the diagonal
   *
   * @param t
   *   Input operator
   */
  template <typename T1>
    auto __MATX_INLINE__ diag(T1 t) { return detail::DiagOp<T1, T1::Rank()>(t); } 
} // end namespace matx
