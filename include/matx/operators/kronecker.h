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
   * Kronecker tensor product
   *
   * Performs a Kronecker tensor product on two matrices. For input tensors A
   * (MxN) and B (PxQ), A is repeated and multiplied by each element in B to
   * create a new matrix of size M*P x N*Q.
   */
  namespace detail {
    template <typename T1, typename T2, int DIM>
      class KronOp : public BaseOp<KronOp<T1, T2, DIM>>
    {
      private:
        typename base_type<T1>::type op1_;
        typename base_type<T2>::type op2_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

      __MATX_INLINE__ std::string str() const { return "kron(" + op1_.str() + "," + op2_.str() + ")"; }

        __MATX_INLINE__ KronOp(T1 op1, T2 op2) : op1_(op1), op2_(op2)
      {
        static_assert(RankGTE(Rank(), 2), "Kronecker product must be used on tensors with rank 2 or higher");
      }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          auto tup1 = cuda::std::make_tuple(indices...);
          auto tup2 = cuda::std::make_tuple(indices...);
          cuda::std::get<Rank() - 2>(tup2) = pp_get<Rank() - 2>(indices...) % op2_.Size(Rank() - 2);
          cuda::std::get<Rank() - 1>(tup2) = pp_get<Rank() - 1>(indices...) % op2_.Size(Rank() - 1);

          cuda::std::get<Rank() - 2>(tup1) = pp_get<Rank() - 2>(indices...) / op2_.Size(Rank() - 2);
          cuda::std::get<Rank() - 1>(tup1) = pp_get<Rank() - 1>(indices...) / op2_.Size(Rank() - 1);

          return cuda::std::apply(op2_, tup2) * cuda::std::apply(op1_, tup1);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          auto tup1 = cuda::std::make_tuple(indices...);
          auto tup2 = cuda::std::make_tuple(indices...);
          cuda::std::get<Rank() - 2>(tup2) = pp_get<Rank() - 2>(indices...) % op2_.Size(Rank() - 2);
          cuda::std::get<Rank() - 1>(tup2) = pp_get<Rank() - 1>(indices...) % op2_.Size(Rank() - 1);

          cuda::std::get<Rank() - 2>(tup1) = pp_get<Rank() - 2>(indices...) / op2_.Size(Rank() - 2);
          cuda::std::get<Rank() - 1>(tup1) = pp_get<Rank() - 1>(indices...) / op2_.Size(Rank() - 1);

          return cuda::std::apply(op2_, tup2) * cuda::std::apply(op1_, tup1);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            op2_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }          
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op1_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            op2_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          } 
        }          

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return op1_.Size(dim) * op2_.Size(dim);
        }
    };
  }

  /**
   * Kronecker tensor product
   *
   * The Kronecker tensor product is formed by the matrix b by ever element in the
   * matrix a. The resulting matrix has the number of rows and columns equal to
   * the product of the rows and columns of matrices a and b, respectively.
   *
   * @tparam T1
   *   Type of first input
   * @tparam T2
   *   Type of second input
   * @param a
   *   Operator or view for first input
   * @param b
   *   Operator or view for second input
   *
   * @returns
   *   New operator of the kronecker product
   */
  template <typename T1, typename T2>
    auto __MATX_INLINE__ kron(T1 a, T2 b)
    {
      return detail::KronOp<T1, T2, T1::Rank()>(a, b);
    };
} // end namespace matx
