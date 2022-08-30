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
   * Conditionally execute an operator, otherwise execute a different operator
   *
   * Compares two operators or views and conditionally executes the second
   * statement if the first is true, otherwise executes the third statement.
   * Values from an operator are executed individually, and the only requirement
   * for the conditional is the comparison operator must be defined for the
   * particular type. For example, operator< on two integers is okay, but the same
   * operator on two complex numbers will give a compiler error.
   *
   */
  template <typename C1, typename T1, typename T2>
    class IFELSE : public BaseOp<IFELSE<C1, T1, T2>>
  {
    private:
      typename detail::base_type<C1>::type cond_;
      typename detail::base_type<T1>::type op1_;
      typename detail::base_type<T2>::type op2_;    
      std::array<index_t, detail::matx_max(detail::get_rank<C1>(), detail::get_rank<T1>(), detail::get_rank<T2>())> size_;

    public:
      using scalar_type = void; ///< Scalar type for type extraction

      /**
       * @brief Constructor for an IFELSE statement
       * 
       * @param cond Condition to perform the IF/ELSE on
       * @param op1 Operator if conditional branch is true
       * @param op2 Operator if conditional branch is false
       */
      __MATX_INLINE__ IFELSE(C1 cond, T1 op1, T2 op2) : cond_(cond), op1_(op1), op2_(op2)
    {
      static_assert((!is_tensor_view_v<T1> && !is_tensor_view_v<T2>),
          "Only operator emmitters are allowed in IFELSE. Tensor views "
          "are not allowed");
      constexpr int32_t rank0 = detail::get_rank<C1>();
      constexpr int32_t rank1 = detail::get_rank<T1>();
      constexpr int32_t rank2 = detail::get_rank<T2>();
      static_assert(rank0 == -1 || rank0 == Rank());
      static_assert(rank1 == -1 || rank1 == Rank());
      static_assert(rank2 == -1 || rank2 == Rank());

      if constexpr (Rank() > 0)
      {
        for (int i = 0; i < Rank(); i++)
        {
          index_t size0 = detail::get_expanded_size<Rank()>(cond_, i);
          index_t size1 = detail::get_expanded_size<Rank()>(op1, i);
          index_t size2 = detail::get_expanded_size<Rank()>(op2, i);
          size_[i] = detail::matx_max(size0, size1, size2);
          MATX_ASSERT(size0 == 0 || size0 == Size(i), matxInvalidSize);
          MATX_ASSERT(size1 == 0 || size1 == Size(i), matxInvalidSize);
          MATX_ASSERT(size2 == 0 || size2 == Size(i), matxInvalidSize);          
        }
      }
    }

      /**
       * @brief Operator() for getting values of an if/else
       * 
       * @tparam Is Index types
       * @param indices Index values
       */
      template <typename... Is>
        auto __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ operator()(Is... indices) const {
          if (get_value(cond_, indices...)) {
            get_value(op1_, indices...);
          }
          else {
            get_value(op2_, indices...);
          }
        }      

      /**
       * @brief Rank of IF/ELSE operator
       * 
       * @return Rank
       */
      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return detail::matx_max(detail::get_rank<C1>(), detail::get_rank<T1>(), detail::get_rank<T2>());
      }

      /**
       * @brief Size of dimension of operator
       * 
       * @param dim Dimension to get size of
       * @return Size of dimension 
       */
      constexpr index_t __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const
      {
        return size_[dim];
      }
  };
} // end namespace matx
