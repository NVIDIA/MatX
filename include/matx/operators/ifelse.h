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
#include "matx/core/tensor_utils.h"
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
      typename detail::base_type_t<C1> cond_;
      typename detail::base_type_t<T1> op1_;
      typename detail::base_type_t<T2> op2_;    
      cuda::std::array<index_t, detail::matx_max(detail::get_rank<C1>(), detail::get_rank<T1>(), detail::get_rank<T2>())> size_;

    public:
      using value_type = void; ///< Scalar type for type extraction

      __MATX_INLINE__ std::string str() const { 
        return  "if(" + detail::get_type_str(cond_) + ") then {" +  detail::get_type_str(op1_) + "} else {" + detail::get_type_str(op2_) + "}"; 
      }

      /**
       * @brief Constructor for an IFELSE statement
       * 
       * @param cond Condition to perform the IF/ELSE on
       * @param op1 Operator if conditional branch is true
       * @param op2 Operator if conditional branch is false
       */
      __MATX_INLINE__ IFELSE(const C1 &cond, const T1 &op1, const T2 &op2) : 
                              cond_(cond), op1_(op1), op2_(op2)
      {
        static_assert((!is_tensor_view_v<T1> && !is_tensor_view_v<T2>),
            "Only operator emmitters are allowed in IFELSE. Tensor views "
            "are not allowed");
        constexpr int32_t rank0 = detail::get_rank<C1>();
        constexpr int32_t rank1 = detail::get_rank<T1>();
        constexpr int32_t rank2 = detail::get_rank<T2>();
        static_assert(rank0 == matxNoRank || rank0 == Rank());
        static_assert(rank1 == matxNoRank || rank1 == Rank());
        static_assert(rank2 == matxNoRank || rank2 == Rank());

        if constexpr (Rank() > 0)
        {
          for (int i = 0; i < Rank(); i++)
          {
            index_t size0 = detail::get_expanded_size<Rank()>(cond_, i);
            index_t size1 = detail::get_expanded_size<Rank()>(op1, i);
            index_t size2 = detail::get_expanded_size<Rank()>(op2, i);
            size_[i] = detail::matx_max(size0, size1, size2);
          }
        }

        ASSERT_COMPATIBLE_OP_SIZES(op1_);
        ASSERT_COMPATIBLE_OP_SIZES(op2_);
        ASSERT_COMPATIBLE_OP_SIZES(cond_);
      }

      /**
       * @brief Operator() for getting values of an if/else
       * 
       * @tparam Is Index types
       * @param indices Index values
       */
      template <detail::VecWidth InWidth, detail::VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto operator()(Is... indices) const {
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

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T1>()) {
          op1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<T2>()) {
          op2_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }        

        if constexpr (is_matx_op<C1>()) {
          cond_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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

        if constexpr (is_matx_op<C1>()) {
          cond_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
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
