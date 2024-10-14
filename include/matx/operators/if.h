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
   * Conditionally execute an operator
   *
   * Compares two operators or views and conditionally executes the second
   * statement if the first is true. Values from an operator are executed
   * individually, and the only requirement for the conditional is the comparison
   * operator must be defined for the particular type. For example, operator< on
   * two integers is okay, but the same operator on two complex numbers will give
   * a compiler error.
   *
   */
  template <typename T1, typename T2>
    class IFOP : public BaseOp<IFOP<T1, T2>>
  {
    private:
      typename detail::base_type_t<T1> cond_;
      typename detail::base_type_t<T2> op_;
      cuda::std::array<index_t, detail::matx_max(detail::get_rank<T1>(), detail::get_rank<T2>())> size_;

    public:
      using value_type = void; ///< Scalar type for type extraction

      __MATX_INLINE__ std::string str() const { return  "if(" + cond_.str() + ") then {" +  op_.str() + "}"; }
      /**
       * @brief Constructor for an IF statement
       * 
       * @param cond Condition to perform the IF/ELSE on
       * @param op Operator if conditional branch is true
       */    
      __MATX_INLINE__ IFOP(const T1 &cond, const T2 &op) : cond_(cond), op_(op)
      {
        static_assert((!is_tensor_view_v<T2>),
            "Only operator emmitters are allowed in IF. Tensor views are "
            "not allowed");
        constexpr index_t rank1 = detail::get_rank<T1>();
        constexpr index_t rank2 = detail::get_rank<T2>();
        static_assert(rank1 == -1 || rank1 == Rank());
        static_assert(rank2 == -1 || rank2 == Rank());

        if constexpr (Rank() > 0)
        {
          for (int i = 0; i < Rank(); i++)
          {
            index_t size1 = detail::get_expanded_size<Rank()>(cond_, i);
            index_t size2 = detail::get_expanded_size<Rank()>(op_, i);
            size_[i] = detail::matx_max(size1, size2);          
          }
        
          ASSERT_COMPATIBLE_OP_SIZES(op_);
          ASSERT_COMPATIBLE_OP_SIZES(cond_);
        }
      }

      /**
       * @brief Operator() for getting values of an if operator
       * 
       * @tparam Is Index types
       * @param indices Index values
       */
      template <detail::VecWidth InWidth, detail::VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto operator()(Is... indices) const {
        if (get_value(cond_, indices...)) {
          get_value(op_, indices...);
        }
      }   

      /**
       * @brief Rank of IF operator
       * 
       * @return Rank
       */    
      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return detail::matx_max(detail::get_rank<T1>(), detail::get_rank<T2>());
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


      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T2>()) {
          op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<T1>()) {
          cond_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }        
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T2>()) {
          op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }     

        if constexpr (is_matx_op<T1>()) {
          cond_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }      
  };

  /**
   *
   * @brief Compares two operators or views and conditionally executes the second
   * statement if the first is true. Values from an operator are executed
   * individually, and the only requirement for the conditional is the comparison
   * operator must be defined for the particular type. For example, operator< on
   * two integers is okay, but the same operator on two complex numbers will give
   * a compiler error.
   * 
   * @param t1 op1
   *
   * @param t2 op2
   */
  template <typename T1, typename T2>
    auto IF(T1 t1, T2 t2) {
      return IFOP<T1,T2>(t1,t2);
    }
} // end namespace matx
