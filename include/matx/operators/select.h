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
   * Selects elements of an operator given a list of indices in another operator
   *
   */
  namespace detail {
    template <typename T, typename IdxType>
      class SelectOp : public BaseOp<SelectOp<T, IdxType>>
    {
      private:
        typename base_type<T>::type op_;
        typename base_type<IdxType>::type idx_;

      public:
        using matxop = bool;
        using value_type = typename T::value_type;
        static_assert(IdxType::Rank() == 1, "Rank of index operator must be 1");

        __MATX_INLINE__ std::string str() const { return "select(" + op_.str() + ")"; }

        __MATX_INLINE__ SelectOp(T op, IdxType idx) : op_(op), idx_(idx) {};  

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(index_t i) const 
        {
          auto arrs = detail::GetIdxFromAbs(op_, idx_(i));
          return op_(arrs);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(index_t i) 
        {
          auto arrs = detail::GetIdxFromAbs(op_, idx_(i));
          return op_(arrs);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<IdxType>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return idx_.Size(dim);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<IdxType>()) {
            idx_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<IdxType>()) {
            idx_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }          
        }        
    };
  }   

  /**
   * @brief Helper function to select values from a predicate operator
   * 
   * select() is used to index from a source operator using indices stored
   * in another operator. This is commonly used with the find_idx executor 
   * which returns the indices of values meeting a selection criteria.
   * 
   * @tparam T Input type
   * @tparam IdxType Operator with indices
   * @param t Input operator
   * @param idx Index tensor
   * @return Value in t from each location in idx
   */
  template <typename T, typename IdxType>
    auto __MATX_INLINE__ select(T t, IdxType idx)
    {
      return detail::SelectOp<T, IdxType>(t, idx);
    };   
} // end namespace matx
