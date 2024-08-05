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
   * Reverse the indexing of a View or operator on a single dimension
   *
   * Allows a view or operator to be indexed in reverse order. After applying the
   * operator, index 0 is the last element in the selected dimension, index 1 is
   * second to last, etc.
   *
   */
  namespace detail {
    template <int DIM, typename T1>
      class ReverseOp : public BaseOp<ReverseOp<DIM, T1>>
    {
      private:
        typename base_type<T1>::type op_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;
        using value_type = typename T1::value_type;
        using self_type = ReverseOp<DIM, T1>;

        __MATX_INLINE__ std::string str() const { return "reverse(" + op_.str() + ")"; }

        __MATX_INLINE__ ReverseOp(T1 op) : op_(op){};


        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          if constexpr (Rank() == 0) {
            return op_();
          } 
          else {
            auto tup = cuda::std::make_tuple(indices...);
            cuda::std::get<DIM>(tup) = Size(DIM) - cuda::std::get<DIM>(tup) - 1;
            return cuda::std::apply(op_, tup);
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) 
        {
          if constexpr (Rank() == 0) {
            return op_();
          } 
          else {
            auto tup = cuda::std::make_tuple(indices...);
            cuda::std::get<DIM>(tup) = Size(DIM) - cuda::std::get<DIM>(tup) - 1;
            return cuda::std::apply(op_, tup);
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return op_.Size(dim);
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

        ~ReverseOp() = default;
        ReverseOp(const ReverseOp &rhs) = default;
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
    };
  }

  /**
   * @brief Operator to logically reverse elements of an operator. Base case for variadic template.
   *
   * @tparam DIM Dimension to apply the reverse
   * @tparam Op Input operator/tensor type
   * @param t Input operator
   */
  template <int DIM, typename Op>
    auto __MATX_INLINE__ reverse(Op t)
    {
      return detail::ReverseOp<DIM, Op>(t);
    };

  /**
   * @brief Operator to logically reverse elements of an operator.
   *
   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam DIM Dimension to apply the reverse
   * @tparam DIMS... list of multiple dimensions to reverse along
   * @tparam Op Input operator/tensor type
   * @param t Input operator
   */
  template <int DIM1, int DIM2, int... DIMS, typename Op_type>
    auto __MATX_INLINE__ reverse(Op_type t)
    {
      // recursively call remap on remaining bits
      auto op = reverse<DIM2, DIMS...>(t);

      // construct remap op
      return detail::ReverseOp<DIM1, decltype(op)>(op);
    };

  /**
   * Flip the vertical axis of a tensor.
   */
  template <typename T1>
    auto __MATX_INLINE__ flipud(T1 t)
    {
      if constexpr (T1::Rank() == 1)
      {
        return detail::ReverseOp<T1::Rank() - 1 , T1>(t);
      }

      return detail::ReverseOp<T1::Rank() - 2, T1>(t);
    };

  /**
   * Flip the horizontal axis of a tensor.
   */
  template <typename T1>
    auto __MATX_INLINE__ fliplr(T1 t)
    {
      if constexpr (T1::Rank() == 1)
      {
        return detail::ReverseOp<T1::Rank() - 1, T1>(t);
      }

      return detail::ReverseOp<T1::Rank() - 1, T1>(t);
    };
} // end namespace matx
