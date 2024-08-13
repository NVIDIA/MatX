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
   * logically reshapes dimensions of a tensor/operator
   * TotalSize for reshape and input operator must match
   */
  namespace detail {
    template <int RANK, typename T, typename ShapeType>
      class ReshapeOp : public BaseOp<ReshapeOp<RANK, T, ShapeType>>
    {
      public: 
        using value_type = typename T::value_type;
	
      private:
        typename base_type<T>::type op_;
	      ShapeType sizes_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;
        using self_type = ReshapeOp<RANK, T, ShapeType>;

        __MATX_INLINE__ std::string str() const { return "reshape(" + op_.str() + ")"; }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return RANK;
        }

        static_assert(Rank() > 0, "ReshapeOp: Rank of operator must be greater than 0.");
        static_assert(T::Rank() > 0, "ReshapeOp: Rank of input operator must be greater than 0.");

        __MATX_INLINE__ ReshapeOp(const T &op, ShapeType &&s) : op_(op), sizes_(s) {

          index_t size = 1;

          for(int32_t i = 0; i < Rank(); i++) {
            size *= sizes_[i];
          }

          MATX_ASSERT_STR(size == TotalSize(op_), matxInvalidSize, "ReshapeOp: TotalSize of reshape must match");
        };

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          cuda::std::array<index_t, Rank()> inds{indices...};
          cuda::std::array<index_t, T::Rank()> ninds;

          index_t idx = 0;
          index_t stride = 1;

          // linearlize incoming index
#pragma unroll
          for(int i = Rank() - 1 ; i >= 0 ; i--) {
            idx += stride * inds[i];
            stride *= Size(i);
          }

          // extract new indices
#pragma unroll
          for(int i = T::Rank() - 1; i >= 0; i--) {
            ninds[i] = idx % op_.Size(i);
            idx /= op_.Size(i);
          }

          return cuda::std::apply(op_, ninds);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          cuda::std::array<index_t, Rank()> inds{indices...};
          cuda::std::array<index_t, T::Rank()> ninds;

          index_t idx = 0;
          index_t stride = 1;

          // linearlize incoming index
#pragma unroll
          for(int i = Rank() - 1 ; i >= 0 ; i--) {
            idx += stride * inds[i];
            stride *= Size(i);
          }

          // extract new indices
#pragma unroll
          for(int i = T::Rank() - 1; i >= 0; i--) {
            ninds[i] = idx % op_.Size(i);
            idx /= op_.Size(i);
          }

          return cuda::std::apply(op_, ninds);
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int32_t dim) const
        {
          return sizes_[dim];
        }

        template <typename S2, typename Executor>
        __MATX_INLINE__ void PreRun(S2 &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<S2>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename S2, typename Executor>
        __MATX_INLINE__ void PostRun(S2 &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<S2>(shape), std::forward<Executor>(ex));
          }
        }  

        ~ReshapeOp() = default;
        ReshapeOp(const ReshapeOp &rhs) = default;
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
   * @brief Operator to reshape a tensor or operator.
   *
   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam RANK the reshaped rank
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param s  the size of each reshaped dimension
   * @return reshaped operator
   */
  template <int RANK, typename T, typename ShapeType,
           std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
             __MATX_INLINE__ auto reshape(const T &op, ShapeType &&s)
  {
    return detail::ReshapeOp<RANK, T, ShapeType>(op, std::forward<ShapeType>(s));
  }  
  
    /**
   * @brief Operator to reshape a tensor or operator.
   *
   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam RANK the reshaped rank
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param sizes the size of each reshaped dimension
   * @return reshaped operator
   */
  template <int RANK, typename T>
    __MATX_INLINE__ auto reshape( const T &op, 
        const index_t (&sizes)[RANK]) {
      return reshape<RANK, T>(op, detail::to_array(sizes));
    }
} // end namespace matx
