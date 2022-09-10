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
  namespace detail {
    template <int DIM, typename T1>
      class LCollapseOp : public BaseOp<LCollapseOp<DIM, T1>>
    {
      private:
        typename base_type<T1>::type op_;
        index_t size_;  // size of collapsed dim

      public:
        using matxop = bool;
        using scalar_type = typename T1::scalar_type;
        using shape_type = typename T1::shape_type;
        using matxoplvalue = bool;

        __MATX_INLINE__ LCollapseOp(const T1 &op) : op_(op)
      {
        static_assert(DIM < T1::Rank(),  "Collapse DIM must be less than Rank() of operator");
        static_assert(DIM > 0, "Collapse DIM must have a magnitude greater than 0");
        static_assert(T1::Rank() > 1, "Collapse must be called on operators with rank > 1");

        // comptue size of collapsed dimension
        size_ = 1;

        // Collapse left-most dims
#pragma unroll
        for(int i = 0 ; i <= DIM; i++) {
          size_ *= op_.Size(i);
        }
      }

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
          {
            // indices coming in
            std::array<index_t, Rank()> in{indices...};  // index coming in
            std::array<index_t, T1::Rank()> out;         // index going out

#pragma unroll
            for(int i = 1; i < Rank(); i++) {
              // copy all but first input index into out array
              out[DIM+i] = in[i];
            }

            // expand first input index into DIM indices
            auto ind = in[0];
#pragma unroll
            for(int i = 0; i <= DIM; i++) {
              int d = DIM - i;
              out[d] = ind % op_.Size(d);
              ind /= op_.Size(d);
            }

            return mapply(op_, out);
          }    
        
	template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto& operator()(Is... indices) 
          {
            // indices coming in
            std::array<index_t, Rank()> in{indices...};  // index coming in
            std::array<index_t, T1::Rank()> out;         // index going out

#pragma unroll
            for(int i = 1; i < Rank(); i++) {
              // copy all but first input index into out array
              out[DIM+i] = in[i];
            }

            // expand first input index into DIM indices
            auto ind = in[0];
#pragma unroll
            for(int i = 0; i <= DIM; i++) {
              int d = DIM - i;
              out[d] = ind % op_.Size(d);
              ind /= op_.Size(d);
            }

            return mapply(op_, out);
          }    

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T1::Rank() - DIM;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          if(dim == 0)  // if asking for the first dim, return collapsed size
            return size_;
          else // otherwise return the un-collapsed size from operator
            return op_.Size(DIM+dim);
        }
        
        template<typename R> __MATX_INLINE__ auto operator=(const R &rhs) { return set(*this, rhs); }
    };
  }
  /**
   * lcollapse operator
   *
   * The lcollapse operator takes a tensor and collapses the left most dimensions into a single dimension.
   *
   * @tparam DIM
   *   The number of dimensions to collapse
   * @tparam T1
   *   Operator type
   *
   * @param a
   *   The operator being collapsed
   *
   * @returns
   *   Operator with collapsed input
   */
  template <int DIM, typename T1>
    auto __MATX_INLINE__ lcollapse(const T1 &a)
    {
      return detail::LCollapseOp<DIM, T1>(a);
    }

  namespace detail {
    template <int DIM, typename T1>
      class RCollapseOp : public BaseOp<RCollapseOp<DIM, T1>>
    {
      private:
        typename base_type<T1>::type op_;
        index_t size_;  // size of collapsed dim

      public:
        using matxop = bool;
        using scalar_type = typename T1::scalar_type;
        using shape_type = typename T1::shape_type;
        using matxlvalue = bool;

        __MATX_INLINE__ RCollapseOp(const T1 &op) : op_(op)
      {
        static_assert(DIM < T1::Rank(),  "Collapse DIM must be less than Rank() of operator");
        static_assert(DIM > 0, "Collapse DIM must have a magnitude greater than 0");
        static_assert(T1::Rank() > 1, "Collapse must be called on operators with rank > 1");

        // comptue size of collapsed dimension
        size_ = 1;

        // Collapse right-most dims
#pragma unroll
        for(int i = 0 ; i <= DIM; i++) {
          size_ *= op_.Size(T1::Rank() - 1 - i);
        }
      }

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
          {
            // indices coming in
            std::array<index_t, Rank()> in{indices...};  // index coming in
            std::array<index_t, T1::Rank()> out;         // index going out

#pragma unroll
            for(int i = 0 ; i < Rank() - 1; i++) {
              // copy all but last index into out array
              out[i] = in[i];
            }

            // expand last index into DIM indices
            auto ind = in[Rank() - 1];
#pragma unroll
            for(int i = 0; i <= DIM; i++) {
              int d = T1::Rank() - 1 - i;
              out[d] = ind % op_.Size(d);
              ind /= op_.Size(d);
            }

            return mapply(op_, out);
          }    
        
	template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto& operator()(Is... indices)
          {
            // indices coming in
            std::array<index_t, Rank()> in{indices...};  // index coming in
            std::array<index_t, T1::Rank()> out;         // index going out

#pragma unroll
            for(int i = 0 ; i < Rank() - 1; i++) {
              // copy all but last index into out array
              out[i] = in[i];
            }

            // expand last index into DIM indices
            auto ind = in[Rank() - 1];
#pragma unroll
            for(int i = 0; i <= DIM; i++) {
              int d = T1::Rank() - 1 - i;
              out[d] = ind % op_.Size(d);
              ind /= op_.Size(d);
            }

            return mapply(op_, out);
          }    

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T1::Rank() - DIM;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          if(dim == Rank()-1)  // if asking for the last dim, return collapsed size
            return size_;
          else // otherwise return the un-collapsed size from operator
            return op_.Size(dim);
        }
        
        template<typename R> __MATX_INLINE__ auto operator=(const R &rhs) { return set(*this, rhs); }
    };
  }
  /**
   * rcollapse operator
   *
   * The rcollapse operator takes a tensor and collapses the right most dimensions into a single dimension.
   *
   * @tparam DIM
   *   The number of dimensions to collapse
   * @tparam T1
   *   Operator type
   *
   * @param a
   *   The parameter being collapsed
   *
   * @returns
   *   Operator with collapsed input
   */
  template <int DIM, typename T1>
    auto __MATX_INLINE__ rcollapse(const T1 &a)
    {
      return detail::RCollapseOp<DIM, T1>(a);
    }
} // end namespace matx
