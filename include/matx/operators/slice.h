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
   * Slices elements from an operator/tensor.
   */
  namespace detail {
    template <int DIM, typename T>
      class SliceOp : public BaseOp<SliceOp<DIM, T>>
    {
      public: 
        using scalar_type = typename T::scalar_type;
        using shape_type = typename T::shape_type; 

      private:
        typename base_type<T>::type op_;
        std::array<shape_type, DIM> sizes_;
        std::array<int32_t, DIM> dims_;
        std::array<shape_type, T::Rank()> starts_;
        std::array<shape_type, T::Rank()> strides_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        static_assert(T::Rank()>0, "SliceOp: Rank of operator must be greater than 0.");
        static_assert(DIM<=T::Rank(), "SliceOp: DIM must be less than or equal to operator rank.");

        __MATX_INLINE__ SliceOp(T op, const shape_type (&starts)[T::Rank()], const shape_type (&ends)[T::Rank()], const shape_type (&strides)[T::Rank()]) : op_(op) {
          int32_t d = 0;
          for(int32_t i = 0; i < T::Rank(); i++) {
            shape_type start = starts[i];
            shape_type end = ends[i];

            starts_[i] = start;
            strides_[i] = strides[i];

            // compute dims and sizes
            if(end != matxDropDim) {
              dims_[d] = i;

              if(end == matxEnd) {
                sizes_[d] = op.Size(i) - start;
              } else {
                sizes_[d] = end - start;
              }

              //adjust size by stride
              sizes_[d] = (shape_type)std::ceil(static_cast<double>(sizes_[d])/ static_cast<double>(strides_[d]));
              d++;
            }
          }
          MATX_ASSERT_STR(d==Rank(), matxInvalidDim, "SliceOp: Number of dimensions without matxDropDim must equal new rank.");
        };

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
          {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            std::array<index_t, Rank()> inds{indices...};
            std::array<index_t, T::Rank()> ind{indices...};

#pragma unroll 
            for(int32_t i = 0; i < T::Rank(); i++) {
              ind[i] = starts_[i];
            }

#pragma unroll 
            for(int32_t i = 0; i < Rank(); i++) {
              ind[dims_[i]] += inds[i] * strides_[i]; 
            }

            //return op_(ind);
            return mapply(op_, ind);
          }

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto& operator()(Is... indices)
          {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            std::array<shape_type, Rank()> inds{indices...};
            std::array<shape_type, T::Rank()> ind{indices...};

#pragma unroll 
            for(int i = 0; i < T::Rank(); i++) {
              ind[i] = starts_[i];
            }

#pragma unroll 
            for(int i = 0; i < Rank(); i++) {
              ind[dims_[i]] += inds[i] * strides_[i]; 
            }

            //return op_(ind);
            return mapply(op_, ind);
          }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return DIM;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ shape_type Size(int32_t dim) const
        {
          return sizes_[dim];
        }

        template<typename R> __MATX_INLINE__ auto operator=(const R &rhs) { return set(*this, rhs); }
    };
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam T_wStrideT Input operator/tensor type
   * @param Op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <typename T_wStrideT>
    __MATX_INLINE__ auto slice( const T_wStrideT op, 
        const typename T_wStrideT::shape_type (&starts)[T_wStrideT::Rank()],
        const typename T_wStrideT::shape_type (&ends)[T_wStrideT::Rank()],
        const typename T_wStrideT::stride_type (&strides)[T_wStrideT::Rank()]) {
      return detail::SliceOp<T_wStrideT::Rank(),T_wStrideT>(op, starts, ends, strides);
    }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam T_wShapeT Input operator/tensor type
   * @param Op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <typename T_wShapeT>
    __MATX_INLINE__ auto slice( const T_wShapeT op, 
        const typename T_wShapeT::shape_type (&starts)[T_wShapeT::Rank()],
        const typename T_wShapeT::shape_type (&ends)[T_wShapeT::Rank()]) {
              typename T_wShapeT::shape_type strides[T_wShapeT::Rank()];
      for(int i = 0; i < T_wShapeT::Rank(); i++)
        strides[i] = 1;
      return detail::SliceOp<T_wShapeT::Rank(),T>(op, starts, ends, strides);
    }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam N The Rank of the output operator
   * @tparam T_wStridet_2 Input operator/tensor type
   * @param Op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <int N, typename T_wStridet_2>
    __MATX_INLINE__ auto slice( const T_wStridet_2 op, 
        const typename T_wStridet_2::shape_type (&starts)[T_wStridet_2::Rank()],
        const typename T_wStridet_2::shape_type (&ends)[T_wStridet_2::Rank()],
        const typename T_wStridet_2::stride_type (&strides)[T_wStridet_2::Rank()]) {
      return detail::SliceOp<N,T_wStridet_2>(op, starts, ends, strides);
    }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam N The Rank of the output operator
   * @tparam T_wShapet_2 Input operator/tensor type
   * @param Op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <int N, typename T_wShapet_2>
    __MATX_INLINE__ auto slice( const T_wShapet_2 op, 
        const typename T_wShapet_2::shape_type (&starts)[T_wShapet_2::Rank()],
        const typename T_wShapet_2::shape_type (&ends)[T_wShapet_2::Rank()]) {
              typename T_wShapet_2::shape_type strides[T_wShapet_2::Rank()];
      for(int i = 0; i < T_wShapet_2::Rank(); i++)
        strides[i] = 1;
      return detail::SliceOp<N,T_wShapet_2>(op, starts, ends, strides);
    }
} // end namespace matx
