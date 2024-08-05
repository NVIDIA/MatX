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
        using value_type = typename T::value_type;
        using shape_type = index_t; 
        using self_type = SliceOp<DIM, T>;

      private:
        typename base_type<T>::type op_;
        cuda::std::array<shape_type, DIM> sizes_;
        cuda::std::array<int32_t, DIM> dims_;
        cuda::std::array<shape_type, T::Rank()> starts_;
        cuda::std::array<shape_type, T::Rank()> strides_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        static_assert(T::Rank()>0, "SliceOp: Rank of operator must be greater than 0.");
        static_assert(DIM<=T::Rank(), "SliceOp: DIM must be less than or equal to operator rank.");

        __MATX_INLINE__ std::string str() const { return "slice(" + op_.str() + ")"; }

        __MATX_INLINE__ SliceOp(T op, const cuda::std::array<shape_type, T::Rank()> &starts,
                                      const cuda::std::array<shape_type, T::Rank()> &ends,
                                      const cuda::std::array<shape_type, T::Rank()> &strides) : op_(op) {
          int32_t d = 0;
          for(int32_t i = 0; i < T::Rank(); i++) {
            shape_type start = starts[i] < 0 ? op.Size(i) + starts[i] : starts[i];
            shape_type end   = ends[i]   < 0 ? op.Size(i) + ends[i]   : ends[i];

            MATX_ASSERT_STR((start > matxIdxSentinel) || (start < op.Size(i)), matxInvalidDim,
              "Slice slice index out of range of operator");
            MATX_ASSERT_STR((end > matxIdxSentinel) || (end <= op.Size(i)), matxInvalidDim,
              "Slice end index out of range of operator");

            starts_[i] = start;
            strides_[i] = strides[i];

            // compute dims and sizes
            if(end != matxDropDim) {
              MATX_ASSERT_STR(end != matxKeepDim, matxInvalidParameter, "matxKeepDim only valid for clone(), not slice()");
              
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
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
          {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            cuda::std::array<index_t, Rank()> inds{indices...};
            cuda::std::array<index_t, T::Rank()> ind{indices...};

#pragma unroll 
            for(int32_t i = 0; i < T::Rank(); i++) {
              ind[i] = starts_[i];
            }

#pragma unroll 
            for(int32_t i = 0; i < Rank(); i++) {
              ind[dims_[i]] += inds[i] * strides_[i]; 
            }

            //return op_(ind);
            return cuda::std::apply(op_, ind);
          }

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
          {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            cuda::std::array<shape_type, Rank()> inds{indices...};
            cuda::std::array<shape_type, T::Rank()> ind{indices...};

#pragma unroll 
            for(int i = 0; i < T::Rank(); i++) {
              ind[i] = starts_[i];
            }

#pragma unroll 
            for(int i = 0; i < Rank(); i++) {
              ind[dims_[i]] += inds[i] * strides_[i]; 
            }

            //return op_(ind);
            return cuda::std::apply(op_, ind);
          }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return DIM;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ shape_type Size(int32_t dim) const
        {
          return sizes_[dim];
        }

        ~SliceOp() = default;
        SliceOp(const SliceOp &rhs) = default;
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

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op, 
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      const cuda::std::array<index_t, OpType::Rank()> &strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.Slice(starts, ends, strides);
    } else {
      return detail::SliceOp<OpType::Rank(),OpType>(op, starts, ends, strides);
    }
  }

  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op, 
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()],
      const index_t (&strides)[OpType::Rank()]) 
  {
    return slice(op, 
        detail::to_array(starts), 
        detail::to_array(ends),
        detail::to_array(strides));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @return sliced operator
   */
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op, 
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends)
  {
    cuda::std::array<index_t, OpType::Rank()> strides;
    strides.fill(1);

    return slice(op, starts, ends, strides);
  }
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op, 
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()]) 
  {
    return slice(op, 
        detail::to_array(starts), 
        detail::to_array(ends));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue. 
   *
   * The Rank template parameter N is optional when rank does not change
   *
   * @tparam N The Rank of the output operator - optional when slice produces same rank as input
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <int N, typename OpType>
    __MATX_INLINE__ auto slice( const OpType op, 
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      const cuda::std::array<index_t, OpType::Rank()> &strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.template Slice<N>(starts, ends, strides);
    } else {
      return detail::SliceOp<N,OpType>(op, starts, ends, strides);
    }
  }

  template <int N, typename OpType>
    __MATX_INLINE__ auto slice( const OpType op, 
        const index_t (&starts)[OpType::Rank()],
        const index_t (&ends)[OpType::Rank()],
        const index_t (&strides)[OpType::Rank()]) 
  {
    return slice<N,OpType>(op, 
        detail::to_array(starts), 
        detail::to_array(ends),
        detail::to_array(strides));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue. 
   
   * The Rank template parameter N is optional when rank does not change
   *
   * @tparam N The Rank of the output operator - optional when slice produces same rank as input
   * @tparam OpType Input operator/tensor type
   * @param opIn Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @return sliced operator
   */
  template <int N, typename OpType>
  __MATX_INLINE__ auto slice (const OpType opIn, 
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends)
  {
    cuda::std::array<index_t, OpType::Rank()> strides;
    strides.fill(1);
    return slice<N,OpType>(opIn, starts, ends, strides);
  }

  template <int N, typename OpType>
  __MATX_INLINE__ auto slice (const OpType opIn, 
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()]) 
  {
    return slice<N,OpType>(opIn, 
        detail::to_array(starts), 
        detail::to_array(ends));
  }
} // end namespace matx
