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
      class OverlapOp : public BaseOp<OverlapOp<DIM, T>>
    {
      public:
        using value_type = typename T::value_type;
        using shape_type = index_t;
        using self_type = OverlapOp<DIM, T>;

      private:
        typename detail::base_type_t<T> op_;
        cuda::std::array<int32_t, DIM> dims_;
        cuda::std::array<shape_type, DIM+1> n_;
        cuda::std::array<shape_type, DIM+1> s_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        static_assert(DIM == 1, "overlap() only supports input rank 1 currently");

        __MATX_INLINE__ std::string str() const { return "overlap(" + op_.str() + ")"; }
        __MATX_INLINE__ OverlapOp(const T &op, const cuda::std::array<shape_type, DIM> &windows,
                                      const cuda::std::array<shape_type, DIM> &strides) : op_(op) {

          // This only works for 1D tensors going to 2D at the moment. Generalize to
          // higher dims later
          index_t window_size = windows[0];
          index_t stride_size = strides[0];

          MATX_ASSERT(stride_size <= window_size, matxInvalidSize);
          MATX_ASSERT(stride_size > 0, matxInvalidSize);

          // Figure out the actual length of the sequence we can use. It might be
          // shorter than the original operator if the window/stride doesn't line up
          // properly to make a rectangular matrix.
          shape_type adj_el = op_.Size(0) - window_size;
          adj_el -= adj_el % stride_size;

          n_[1] = window_size;
          s_[1] = 1;
          n_[0] = adj_el / stride_size + 1;
          s_[0] = stride_size;
        };

        template <detail::VecWidth InWidth, detail::VecWidth OutWidth>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(index_t i0, index_t i1) const
        {
          return op_<InWidth, OutWidth>(i0*s_[0] + i1);
        }

        template <detail::VecWidth InWidth, detail::VecWidth OutWidth>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(index_t i0, index_t i1)
        {
          return op_<InWidth, OutWidth>(i0*s_[0] + i1);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return DIM + 1;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ shape_type Size(int32_t dim) const
        {
          return n_[dim];
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

        ~OverlapOp() = default;
        OverlapOp(const OverlapOp &rhs) = default;
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
   * @brief Create an overlapping tensor view
   *
   * Creates an overlapping tensor view where an existing tensor can be
   * repeated into a higher rank with overlapping elements. For example, the
   * following 1D tensor [1 2 3 4 5] could be cloned into a 2d tensor with a
   * window size of 2 and overlap of 1, resulting in:
   *
   \verbatim
    [1 2
     2 3
     3 4
     4 5]
   \endverbatim
   * Currently this only works on 1D tensors going to 2D, but may be expanded
   * for higher dimensions in the future. Note that if the window size does not
   * divide evenly into the existing column dimension, the view may chop off the
   * end of the data to make the tensor rectangular.
   *
   * @tparam OpType
   *   Type of operator input
   * @tparam N
   *   Rank of overlapped window
   * @param op input operator
   * @param windows
   *   Window size (columns in output)
   * @param strides
   *   Strides between data elements
   *
   * @returns Overlapping view of data
   *
   */
  template <typename OpType, int N>
  __MATX_INLINE__ auto overlap( const OpType &op,
      const cuda::std::array<index_t, N> &windows,
      const cuda::std::array<index_t, N> &strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.template OverlapView<N>(windows, strides);
    } else {
      return detail::OverlapOp<N, OpType>(op, windows, strides);
    }
  }

  template <typename OpType, int N>
  __MATX_INLINE__ auto overlap( const OpType &op,
      const index_t (&windows)[N],
      const index_t (&strides)[N])
  {
    return overlap<OpType, N>(op,
        detail::to_array(windows),
        detail::to_array(strides));
  }

} // end namespace matx
