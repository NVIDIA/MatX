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

#include "matx/generators/range.h"

namespace matx
{
  namespace detail {
    template <class T> class LinspaceOp {
      private:
        Range<T> range_;

      public:
        using value_type = T;
        using matxop = bool;

        __MATX_INLINE__ std::string str() const { return "linspace"; }


        inline LinspaceOp(T first, T last, index_t count)
        {
#ifdef __CUDA_ARCH__
          range_ = Range<T>{first, (last - first) / static_cast<T>(count - 1)};
#else
          // Host has no support for most half precision operators/intrinsics
          if constexpr (is_matx_half_v<T>) {
            range_ = Range<T>{static_cast<float>(first),
              (static_cast<float>(last) - static_cast<float>(first)) /
                static_cast<float>(count - 1)};
          }
          else {
            range_ = Range<T>{first, (last - first) / static_cast<T>(count - 1)};
          }
#endif
        }

        __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ T operator()(index_t idx) const { return range_(idx); }
    };
  }


  /**
   * @brief Create a linearly-spaced range of values
   *
   * Creates a set of values using a start and end that are linearly-
   * spaced apart over the set of values. Distance is determined
   * by the shape and selected dimension.
   * 
   * @tparam T Operator type
   * @tparam Dim Dimension to operate over
   * @tparam ShapeType Shape type
   * @param s Shape object
   * @param first First value
   * @param last Last value
   * @return Operator with linearly-spaced values 
   */
  template <int Dim, typename ShapeType, typename T,
           std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
             inline auto linspace(ShapeType &&s, T first, T last)
             {
               constexpr int RANK = cuda::std::tuple_size<std::decay_t<ShapeType>>::value;
               static_assert(RANK > Dim);
               auto count =  *(s.begin() + Dim);
               detail::LinspaceOp<T> l(first, last, count);
               return detail::matxGenerator1D_t<detail::LinspaceOp<T>, Dim, ShapeType>(std::forward<ShapeType>(s), l);
             }

  /**
   * @brief Create a linearly-spaced range of values
   *
   * Creates a set of values using a start and end that are linearly-
   * spaced apart over the set of values. Distance is determined
   * by the shape and selected dimension.
   * 
   * @tparam Dim Dimension to operate over
   * @tparam RANK Rank of shape
   * @tparam T Operator type
   * @param s Array of sizes
   * @param first First value
   * @param last Last value
   * @return Operator with linearly-spaced values 
   */
  template <int Dim, int RANK, typename T>
    inline auto linspace(const index_t (&s)[RANK], T first, T last)
    {
      return linspace<Dim>(detail::to_array(s), first, last);
    }
} // end namespace matx
