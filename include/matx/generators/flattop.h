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

#include "matx/generators/generator1d.h"

namespace matx
{
  namespace detail {
    template <typename T> class FlatTop {
      private:
        index_t size_;

        static constexpr T a0 = 0.21557895;
        static constexpr T a1 = 0.41663158;
        static constexpr T a2 = 0.277263158;
        static constexpr T a3 = 0.083578947;
        static constexpr T a4 = 0.006947368;  

      public:

        using value_type = T;

        __MATX_INLINE__ std::string str() const { return "flattop"; }

        inline __MATX_HOST__ __MATX_DEVICE__ FlatTop(index_t size) : size_(size){};

        inline __MATX_HOST__ __MATX_DEVICE__ T operator()(index_t i) const
        {
          return  a0  
            - a1 * cuda::std::cos(2*M_PI*i / (size_ - 1))
            + a2 * cuda::std::cos(4*M_PI*i / (size_ - 1))
            - a3 * cuda::std::cos(6*M_PI*i / (size_ - 1))
            + a4 * cuda::std::cos(8*M_PI*i / (size_ - 1));
        }
    };
  }

  /**
   * Creates a Flattop window operator of shape s with the
   * window applies along the x, y, z, or w dimension
   *
   * @tparam T
   *   Data type
   * @tparam Dim
   *   Dimension to create window over
   * @tparam RANK
   *   The RANK of the shape
   *
   * @param s
   *   The shape of the tensor
   *
   * Returns values for a Flattop window across the selected dimension.
   */
  template <int Dim, typename ShapeType, typename T = float,
           std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
             inline auto flattop(ShapeType &&s)
             {
               constexpr int RANK = cuda::std::tuple_size<std::decay_t<ShapeType>>::value;
               static_assert(RANK > Dim);
               detail::FlatTop<T> h( *(s.begin() + Dim));
               return detail::matxGenerator1D_t<detail::FlatTop<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
             }

  /**
   * Creates a Flattop window operator of shape s with the
   * window applies along the x, y, z, or w dimension
   *
   * @tparam T
   *   Data type
   * @tparam Dim
   *   Dimension to create window over
   * @tparam RANK
   *   The RANK of the shape
   *
   * @param s
   *   The shape of the tensor
   *
   * Returns values for a Flattop window across the selected dimension.
   */
  template <int Dim, int RANK, typename T = float>
    inline auto flattop(const index_t (&s)[RANK])
    {
      return flattop<Dim>(detail::to_array(s));
    }


} // end namespace matx
