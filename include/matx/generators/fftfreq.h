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
    template <class T> class FFTFreqOp {
      private:
        index_t n_;
        float d_;

      public:
        using value_type = T;
        using matxop = bool;

        __MATX_INLINE__ std::string str() const { return "fftfreq"; }

        inline FFTFreqOp(index_t n, float d = 1.0)
        {
          n_ = n;
          d_ = d;
        }

        __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ T operator()(index_t idx) const {
          index_t offset = idx >= (n_+1)/2 ? -n_ : 0;
          return static_cast<T>(idx + offset) / (d_*(T)n_);
        }
    };
  }


  /**
   * @brief Return FFT sample frequencies
   *
   * Returns the bin centers in cycles/unit of the sampling frequency known by the user.
   *
   * @tparam T Type of output
   * @param n Number of elements
   * @param d Sample spacing (defaults to 1.0)
   * @return Operator with sampling frequencies
   */
  template <typename T = float>
    inline auto fftfreq(index_t n, float d = 1.0)
    {
      detail::FFTFreqOp<T> l(n, d);
      cuda::std::array<index_t, 1> s{n};
      return detail::matxGenerator1D_t<detail::FFTFreqOp<T>, 0, decltype(s)>(std::move(s), l);
    }
} // end namespace matx
