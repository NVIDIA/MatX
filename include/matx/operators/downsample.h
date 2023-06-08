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
   * Upsamples a tensor by stuffing zeros
   */
  namespace detail {
    template <typename T>
      class DownsampleOp : public BaseOp<DownsampleOp<T>>
    {
      private:
        T op_;
        int32_t dim_;
        uint32_t n_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;
        using scalar_type = typename T::scalar_type;

        __MATX_INLINE__ std::string str() const { return "downsample(" + op_.str() + ")"; }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T::Rank();
        }

        __MATX_INLINE__ DownsampleOp(const T &op, int32_t dim, uint32_t n) : op_(op), dim_(dim), n_(n) {
        };

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          static_assert(sizeof...(Is)==Rank());
          static_assert((std::is_convertible_v<Is, index_t> && ... ));

          // convert variadic type to array so we can read/update
          std::array<index_t, Rank()> ind{indices...};

          ind[dim_] *= n_;
          return mapply(op_, ind);
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int32_t dim) const
        {
          return (op_.Size(dim) + n_ - 1) / n_;
        }

        template<typename R> __MATX_INLINE__ auto operator=(const R &rhs) {
          return set(*this, rhs);
        }

        static_assert(Rank() > 0, "DownsampleOp: Rank of operator must be greater than 0.");
        static_assert(T::Rank() > 0, "DownsampleOp: Rank of input operator must be greater than 0.");
    };
  }

  /**
   * @brief Downsample across one dimension
   *
   * Downsamples an input tensor across dimension `dim` by a factor of `n`. Downsampling is performed
   * by dropping all samples except every `n`th sample.
   *
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param dim Dimension to downsample
   * @param n Downsample rate
   * @return Upsampled operator
   */
  template <typename T>
  __MATX_INLINE__ auto downsample( const T &op, int32_t dim, uint32_t n) {
    return detail::DownsampleOp<T>(op, dim, n);
  }
} // end namespace matx
