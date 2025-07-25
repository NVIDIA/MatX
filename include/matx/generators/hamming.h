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
#include <type_traits>

namespace matx
{
  namespace detail {
    template <typename T> class Hamming : public BaseOp<Hamming<T>> {
      private:
        index_t size_;

      public:
        using value_type = T;
        using matxop = bool;

        __MATX_INLINE__ std::string str() const { return "hamming"; }
	
        inline __MATX_HOST__ __MATX_DEVICE__ Hamming(index_t size) : size_(size){};

        template <detail::ElementsPerThread EPT>
        inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const 
        {
          return detail::ApplyGeneratorVecFunc<EPT, T>([this](index_t idx) { return T(.54) - T(.46) * cuda::std::cos(T(2 * M_PI) * T(idx) / T(size_ - 1)); }, i);
        }

        inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const 
        {
          return this->operator()<detail::ElementsPerThread::ONE>(i);
        }

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size([[maybe_unused]] int dim) const
        {
          return size_;
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }
    };
  }


  /**
   * Creates a Hamming window operator of shape s with the
   * window applies along the specified dimension
   *
   * @tparam T
   *   Data type
   * @tparam Dim
   *   Dimension to create window over
   * @tparam RANK
   *   The RANK of the shape, can be deduced from shape
   *
   * @param s
   *   The shape of the tensor
   *
   * Returns values for a Hamming window across the selected dimension.
   */
  template <int Dim, typename ShapeType, typename T = float, 
           std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
             inline auto hamming(ShapeType &&s)
             {
               constexpr int RANK = cuda::std::tuple_size<std::decay_t<ShapeType>>::value;
               static_assert(RANK > Dim);
               detail::Hamming<T> h( *(s.begin() + Dim));
               return detail::matxGenerator1D_t<detail::Hamming<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
             }

  /**
   * Creates a Hamming window operator of C-array shape s with the
   * window applies along the specified dimension
   *
   * @tparam T
   *   Data type
   * @tparam Dim
   *   Dimension to create window over
   * @tparam RANK
   *   The RANK of the shape, can be deduced from shape
   *
   * @param s
   *   C array representing shape of the tensor
   *
   * Returns values for a Hamming window across the selected dimension.
   */
  template <int Dim, int RANK, typename T = float>
    inline auto hamming(const index_t (&s)[RANK])
    {
      const auto shape = detail::to_array(s);
      return hamming<Dim, decltype(shape), T>(std::forward<decltype(shape)>(shape));
    }
} // end namespace matx
