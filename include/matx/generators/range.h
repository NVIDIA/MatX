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
    template <class T> class Range : public BaseOp<Range<T>> {
      private:
        T first_;
        T step_;

      public:
        using value_type = T;
        using matxop = bool;

        Range() = default;

        __MATX_INLINE__ std::string str() const { return "range"; }

        Range(T first, T step) : first_(first), step_(step) {}

        template <detail::ElementsPerThread EPT>
        __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ auto operator()(index_t idx) const
        {
          return detail::ApplyGeneratorVecFunc<EPT, T>([this](index_t i) {
            if constexpr (is_matx_half_v<T>) {
MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
              return first_ + T(static_cast<T>((float)i) * step_);
MATX_IGNORE_WARNING_POP_GCC
            }
            else {
MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
              return first_ + T(static_cast<T>(i) * step_);
MATX_IGNORE_WARNING_POP_GCC
            }
          }, idx);
        }

        __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ auto operator()(index_t idx) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(idx);
        }

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size([[maybe_unused]] int dim) const
        {
          return index_t(0); // Range is used with matxGenerator1D_t which provides the size
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }
    };
  }

  /**
   * Create a range of values along the x dimension
   *
   * Creates a range of values of type T with a start and step size.
   * Value is determined by the index in operator()
   *
   * @param s
   *   Tensor shape
   * @param first
   *   Starting value
   * @param step
   *   Step size
   *
   */
  template <int Dim, typename ShapeType, typename T = float,
           std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
             inline auto range(ShapeType &&s, T first, T step)
             {
               constexpr int RANK = cuda::std::tuple_size<std::decay_t<ShapeType>>::value;
               static_assert(RANK > Dim);
               detail::Range<T> r(first, step);
               return detail::matxGenerator1D_t<detail::Range<T>, Dim, ShapeType>(std::forward<ShapeType>(s), r);
             }

  /**
   * Create a range of values along the x dimension
   *
   * Creates a range of values of type T with a start and step size.
   * Value is determined by the index in operator()
   *
   * @param s
   *   Tensor shape
   * @param first
   *   Starting value
   * @param step
   *   Step size
   *
   */
  template <int Dim, int RANK, typename T = float>
    inline auto range(const index_t (&s)[RANK], T first, T step)
    {
      return range<Dim>(detail::to_array(s), first, step);
    }


} // end namespace matx
