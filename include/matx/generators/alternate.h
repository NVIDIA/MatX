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
    template <typename T> 
      class Alternating : public BaseOp<Alternating<T>>{
      private:
        index_t size_;

      public:
        using value_type = T;
        using matxop = bool;        

	      __MATX_INLINE__ std::string str() const { return "alternate"; }
        __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Alternating(index_t size) : size_(size) {};

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return my_cap;
          } else {        
            auto self_has_cap = detail::capability_attributes<Cap>::default_value;
            return self_has_cap;
          }
        }       

        template <typename CapType>
        __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
        {
          return detail::ApplyGeneratorVecFunc<CapType, T>([](index_t idx) { return (-2 * (idx & 1)) + 1; }, i);
        }

        __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
        {
          return this->operator()<DefaultCapabilities>(i);
        }

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size([[maybe_unused]] int dim) const
        {
          return size_;
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }
    };
  }

  /**
   * Creates an alternating +1/-1 sequence
   *
   * @tparam T
   *   Data type
   *
   * @param len
   *   Length of sequence
   *
   */
  template <typename T = float>
    inline auto alternate(const index_t len)
    {
      return detail::Alternating<T>{len};
    }
} // end namespace matx
