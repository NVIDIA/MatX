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

#include <cstdint>
#include <type_traits>

#include "matx/core/allocator.h"
#include "matx/core/error.h"
#include "matx/core/tensor.h"
#include "matx/core/type_utils.h"
#include "matx/transforms/fft/fft_cuda.h"

namespace matx {

/* Operator for perfoming the 2*exp(-j*pi*k/(2N)) part of the DCT */
namespace detail {
template <typename I, typename Out> class dctOp : public BaseOp<dctOp<I, Out>> {
private:
  using input_type = typename I::value_type;
  using value_type = typename input_type::value_type; // Real type from complex

  typename detail::base_type_t<Out> out_;
  typename detail::base_type_t<I> in_;
  index_t N_;

public:
  dctOp(Out out, I in, index_t N) : out_(out), in_(in), N_(N) {}

  template <typename CapType>
  __MATX_DEVICE__ inline void operator()(index_t idx)
  {
    const auto in_val = get_value<CapType>(in_, idx);
    if constexpr(CapType::ept == ElementsPerThread::ONE) {
      out_(idx) = static_cast<value_type>(
        in_val.real() * cuda::std::cos(static_cast<value_type>(-1 * M_PI) * idx / (static_cast<value_type>(2.0f) * N_)) -
        in_val.imag() * cuda::std::sin(static_cast<value_type>(-1 * M_PI) * idx / (static_cast<value_type>(2.0f) * N_)));
    }
  }

  __MATX_DEVICE__ inline void operator()(index_t idx)
  {
    this->operator()<DefaultCapabilities>(idx);
  }

  template <OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const
  {
    if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
      const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
      return combine_capabilities<Cap>(my_cap, 
        detail::get_operator_capability<Cap>(out_, in), 
        detail::get_operator_capability<Cap>(in_, in));
    }
    else {
      auto self_has_cap = capability_attributes<Cap>::default_value;
      return combine_capabilities<Cap>(self_has_cap, 
        detail::get_operator_capability<Cap>(out_, in), 
        detail::get_operator_capability<Cap>(in_, in));
    }
  }

  constexpr __MATX_HOST__ __MATX_DEVICE__ inline index_t Size(int i) const
  {
    return out_.Size(i);
  }

  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
  {
    return Out::Rank();
  }
};
}

/**
 * Discrete Cosine Transform
 *
 * Computes the DCT of input sequence "in". The input and output ranks must be
 * 1, and the sizes must match. This implementation uses the 2N padded version of
 * Makhoul's method which offloads the complex processing to cuFFT.
 *
 * @tparam T
 *   Input data type
 * @tparam RANK
 *   Rank of input and output tensor. Must be 1
 *
 * @param out
 *   Output tensor
 * @param in
 *   Input tensor
 * @param stream
 *   CUDA stream
 *
 **/
template <typename OutputTensor, typename InputTensor>
void dct(OutputTensor &out, const InputTensor &in,
         const cudaStream_t stream = 0)
{
  static_assert(OutputTensor::Rank() == InputTensor::Rank(), "DCT input and output tensor ranks must match");
  MATX_STATIC_ASSERT(OutputTensor::Rank() == 1, matxInvalidDim);
  index_t N = in.Size(OutputTensor::Rank() - 1);

  tensor_t<typename OutputTensor::value_type, 1> tmp{{N + 1}};
  (tmp = rfft(concat(0, in, reverse<0>(in)), 0, FFTNorm::BACKWARD)).run(stream);
  auto s = slice(tmp, {0}, {N});
  detail::dctOp(out, s, N).run(stream);
}

}; // namespace matx