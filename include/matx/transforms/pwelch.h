////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2023, NVIDIA Corporation
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

namespace matx
{
  template <typename PxxType, typename xType, typename wType>
    __MATX_INLINE__ void pwelch_impl(PxxType Pxx, const xType& x, const wType& w, index_t nperseg, index_t noverlap, index_t nfft, cudaStream_t stream=0)
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      MATX_ASSERT_STR(Pxx.Rank() == x.Rank(), matxInvalidDim, "pwelch:  Pxx rank must be the same as x rank");
      MATX_ASSERT_STR(nfft >= nperseg, matxInvalidDim, "pwelch:  nfft must be >= nperseg");
      MATX_ASSERT_STR((noverlap >= 0) && (noverlap < nperseg), matxInvalidDim, "pwelch:  Must have 0 <= noverlap < nperseg");

      // Create overlapping view
      auto x_with_overlaps = overlap(x,{nperseg}, {nperseg - noverlap});

      // Create temporary space for fft outputs
      index_t batches = x_with_overlaps.Shape()[0];
      auto X_with_overlaps = make_tensor<cuda::std::complex<typename PxxType::value_type>>({batches,static_cast<index_t>(nfft)},MATX_ASYNC_DEVICE_MEMORY,stream);

      if constexpr (std::is_same_v<wType, std::nullopt_t>)
      {
        (X_with_overlaps = fft(x_with_overlaps,nfft)).run(stream);
      }
      else
      {
        (X_with_overlaps = fft(x_with_overlaps * w,nfft)).run(stream);
      }

      // Compute magnitude squared in-place
      (X_with_overlaps = conj(X_with_overlaps) * X_with_overlaps).run(stream);
      auto mag_sq_X_with_overlaps = X_with_overlaps.RealView();

      // Perform the reduction across 'batches' rows and normalize
      auto norm_factor = static_cast<typename PxxType::value_type>(1.) / static_cast<typename PxxType::value_type>(batches);
      (Pxx = sum(mag_sq_X_with_overlaps, {0}) * norm_factor).run(stream);
    }

} // end namespace matx
