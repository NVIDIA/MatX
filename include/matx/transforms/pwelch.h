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

  enum PwelchOutputScaleMode {
    PwelchOutputScaleMode_Spectrum,
    PwelchOutputScaleMode_Density,
    PwelchOutputScaleMode_Spectrum_dB,
    PwelchOutputScaleMode_Density_dB
  };

  namespace detail {
    template<PwelchOutputScaleMode OUTPUT_SCALE_MODE, typename T_IN, typename T_OUT>
    __global__ void pwelch_kernel(const T_IN t_in, T_OUT t_out, typename T_OUT::value_type fs)
    {
      const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      const index_t batches = t_in.Shape()[0];
      const index_t nfft = t_in.Shape()[1];

      if (tid < nfft)
      {
        typename T_OUT::value_type pxx = 0;
        constexpr typename T_OUT::value_type ten = 10;

        for (index_t batch = 0; batch < batches; batch++)
        {
          pxx += cuda::std::norm(t_in(batch, tid));
        }

        if constexpr (OUTPUT_SCALE_MODE == PwelchOutputScaleMode_Spectrum)
        {
          t_out(tid) = pxx / batches;
        }
        else if constexpr (OUTPUT_SCALE_MODE == PwelchOutputScaleMode_Density)
        {
          t_out(tid) = pxx / (batches * fs);
        }
        else if constexpr (OUTPUT_SCALE_MODE == PwelchOutputScaleMode_Spectrum_dB)
        {
          pxx /= batches;
          if (pxx != 0)
          {
            t_out(tid) = ten * cuda::std::log10(pxx);
          }
          else
          {
            t_out(tid) = cuda::std::numeric_limits<typename T_OUT::value_type>::lowest();
          }
        }
        else if constexpr (OUTPUT_SCALE_MODE == PwelchOutputScaleMode_Density_dB)
        {
          pxx /= (batches * fs);
          if (pxx != 0)
          {
            t_out(tid) = ten * cuda::std::log10(pxx);
          }
          else
          {
            t_out(tid) = cuda::std::numeric_limits<typename T_OUT::value_type>::lowest();
          }
        }
      }
    }
  };

  extern int g_pwelch_alg_mode;
  template <typename PxxType, typename xType, typename wType>
    __MATX_INLINE__ void pwelch_impl(PxxType Pxx, const xType& x, const wType& w, index_t nperseg, index_t noverlap, index_t nfft, PwelchOutputScaleMode output_scale_mode, typename PxxType::value_type fs, cudaStream_t stream=0)
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

      int tpb = 512;
      int bpk = (static_cast<int>(nfft) + tpb - 1) / tpb;

      if (output_scale_mode == PwelchOutputScaleMode_Spectrum)
      {
        detail::pwelch_kernel<PwelchOutputScaleMode_Spectrum><<<bpk, tpb, 0, stream>>>(X_with_overlaps, Pxx, fs);
      }
      else if (output_scale_mode == PwelchOutputScaleMode_Density)
      {
        detail::pwelch_kernel<PwelchOutputScaleMode_Density><<<bpk, tpb, 0, stream>>>(X_with_overlaps, Pxx, fs);
      }
      else if (output_scale_mode == PwelchOutputScaleMode_Spectrum_dB)
      {
        detail::pwelch_kernel<PwelchOutputScaleMode_Spectrum_dB><<<bpk, tpb, 0, stream>>>(X_with_overlaps, Pxx, fs);
      }
      else //if (output_scale_mode == PwelchOutputScaleMode_Density_dB)
      {
        detail::pwelch_kernel<PwelchOutputScaleMode_Density_dB><<<bpk, tpb, 0, stream>>>(X_with_overlaps, Pxx, fs);
      }
    }

} // end namespace matx
