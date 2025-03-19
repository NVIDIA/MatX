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

namespace matx {

  enum PwelchOutputScaleMode {
    PwelchOutputScaleMode_Spectrum,
    PwelchOutputScaleMode_Density,
    PwelchOutputScaleMode_Spectrum_dB,
    PwelchOutputScaleMode_Density_dB
  };

  namespace detail {

#ifdef __CUDACC__
    template<PwelchOutputScaleMode OUTPUT_SCALE_MODE, typename T_IN, typename T_OUT, typename fsType>
    __global__ void pwelch_kernel(const T_IN t_in, T_OUT t_out, fsType fs)
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

        if constexpr (OUTPUT_SCALE_MODE == PwelchOutputScaleMode_Spectrum) {
          t_out(tid) = pxx / batches;
        }
        else if constexpr (OUTPUT_SCALE_MODE == PwelchOutputScaleMode_Density) {
          t_out(tid) = pxx / (batches * fs);
        }
        else if constexpr (OUTPUT_SCALE_MODE == PwelchOutputScaleMode_Spectrum_dB) {
          pxx /= batches;
          if (pxx != 0) {
            t_out(tid) = ten * cuda::std::log10(pxx);
          }
          else {
            t_out(tid) = cuda::std::numeric_limits<typename T_OUT::value_type>::lowest();
          }
        }
        else if constexpr (OUTPUT_SCALE_MODE == PwelchOutputScaleMode_Density_dB) {
          pxx /= (batches * fs);
          if (pxx != 0) {
            t_out(tid) = ten * cuda::std::log10(pxx);
          }
          else {
            t_out(tid) = cuda::std::numeric_limits<typename T_OUT::value_type>::lowest();
          }
        }
      }
    }
#endif

  };
};
