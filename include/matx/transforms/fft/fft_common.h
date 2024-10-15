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

namespace matx {

enum class FFTNorm {
  BACKWARD, /// fft is unscaled, ifft is 1/N
  FORWARD, /// fft is scaled 1/N, ifft is not scaled
  ORTHO /// fft is scaled 1/sqrt(N), ifft is scaled 1/sqrt(N)
};

namespace detail {

  static constexpr int MAX_FFT_RANK = 2;

  enum class FFTType {
    C2C,
    R2C,
    C2R,
    Z2Z,
    D2Z,
    Z2D
  };

  enum class FFTDirection {
    FORWARD,
    BACKWARD
  };
    
  template <typename OutputTensor, typename InputTensor, typename Executor>
  __MATX_INLINE__ auto  GetFFTInputView([[maybe_unused]] OutputTensor &o,
                      const InputTensor &i, uint64_t fft_size,
                      [[maybe_unused]] const Executor &exec)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    using index_type = typename OutputTensor::shape_type;
    using T1    = typename OutputTensor::value_type;
    using T2    = typename InputTensor::value_type;
    constexpr int RANK = OutputTensor::Rank();

    index_type starts[RANK] = {0};
    index_type ends[RANK];
    index_type in_size = i.Lsize();
    index_type nom_fft_size = in_size;
    index_type act_fft_size;

    static_assert(!is_matx_static_descriptor_v<decltype(i.Descriptor())>, "FFTs cannot use static descriptors at this time");

    // Auto-detect FFT size
    if (fft_size == 0) {
      act_fft_size = o.Lsize();

      // If R2C transform, set length of FFT appropriately
      if constexpr ((std::is_same_v<T2, float> &&
                    std::is_same_v<T1, cuda::std::complex<float>>) ||
                    (std::is_same_v<T2, double> &&
                    std::is_same_v<T1, cuda::std::complex<double>>) ||
                    (std::is_same_v<T2, matxBf16> &&
                    std::is_same_v<T1, matxBf16Complex>) ||
                    (std::is_same_v<T2, matxFp16> &&
                    std::is_same_v<T1, matxFp16Complex>)) { // R2C
        nom_fft_size = in_size;
        act_fft_size = (o.Lsize() - 1) * 2;
      }
      else if constexpr ((std::is_same_v<T1, float> &&
                          std::is_same_v<T2, cuda::std::complex<float>>) ||
                        (std::is_same_v<T1, double> &&
                          std::is_same_v<T2, cuda::std::complex<double>>) ||
                        (std::is_same_v<T1, matxBf16> &&
                          std::is_same_v<T2, matxBf16Complex>) ||
                        (std::is_same_v<T1, matxFp16> &&
                          std::is_same_v<T2, matxFp16Complex>)) { // C2R
        nom_fft_size = (in_size - 1) * 2;
        act_fft_size = (o.Lsize() / 2) + 1;
      }
    }
    else {
      // Force FFT size
      act_fft_size = static_cast<index_type>(fft_size);
    }

    // Set up new shape if transform size doesn't match tensor
    if (nom_fft_size != act_fft_size) {
      std::fill_n(ends, RANK, matxEnd);

      // FFT shorter than the size of the input signal. Create a new view of this
      // slice.
      if (act_fft_size < nom_fft_size) {
        ends[RANK - 1] = act_fft_size;
        return slice(i, starts, ends);
      }
      else { // FFT length is longer than the input. Pad input

        // If the input needs to be padded we have to temporarily allocate a new
        // buffer, zero the output, then copy our input buffer. This is not very
        // efficient, but if cufft adds a zero-padding feature later we can take
        // advantage of that without changing the API.

        // Create a new shape where n is the size of the last dimension
        auto shape = i.Shape();
        *(shape.end() - 1) = act_fft_size;
        auto tot = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<typename decltype(shape)::value_type>());

        // Make a new buffer large enough for our input
        if constexpr (is_cuda_executor_v<Executor>) {
          const auto stream = exec.getStream();
          auto i_new = make_tensor<T2>(shape, MATX_ASYNC_DEVICE_MEMORY, stream);
          ends[RANK - 1] = i.Lsize();
          auto i_pad_part_v = slice(i_new, starts, ends);
printf("fft copy\n");
          (i_new = static_cast<promote_half_t<T2>>(0)).run(stream);
          printf("fft copy2\n");
          // example-begin copy-test-1
          matx::copy(i_pad_part_v, i, stream);
          // example-end copy-test-1
          return i_new;
        }
        else {
          auto i_new = make_tensor<T2>(shape, MATX_HOST_MALLOC_MEMORY);
          ends[RANK - 1] = i.Lsize();
          auto i_pad_part_v = slice(i_new, starts, ends);

          (i_new = static_cast<promote_half_t<T2>>(0)).run(exec);
          matx::copy(i_pad_part_v, i, exec);
          return i_new;        
        }
      }
    }

    return i;
  }  

  template <typename T1T, typename T2T>
  constexpr __MATX_INLINE__ FFTType DeduceFFTTransformType()
  {
    using T1 = typename T1T::value_type;
    using T2 = typename T2T::value_type;

    // Deduce plan type from view types
    if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
      if constexpr (std::is_same_v<T2, cuda::std::complex<float>>) {
        return FFTType::C2C;
      }
      else if constexpr (std::is_same_v<T2, float>) {

        return FFTType::R2C;
      }
    }
    else if constexpr (std::is_same_v<T1, float> &&
                       std::is_same_v<T2, cuda::std::complex<float>>) {
      return FFTType::C2R;
    }
    else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
      if constexpr (std::is_same_v<T2, cuda::std::complex<double>>) {
        return FFTType::Z2Z;
      }
      else if constexpr (std::is_same_v<T2, double>) {
        return FFTType::D2Z;
      }
    }
    else if constexpr (std::is_same_v<T1, double> &&
                       std::is_same_v<T2, cuda::std::complex<double>>) {
      return FFTType::Z2D;
    }
    else if constexpr (is_complex_half_v<T1>) {
      if constexpr (is_complex_half_v<T2>) {
        return FFTType::C2C;
      }
      else if constexpr (is_half_v<T2>) {
        return FFTType::R2C;
      }
    }
    else if constexpr (is_half_v<T1> && is_complex_half_v<T2>) {
      return FFTType::C2R;
    }
    //else {
      return FFTType::C2C;
    //}    
  }
}

};