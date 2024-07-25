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
#include <optional>

#include "matx/core/allocator.h"
#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/type_utils.h"

#include "matx/transforms/copy.h"
#include "matx/transforms/fft/fft_cuda.h"

namespace matx {
typedef enum {
  AMBGFUN_CUT_TYPE_2D,
  AMBGFUN_CUT_TYPE_DELAY,
  AMBGFUN_CUT_TYPE_DOPPLER,
} AMBGFunCutType_t;
};

namespace matx {
namespace detail {


template <class O, class I1, class I2>
class newYNorm : public BaseOp<newYNorm<O, I1, I2>> {
private:
  O out_;
  I1 xnorm_;
  I2 ynorm_;

public:
  newYNorm(O out, I1 xnorm, I2 ynorm) : out_(out), xnorm_(xnorm), ynorm_(ynorm)
  {
  }

  __MATX_DEVICE__ inline void operator()(index_t idy, index_t idx)
  {

    index_t xcol = idx - (xnorm_.Size(xnorm_.Rank() - 1) - 1) + idy;
    if (xcol >= 0 && xcol < (xnorm_.Size(xnorm_.Rank() - 1))) {
      typename I1::type xnorm = xnorm_(xcol);
      typename I2::type ynorm = ynorm_(idx);

      out_(idy, idx) = ynorm * cuda::std::conj(xnorm);
    }
    else {
      out_(idy, idx) = {0, 0};
    }
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
  {
    return out_.Size(dim);
  }

  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
  {
    return O::Rank();
  }
};

template <class O, class I1>
class AmbgFftXOp : public BaseOp<AmbgFftXOp<O, I1>> {
private:
  O out_;
  I1 x_;
  double fs_;
  double cut_;
  double nfreq_;

public:
  AmbgFftXOp(O out, I1 x, double fs, double cut, double nfreq)
      : out_(out), x_(x), fs_(fs), cut_(cut), nfreq_(nfreq)
  {
  }
  __MATX_DEVICE__ inline void operator()(index_t idx)
  {

    out_(idx) =
        exp(cuda::std::complex<float>{
            0, static_cast<float>(2.0 * M_PI *
                                  (-fs_ / 2.0 + idx * fs_ / nfreq_) * cut_)}) *
        x_(idx);
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
  {
    return out_.Size(dim);
  }
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
  {
    return O::Rank();
  }
};

template <class O, class I1> class AmbgDoppX : public BaseOp<AmbgDoppX<O, I1>> {
private:
  O out_;
  I1 x_;
  double fs_;
  double cut_;

public:
  AmbgDoppX(O out, I1 x, double fs, double cut)
      : out_(out), x_(x), fs_(fs), cut_(cut)
  {
  }
  __MATX_DEVICE__ inline void operator()(index_t idx)
  {

    out_(idx) = exp(cuda::std::complex<float>{
                    0, static_cast<float>(2.0 * M_PI * idx / fs_ * cut_)}) *
                x_(idx);
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
  {
    return out_.Size(dim);
  }
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
  {
    return O::Rank();
  }
};

template <typename AMFTensor, typename XTensor>
void ambgfun_impl(AMFTensor &amf, XTensor &x,
                     std::optional<XTensor> &y,
                     [[maybe_unused]] double fs, ::matx::AMBGFunCutType_t cut,
                     [[maybe_unused]] float cut_val, cudaStream_t stream = 0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  constexpr int RANK = XTensor::Rank();
  using T1 = typename XTensor::value_type;

  MATX_STATIC_ASSERT(is_cuda_complex_v<T1>, matxInvalidType);
  auto ry = x.View();
  //tensor_t<T1, RANK> ry(x);

  auto x_normdiv_v = make_tensor<T1>(x.Shape(),  MATX_ASYNC_DEVICE_MEMORY, stream);
  auto x_norm_v = make_tensor<float>({}, MATX_ASYNC_DEVICE_MEMORY, stream);

  (x_norm_v = sum(abs2(x))).run(stream);
  (x_norm_v = sqrt(x_norm_v)).run(stream);
  (x_normdiv_v = x / x_norm_v).run(stream);

  auto y_normdiv_v = x_normdiv_v.View();

  if (y) {
    ry.Reset(y.value().Data(), y.value().Shape());
    y_normdiv_v.Shallow(make_tensor<T1>(y_normdiv_v.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream));
    auto y_norm_v = make_tensor<float>({}, MATX_ASYNC_DEVICE_MEMORY, stream);

    (y_norm_v = sum(abs2(ry))).run(stream);
    (y_normdiv_v = ry / y_norm_v).run(stream);
  }

  index_t len_seq = x_normdiv_v.Size(RANK - 1) + y_normdiv_v.Size(RANK - 1);
  index_t nfreq = static_cast<index_t>(
      powf(2.0, static_cast<float>(std::ceil(std::log2(len_seq - 1)))));
  index_t xlen = x_normdiv_v.Size(RANK - 1);

  if (cut == ::matx::AMBGFUN_CUT_TYPE_2D) {
          
    auto new_ynorm_v = make_tensor<T1>({len_seq - 1, xlen}, MATX_ASYNC_DEVICE_MEMORY, stream);

    newYNorm(new_ynorm_v, x_normdiv_v, y_normdiv_v).run(stream);

    auto fullfft = make_tensor<T1>({(len_seq - 1), nfreq}, MATX_ASYNC_DEVICE_MEMORY, stream);
    auto partfft = fullfft.Slice({0, 0}, {(len_seq - 1), xlen});

    (fullfft = 0).run(stream);
    matx::copy(partfft, new_ynorm_v, stream);

    ifft_impl(fullfft, fullfft, 0, FFTNorm::BACKWARD, stream);

    // We need to temporarily allocate a complex output version of AMF since we
    // have no way to convert complex to real in an operator currently
    auto amf_tmp_v = make_tensor<T1>({(len_seq - 1), nfreq}, MATX_ASYNC_DEVICE_MEMORY, stream);

    (amf_tmp_v = (float)nfreq * abs(fftshift1D(fullfft))).run(stream);
    matx::copy(amf, amf_tmp_v.RealView(), stream);
  }
  else if (cut == ::matx::AMBGFUN_CUT_TYPE_DELAY) {
    auto fullfft_x = make_tensor<T1>({nfreq}, MATX_ASYNC_DEVICE_MEMORY, stream);
    auto partfft_x = fullfft_x.Slice({0}, {xlen});
    (fullfft_x = 0).run(stream);
    matx::copy(partfft_x, x_normdiv_v, stream);

    fft_impl(fullfft_x, fullfft_x, 0, FFTNorm::BACKWARD, stream);
    AmbgFftXOp(fullfft_x, fullfft_x, fs, cut_val, (float)nfreq).run(stream);
    ifft_impl(fullfft_x, fullfft_x, 0, FFTNorm::BACKWARD, stream);

    auto fullfft_y = make_tensor<T1>({nfreq}, MATX_ASYNC_DEVICE_MEMORY, stream);
    (fullfft_y = 0).run(stream);

    auto partfft_y = fullfft_y.Slice({0}, {xlen});
    matx::copy(partfft_y, y_normdiv_v, stream);
    (fullfft_y = fullfft_y * conj(fullfft_x)).run(stream);
    ifft_impl(fullfft_y, fullfft_y, 0, FFTNorm::BACKWARD, stream);

    // This allocation should not be necessary, but we're getting compiler
    // errors when cloning/slicing
    auto amf_tmp_v = make_tensor<T1>({fullfft_y.Size(0)}, MATX_ASYNC_DEVICE_MEMORY, stream);

    (amf_tmp_v = (float)nfreq * abs(ifftshift1D(fullfft_y))).run(stream);

    cuda::std::array<index_t, 2> amfv_size = {1, amf.Size(1)};
    auto amfv = make_tensor(amf_tmp_v.GetStorage(), amfv_size);
    matx::copy(amf, amfv.RealView(), stream);
  }
  else if (cut == ::matx::AMBGFUN_CUT_TYPE_DOPPLER) {
    auto fullfft_y = make_tensor<T1>({len_seq - 1}, MATX_ASYNC_DEVICE_MEMORY, stream);
    auto partfft_y = fullfft_y.Slice({0}, {y_normdiv_v.Size(0)});

    (fullfft_y = 0).run(stream);
    matx::copy(partfft_y, y_normdiv_v, stream);
    fft_impl(fullfft_y, fullfft_y, 0, FFTNorm::BACKWARD, stream);

    auto fullfft_x = make_tensor<T1>({len_seq - 1}, MATX_ASYNC_DEVICE_MEMORY, stream);
    (fullfft_x = 0).run(stream);

    cuda::std::array<index_t, 1> xnd_size = {x_normdiv_v.Size(0)};
    auto partfft_x = make_tensor(fullfft_x.GetStorage(), xnd_size);

    AmbgDoppX(partfft_x, x_normdiv_v, fs, cut_val).run(stream);
    fft_impl(fullfft_x, fullfft_x, 0, FFTNorm::BACKWARD, stream);

    // This allocation should not be necessary, but we're getting compiler
    // errors when cloning/slicing
    auto amf_tmp_v = make_tensor<T1>({fullfft_x.Size(0)}, MATX_ASYNC_DEVICE_MEMORY, stream);
    (fullfft_y = fullfft_y * conj(fullfft_x)).run(stream);
    ifft_impl(fullfft_y, fullfft_y, 0, FFTNorm::BACKWARD, stream);

    (amf_tmp_v = abs(fftshift1D(fullfft_y))).run(stream);

    cuda::std::array<index_t, 2> amfv_size = {1, amf.Size(1)};
    auto amfv = make_tensor(amf_tmp_v.GetStorage(), amfv_size);
    matx::copy(amf, amfv.RealView(), stream);
  }
}

}

}; // namespace matx
