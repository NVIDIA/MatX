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

#include "matx_allocator.h"
#include "matx_error.h"
#include "matx_shape.h"
#include "matx_tensor.h"
#include "matx_type_utils.h"

namespace matx {
namespace signal {

typedef enum {
  AMGBFUN_CUT_TYPE_2D,
  AMGBFUN_CUT_TYPE_DELAY,
  AMGBFUN_CUT_TYPE_DOPPLER,
} AMBGFunCutType_t;

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

  __device__ inline void operator()(index_t idy, index_t idx)
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

  __host__ __device__ inline index_t Size(uint32_t i) const
  {
    return out_.Size(i);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
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
  __device__ inline void operator()(index_t idx)
  {

    out_(idx) =
        exp(cuda::std::complex<float>{
            0, static_cast<float>(2.0 * M_PI *
                                  (-fs_ / 2.0 + idx * fs_ / nfreq_) * cut_)}) *
        x_(idx);
  }

  __host__ __device__ inline index_t Size(uint32_t i) const
  {
    return out_.Size(i);
  }
  static inline constexpr __host__ __device__ int32_t Rank()
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
  __device__ inline void operator()(index_t idx)
  {

    out_(idx) = exp(cuda::std::complex<float>{
                    0, static_cast<float>(2.0 * M_PI * idx / fs_ * cut_)}) *
                x_(idx);
  }

  __host__ __device__ inline index_t Size(uint32_t i) const
  {
    return out_.Size(i);
  }
  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return O::Rank();
  }
};

template <typename T1, typename T2, int RANK>
void InternalAmbgFun(tensor_t<T2, 2> &amf, tensor_t<T1, RANK> x,
                     std::optional<tensor_t<T1, RANK>> y,
                     [[maybe_unused]] double fs, AMBGFunCutType_t cut,
                     [[maybe_unused]] float cut_val, cudaStream_t stream = 0)
{
  T1 *x_normdiv, *y_normdiv;
  float *x_norm, *y_norm;

  MATX_ASSERT(is_cuda_complex_v<T1>, matxInvalidType);
  tensor_t<T1, RANK> ry(x);

  matxAlloc(reinterpret_cast<void **>(&x_normdiv),
            sizeof(T1) * x.Size(RANK - 1), MATX_ASYNC_DEVICE_MEMORY, stream);
  matxAlloc(reinterpret_cast<void **>(&x_norm), sizeof(*x_norm),
            MATX_ASYNC_DEVICE_MEMORY, stream);

  auto x_normdiv_v = tensor_t<T1, 1>(x_normdiv, x.Shape());
  auto x_norm_v = tensor_t<float, 0>(x_norm);

  reduce(x_norm_v, norm(x), reduceOpSum<float>(), 0);
  (x_norm_v = sqrt(x_norm_v)).run(stream);
  (x_normdiv_v = x / x_norm_v).run(stream);

  tensor_t<T1, RANK> y_normdiv_v(x_normdiv_v.Data(), x_normdiv_v.Shape());

  if (y) {
    ry.Shallow(y.value());
    matxAlloc(reinterpret_cast<void **>(&y_normdiv),
              sizeof(T1) * ry.Size(RANK - 1), MATX_ASYNC_DEVICE_MEMORY, stream);
    matxAlloc(reinterpret_cast<void **>(&y_norm), sizeof(*y_norm),
              MATX_ASYNC_DEVICE_MEMORY, stream);
    y_normdiv_v.Shallow(tensor_t<T1, RANK>(y_normdiv, ry.Shape()));
    auto y_norm_v = tensor_t<float, 0>(y_norm);

    reduce(y_norm_v, norm(ry), reduceOpSum<float>(), 0);
    (y_normdiv_v = ry / y_norm_v).run(stream);
  }

  index_t len_seq = x_normdiv_v.Size(RANK - 1) + y_normdiv_v.Size(RANK - 1);
  uint32_t nfreq = static_cast<uint32_t>(
      powf(2.0, static_cast<float>(std::ceil(std::log2(len_seq - 1)))));
  index_t xlen = x_normdiv_v.Size(RANK - 1);

  if (cut == AMGBFUN_CUT_TYPE_2D) {
    T1 *new_ynorm;
    matxAlloc(reinterpret_cast<void **>(&new_ynorm),
              sizeof(T1) * (len_seq - 1) * xlen, MATX_ASYNC_DEVICE_MEMORY,
              stream);
    auto new_ynorm_v = tensor_t<T1, 2>(new_ynorm, {len_seq - 1, xlen});

    newYNorm(new_ynorm_v, x_normdiv_v, y_normdiv_v).run(stream);

    T1 *fft_data, *amf_tmp;
    matxAlloc(reinterpret_cast<void **>(&fft_data),
              sizeof(*fft_data) * nfreq * (len_seq - 1),
              MATX_ASYNC_DEVICE_MEMORY, stream);
    auto fullfft = tensor_t<T1, 2>(fft_data, {{(len_seq - 1), nfreq}});
    auto partfft = fullfft.Slice({0, 0}, {(len_seq - 1), xlen});

    (fullfft = 0).run(stream);
    copy(partfft, new_ynorm_v, stream);

    ifft(fullfft, fullfft, stream);

    // We need to temporarily allocate a complex output version of AMF since we
    // have no way to convert complex to real in an operator currently
    matxAlloc(reinterpret_cast<void **>(&amf_tmp),
              sizeof(*amf_tmp) * nfreq * (len_seq - 1),
              MATX_ASYNC_DEVICE_MEMORY, stream);
    auto amf_tmp_v = tensor_t<T1, 2>(amf_tmp, {{(len_seq - 1), nfreq}});
    (amf_tmp_v = (float)nfreq * abs(fftshift1D(fullfft))).run(stream);
    copy(amf, amf_tmp_v.RealView(), stream);

    matxFree(fft_data);
    matxFree(amf_tmp);
  }
  else if (cut == AMGBFUN_CUT_TYPE_DELAY) {
    T1 *fft_data_x, *fft_data_y, *amf_tmp;
    matxAlloc(reinterpret_cast<void **>(&fft_data_x),
              sizeof(*fft_data_x) * nfreq, MATX_ASYNC_DEVICE_MEMORY, stream);
    matxAlloc(reinterpret_cast<void **>(&fft_data_y),
              sizeof(*fft_data_y) * nfreq, MATX_ASYNC_DEVICE_MEMORY, stream);
    auto fullfft_x = tensor_t<T1, 1>(fft_data_x, {nfreq});
    auto partfft_x = fullfft_x.Slice({0}, {xlen});
    (fullfft_x = 0).run(stream);
    copy(partfft_x, x_normdiv_v, stream);

    fft(fullfft_x, fullfft_x, stream);
    AmbgFftXOp(fullfft_x, fullfft_x, fs, cut_val, (float)nfreq).run(stream);
    ifft(fullfft_x, fullfft_x, stream);

    auto fullfft_y = tensor_t<T1, 1>(fft_data_y, {nfreq});
    (fullfft_y = 0).run(stream);

    auto partfft_y = fullfft_y.Slice({0}, {xlen});
    copy(partfft_y, y_normdiv_v, stream);
    (fullfft_y = fullfft_y * conj(fullfft_x)).run(stream);
    ifft(fullfft_y, fullfft_y, stream);

    // This allocation should not be necessary, but we're getting compiler
    // errors when cloning/slicing
    matxAlloc(reinterpret_cast<void **>(&amf_tmp),
              sizeof(*amf_tmp) * fullfft_y.Size(0), MATX_ASYNC_DEVICE_MEMORY,
              stream);
    auto amf_tmp_v = tensor_t<T1, 1>(amf_tmp, {fullfft_y.Size(0)});

    (amf_tmp_v = (float)nfreq * abs(ifftshift1D(fullfft_y))).run(stream);

    auto amfv = tensor_t<T1, 2>(amf_tmp_v.Data(), {1, amf.Size(1)});
    copy(amf, amfv.RealView(), stream);

    matxFree(amf_tmp);
    matxFree(fft_data_x);
    matxFree(fft_data_y);
  }
  else if (cut == AMGBFUN_CUT_TYPE_DOPPLER) {
    T1 *fft_data_x, *fft_data_y, *amf_tmp;
    matxAlloc(reinterpret_cast<void **>(&fft_data_x),
              sizeof(*fft_data_x) * (len_seq - 1), MATX_ASYNC_DEVICE_MEMORY,
              stream);
    matxAlloc(reinterpret_cast<void **>(&fft_data_y),
              sizeof(*fft_data_y) * (len_seq - 1), MATX_ASYNC_DEVICE_MEMORY,
              stream);
    auto fullfft_y = tensor_t<T1, 1>(fft_data_y, {len_seq - 1});
    auto partfft_y = fullfft_y.Slice({0}, {y_normdiv_v.Size(0)});

    (fullfft_y = 0).run(stream);
    copy(partfft_y, y_normdiv_v, stream);
    fft(fullfft_y, fullfft_y, stream);

    auto fullfft_x = tensor_t<T1, 1>(fft_data_x, {len_seq - 1});
    (fullfft_x = 0).run(stream);

    auto partfft_x = tensor_t<T1, 1>(fft_data_x, {x_normdiv_v.Size(0)});

    AmbgDoppX(partfft_x, x_normdiv_v, fs, cut_val).run(stream);
    fft(fullfft_x, fullfft_x, stream);

    // This allocation should not be necessary, but we're getting compiler
    // errors when cloning/slicing
    matxAlloc(reinterpret_cast<void **>(&amf_tmp),
              sizeof(*amf_tmp) * fullfft_x.Size(0), MATX_ASYNC_DEVICE_MEMORY,
              stream);
    auto amf_tmp_v = tensor_t<T1, 1>(amf_tmp, {fullfft_x.Size(0)});
    (fullfft_y = fullfft_y * conj(fullfft_x)).run(stream);
    ifft(fullfft_y, fullfft_y, stream);

    (amf_tmp_v = abs(fftshift1D(fullfft_y))).run(stream);

    auto amfv = tensor_t<T1, 2>(amf_tmp_v.Data(), {1, amf.Size(1)});
    copy(amf, amfv.RealView(), stream);
    matxFree(amf_tmp);
    matxFree(fft_data_x);
    matxFree(fft_data_y);
  }

  matxFree(x_normdiv);
  matxFree(x_norm);

  if (y) {
    matxFree(y_normdiv);
    matxFree(y_norm);
  }
}

/**
 * Cross-ambiguity function
 *
 * Generates a cross-ambiguity magnitude function from inputs x and y. The
 * ambiguity function generates a 2D delay vs doppler matrix of the cross
 * ambiguity of x and y.
 *
 * @tparam T1
 *   x/y vector types
 * @tparam T2
 *   Type of output
 * @tparam RANK
 *   Rank of input matrix. Must be 1 currently
 *
 * @param amf
 *   2D output matrix where rows are the Doppler (Hz) shift and columns are the
 * delay in seconds.
 * @param x
 *   First input signal
 * @param y
 *   Second input signal
 * @param fs
 *   Sampling frequency
 * @param cut
 *   Type of cut. 2D is effectively no cut. Delay cut returns a cut with zero
 * time delay. Doppler generates a cut with zero Doppler shift. Note that in
 * both Delay and Doppler mode, the output matrix must be a 2D tensor where the
 * first dimension is 1 to match the type of 2D mode.
 * @param cut_val
 *   Value to perform the cut at
 * @param stream
 *   CUDA stream
 *
 */
template <typename T1, typename T2, int RANK>
inline void ambgfun(tensor_t<T2, 2> &amf, tensor_t<T1, RANK> x,
                    tensor_t<T1, RANK> y, double fs, AMBGFunCutType_t cut,
                    float cut_val = 0.0, cudaStream_t stream = 0)
{
  InternalAmbgFun(amf, x, std::make_optional(y), fs, cut, cut_val, stream);
}

/**
 * Ambiguity function
 *
 * Generates an ambiguity magnitude function from input signal x. The ambiguity
 * function generates a 2D delay vs doppler matrix of the input signal.
 *
 * @tparam T1
 *   x vector type
 * @tparam T2
 *   Type of output
 * @tparam RANK
 *   Rank of input matrix. Must be 1 currently
 *
 * @param amf
 *   2D output matrix where rows are the Doppler (Hz) shift and columns are the
 * delay in seconds.
 * @param x
 *   First input signal
 * @param fs
 *   Sampling frequency
 * @param cut
 *   Type of cut. 2D is effectively no cut. Delay cut returns a cut with zero
 * time delay. Doppler generates a cut with zero Doppler shift. Note that in
 * both Delay and Doppler mode, the output matrix must be a 2D tensor where the
 * first dimension is 1 to match the type of 2D mode.
 * @param cut_val
 *   Value to perform the cut at
 * @param stream
 *   CUDA stream
 *
 */
template <typename T1, typename T2, int RANK>
inline void ambgfun(tensor_t<T2, 2> &amf, tensor_t<T1, RANK> x,
                    double fs, AMBGFunCutType_t cut, float cut_val = 0.0,
                    cudaStream_t stream = 0)
{
  std::optional<tensor_t<T1, RANK>> nil = std::nullopt;
  InternalAmbgFun(amf, x, nil, fs, cut, cut_val, stream);
}
}; // namespace signal
}; // namespace matx
