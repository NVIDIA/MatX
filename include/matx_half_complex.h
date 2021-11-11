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
// Next line is a workaround for libcuda++ 11.6.0 on CTK 11.4.100+. Remove once libcuda++ is fixed
#define _LIBCUDACXX_HAS_NO_INT128
#include "cuComplex.h"
#include "matx_half.h"
#include <complex>
#include <cuda/std/complex>
#include <cuda/std/cmath>
#include <type_traits>

namespace matx {

/**
 * Template class for half precison complex numbers (__half and __nv_bfloat16).
 * CUDA does not have standardized classes/operators available on both host and
 * device to support these operations, so we define it here.
 *
 */
template <typename T> struct alignas(sizeof(T) * 2) matxHalfComplex {
  using value_type = T;

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex() : x(0.0f), y(0.0f) {}

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__
  matxHalfComplex(const cuda::std::complex<float> &x_) noexcept
      : x(x_.real()), y(x_.imag())
  {
  }

  template <typename T2>
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex(const T2 &x_) noexcept
      : x(static_cast<float>(x_)), y(0.0f)
  {
  }

  template <typename T2, typename T3>
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex(const T2 &x_,
                                                      const T3 &y_) noexcept
      : x(static_cast<float>(x_)), y(static_cast<float>(y_))
  {
  }

  __forceinline__ ~matxHalfComplex() = default;

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ operator cuComplex()
  {
    return make_cuFloatComplex(x, y);
  }

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ operator cuda::std::complex<float>()
  {
    return {x, y};
  }

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ operator cuda::std::complex<double>()
  {
    return {x, y};
  }

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ operator std::complex<float>()
  {
    return {x, y};
  }

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ operator std::complex<double>()
  {
    return {x, y};
  }

  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T> &
  operator=(const matxHalfComplex<X> &rhs)
  {
    x = rhs.x;
    y = rhs.y;
    return *this;
  }

  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T> &
  operator+=(const matxHalfComplex<X> &rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T> &
  operator-=(const matxHalfComplex<X> &rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T> &
  operator*=(const matxHalfComplex<X> &rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T> &
  operator/=(const matxHalfComplex<X> &rhs)
  {
    *this = *this / rhs;
    return *this;
  }

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T> &
  operator+=(float f) const
  {
    return {x + f, y + f};
  }

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T> &
  operator-=(float f) const
  {
    return {x + f, y + f};
  }

  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ void real(T r) { x = r; }
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ void imag(T i) { y = i; }
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ constexpr T real() const { return x; }
  __MATX_HOST__ __MATX_DEVICE__ __forceinline__ constexpr T imag() const { return y; }

  T x;
  T y;
};

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator+(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs + rhs;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator+(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return lhs + rhs;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator-(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs - rhs;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator-(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return lhs - rhs;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator*(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs * rhs;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator*(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return lhs * rhs;
}

template <typename T1, typename T2>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T2>
operator*(const T1 &lhs, const matxHalfComplex<T2> &rhs)
{
  matxHalfComplex<T2> tmp{lhs};
  return lhs * rhs;
}

template <typename T1, typename T2>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T1>
operator*(const matxHalfComplex<T1> &lhs, const T2 &rhs)
{
  matxHalfComplex<T1> tmp{rhs};
  return lhs * rhs;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator/(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs / rhs;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator/(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return lhs / rhs;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator-(const matxHalfComplex<T> &l)
{
  return {-l.x, -l.y};
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ bool
operator==(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ bool
operator==(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs == tmp;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ bool
operator==(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return lhs == tmp;
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ bool
operator!=(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return !(lhs == rhs);
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ bool
operator!=(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return !(lhs == tmp);
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ bool
operator!=(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return !(lhs == tmp);
}

template <typename T>
bool operator>(const matxHalfComplex<T> &lhs,
               const matxHalfComplex<T> &rhs) = delete;
template <typename T>
bool operator<(const matxHalfComplex<T> &lhs,
               const matxHalfComplex<T> &rhs) = delete;
template <typename T>
bool operator>=(const matxHalfComplex<T> &lhs,
                const matxHalfComplex<T> &rhs) = delete;
template <typename T>
bool operator<=(const matxHalfComplex<T> &lhs,
                const matxHalfComplex<T> &rhs) = delete;

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator+(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return {lhs.x + rhs.x, lhs.y + rhs.y};
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator-(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return {lhs.x - rhs.x, lhs.y - rhs.y};
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator*(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return {lhs.x * rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x};
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
operator/(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
#ifdef __CUDA_ARCH__
  T s = abs(rhs.x) + abs(rhs.y);
  T oos = T{1.0f} / s;
  T ars = lhs.x * oos;
  T ais = lhs.y * oos;
  T brs = rhs.x * oos;
  T bis = rhs.y * oos;
  s = (brs * brs) + (bis * bis);
  oos = T{1.0f} / s;
  return {((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos};
#else
  float s = fabs(static_cast<float>(rhs.x)) + fabs(static_cast<float>(rhs.y));
  float oos = 1.0f / s;
  float ars = static_cast<float>(lhs.x) * oos;
  float ais = static_cast<float>(lhs.y) * oos;
  float brs = static_cast<float>(rhs.x) * oos;
  float bis = static_cast<float>(rhs.y) * oos;
  s = (brs * brs) + (bis * bis);
  oos = 1.0f / s;
  return {static_cast<T>(((ars * brs) + (ais * bis)) * oos),
          static_cast<T>(((ais * brs) - (ars * bis)) * oos)};
#endif
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
conj(const matxHalfComplex<T> &x)
{
  return {x.real(), -static_cast<float>(x.imag())};
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ T abs(const matxHalfComplex<T> &x)
{
#ifdef __CUDA_ARCH__
  return sqrt(x.real() * x.real() + x.imag() * x.imag());
#else
  return static_cast<T>(
      hypot(static_cast<float>(x.real()), static_cast<float>(x.imag())));
#endif
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ T arg(const matxHalfComplex<T> &x)
{
  return static_cast<T>(cuda::std::atan2(static_cast<float>(x.real()),
                                         static_cast<float>(x.imag())));
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ T atan2(const matxHalfComplex<T> &x)
{
  return static_cast<T>(cuda::std::atan2(static_cast<float>(x.imag()),
                                         static_cast<float>(x.real())));
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ T atan2(const T &x, const T &y)
{
  return static_cast<T>(
      cuda::std::atan2(static_cast<float>(y), static_cast<float>(x)));
}

template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ T norm(const matxHalfComplex<T> &x)
{
  if (isinf(x.real()))
    return static_cast<T>(cuda::std::abs(static_cast<float>(x.real())));
  if (isinf(x.imag()))
    return static_cast<T>(cuda::std::abs(static_cast<float>(x.imag())));

  return x.real() * x.real() + x.imag() * x.imag();
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
log(const matxHalfComplex<T> &x)
{
  return {log(abs(x)), arg(x)};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
log10(const matxHalfComplex<T> &x)
{
  return log(x) / log(static_cast<T>(10.0f));
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
exp(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::exp(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
pow(const matxHalfComplex<T> &x, const matxHalfComplex<T> &y)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  cuda::std::complex<float> tmp2{static_cast<float>(y.real()),
                                 static_cast<float>(y.imag())};
  tmp = cuda::std::pow(tmp, tmp2);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
pow(const matxHalfComplex<T> &x, const T &y)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::pow(tmp, y);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
pow(const T &x, const matxHalfComplex<T> &y)
{
  cuda::std::complex<float> tmp{static_cast<float>(y.real()),
                                static_cast<float>(y.imag())};
  tmp = cuda::std::pow(y, pow);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

// Trig functions
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
sin(const matxHalfComplex<T> &x)
{
  return sinh(matxHalfComplex<T>{-static_cast<float>(x.imag()), x.real()});
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
cos(const matxHalfComplex<T> &x)
{
  return cosh(matxHalfComplex<T>{-static_cast<float>(x.imag()), x.real()});
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
tan(const matxHalfComplex<T> &x)
{
  return tanh(matxHalfComplex<T>{-static_cast<float>(x.imag()), x.real()});
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
asinh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::asinh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
acosh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::acosh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
atanh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::atanh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
asin(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::asin(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
acos(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::acos(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
atan(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::atan(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
sinh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::sinh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
cosh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::cosh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

template <class T>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ matxHalfComplex<T>
tanh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::tanh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

using matxFp16Complex = matxHalfComplex<matxFp16>;
using matxBf16Complex = matxHalfComplex<matxBf16>;

}; // namespace matx