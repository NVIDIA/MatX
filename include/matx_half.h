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

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <type_traits>

namespace matx {

/**
 * Template class for half precison numbers (__half and __nv_bfloat16). CUDA
 * does not have standardized classes/operators available on both host and
 * device to support these operations, so we define it here.
 *
 */
template <typename T> struct alignas(sizeof(T)) matxHalf {
  using value_type = T;

  __host__ __device__ __forceinline__ matxHalf() : x(0.0f) {}
  __host__ __device__ __forceinline__ matxHalf(const matxHalf<T> &x_) noexcept
      : x(x_.x)
  {
  }

  template <typename T2>
  __host__ __device__ __forceinline__ matxHalf(const T2 &x_) noexcept
      : x(static_cast<float>(x_))
  {
  }
  __forceinline__ ~matxHalf() = default;

  __host__ __device__ __forceinline__ operator float() const
  {
    return static_cast<float>(x);
  }
  __host__ __device__ __forceinline__ operator double() const
  {
    return static_cast<double>(x);
  }
  // __host__ __device__ __forceinline__ operator const float() const { return
  // static_cast<float>(x); }
  // __host__ __device__ __forceinline__ operator const double() const { return
  // static_cast<float>(x); }

  __host__ __device__ __forceinline__ matxHalf<T> &
  operator=(const matxHalf<T> &rhs)
  {
    x = rhs.x;
    return *this;
  }

  template <typename T2>
  __host__ __device__ __forceinline__ matxHalf<T> &operator=(const T2 &rhs)
  {
    x = static_cast<float>(rhs);
    return *this;
  }

  template <typename X>
  __host__ __device__ __forceinline__ matxHalf<T> &
  operator+=(const matxHalf<X> &rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  template <typename X>
  __host__ __device__ __forceinline__ matxHalf<T> &
  operator-=(const matxHalf<X> &rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  template <typename X>
  __host__ __device__ __forceinline__ matxHalf<T> &
  operator*=(const matxHalf<X> &rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  template <typename X>
  __host__ __device__ __forceinline__ matxHalf<T> &
  operator/=(const matxHalf<X> &rhs)
  {
    *this = *this / rhs;
    return *this;
  }

  __host__ __device__ __forceinline__ matxHalf<T> &operator+=(float f) const
  {
#ifdef __CUDA_ARCH__
    return {x + f};
#else
    return {static_cast<T>(static_cast<float>(x) + f)};
#endif
  }

  __host__ __device__ __forceinline__ matxHalf<T> &operator-=(float f) const
  {
#ifdef __CUDA_ARCH__
    return {x + f};
#else
    return {static_cast<T>(static_cast<float>(x) + f)};
#endif
  }

  T x;
};

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator+(const matxHalf<T> &lhs, const T &rhs)
{
  matxHalf<T> tmp{rhs};
  return lhs + rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator+(const T &lhs, const matxHalf<T> &rhs)
{
  matxHalf<T> tmp{lhs};
  return lhs + rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator-(const matxHalf<T> &lhs, const T &rhs)
{
  matxHalf<T> tmp{rhs};
  return lhs - rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator-(const T &lhs, const matxHalf<T> &rhs)
{
  matxHalf<T> tmp{lhs};
  return lhs - rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator*(const matxHalf<T> &lhs, const T &rhs)
{
  matxHalf<T> tmp{rhs};
  return lhs * rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator*(const T &lhs, const matxHalf<T> &rhs)
{
  matxHalf<T> tmp{lhs};
  return lhs * rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator/(const matxHalf<T> &lhs, const T &rhs)
{
  matxHalf<T> tmp{rhs};
  return lhs / rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator/(const T &lhs, const matxHalf<T> &rhs)
{
  matxHalf<T> tmp{lhs};
  return lhs / rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T> operator-(const matxHalf<T> &l)
{
  return {-l.x};
}

template <typename T>
__host__ __device__ __forceinline__ bool operator==(const matxHalf<T> &lhs,
                                                    const matxHalf<T> &rhs)
{
#ifdef __CUDA_ARCH__
  return lhs.x == rhs.x;
#else
  return static_cast<float>(lhs.x) == static_cast<float>(rhs.x);
#endif
}

template <typename T>
__host__ __device__ __forceinline__ bool operator==(const matxHalf<T> &lhs,
                                                    const T &rhs)
{
  matxHalf<T> tmp{rhs};
  return lhs == tmp;
}

template <typename T>
__host__ __device__ __forceinline__ bool operator==(const T &lhs,
                                                    const matxHalf<T> &rhs)
{
  matxHalf<T> tmp{lhs};
  return lhs == tmp;
}

template <typename T>
__host__ __device__ __forceinline__ bool operator!=(const matxHalf<T> &lhs,
                                                    const matxHalf<T> &rhs)
{
  return !(lhs == rhs);
}

template <typename T>
__host__ __device__ __forceinline__ bool operator!=(const matxHalf<T> &lhs,
                                                    const T &rhs)
{
  matxHalf<T> tmp{rhs};
  return !(lhs == tmp);
}

template <typename T>
__host__ __device__ __forceinline__ bool operator!=(const T &lhs,
                                                    const matxHalf<T> &rhs)
{
  matxHalf<T> tmp{lhs};
  return !(lhs == tmp);
}

template <typename T>
__host__ __device__ __forceinline__ bool operator>(const matxHalf<T> &lhs,
                                                   const matxHalf<T> &rhs)
{
#ifdef __CUDA_ARCH__
  return lhs.x > rhs.x;
#else
  return static_cast<float>(lhs.x) > static_cast<float>(rhs.x);
#endif
}

template <typename T>
__host__ __device__ __forceinline__ bool operator<(const matxHalf<T> &lhs,
                                                   const matxHalf<T> &rhs)
{
#ifdef __CUDA_ARCH__
  return lhs.x < rhs.x;
#else
  return static_cast<float>(lhs.x) < static_cast<float>(rhs.x);
#endif
}

template <typename T>
__host__ __device__ __forceinline__ bool operator<=(const matxHalf<T> &lhs,
                                                    const matxHalf<T> &rhs)
{
  return lhs < rhs || lhs == rhs;
}

template <typename T>
__host__ __device__ __forceinline__ bool operator>=(const matxHalf<T> &lhs,
                                                    const matxHalf<T> &rhs)
{
  return lhs > rhs || lhs == rhs;
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator+(const matxHalf<T> &lhs, const matxHalf<T> &rhs)
{
#ifdef __CUDA_ARCH__
  return lhs.x + rhs.x;
#else
  return static_cast<T>(static_cast<float>(lhs) + static_cast<float>(rhs));
#endif
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator-(const matxHalf<T> &lhs, const matxHalf<T> &rhs)
{
#ifdef __CUDA_ARCH__
  return lhs.x - rhs.x;
#else
  return static_cast<T>(static_cast<float>(lhs) - static_cast<float>(rhs));
#endif
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator*(const matxHalf<T> &lhs, const matxHalf<T> &rhs)
{
#ifdef __CUDA_ARCH__
  return lhs.x * rhs.x;
#else
  return static_cast<T>(static_cast<float>(lhs.x) * static_cast<float>(rhs.x));
#endif
}

template <typename T>
__host__ __device__ __forceinline__ matxHalf<T>
operator/(const matxHalf<T> &lhs, const matxHalf<T> &rhs)
{
#ifdef __CUDA_ARCH__
  return lhs.x / rhs.x;
#else
  return static_cast<T>(static_cast<float>(lhs.x) / static_cast<float>(rhs.x));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> abs(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return __habs(x.x);
#else
  return static_cast<T>(cuda::std::abs(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
abs(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return __habs(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::abs(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> log(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hlog(x.x);
#else
  return static_cast<T>(cuda::std::log(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> sqrt(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hsqrt(x.x);
#else
  return static_cast<T>(cuda::std::sqrt(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
sqrt(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hsqrt(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::sqrt(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ int isinf(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return __hisinf(x.x);
#else
  return static_cast<int>(cuda::std::isinf(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ int isinf(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return __hisinf(x.x);
#else
  return static_cast<int>(cuda::std::isinf(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
log(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hlog(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::log(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> log10(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hlog10(x.x);
#else
  return static_cast<T>(cuda::std::log10(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
log10(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hlog10(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::log10(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> log2(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hlog2(x.x);
#else
  return static_cast<T>(cuda::std::log2(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
log2(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hlog2(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::log2(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> exp(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hexp(x.x);
#else
  return static_cast<T>(cuda::std::exp(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
exp(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hexp(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::exp(static_cast<float>(x.x)));
#endif
}

// Trig functions
template <class T>
__host__ __device__ __forceinline__ matxHalf<T> pow(const matxHalf<T> &x,
                                                    const matxHalf<T> &y)
{
  auto tmp = cuda::std::pow(static_cast<float>(x.x), static_cast<float>(y.x));
  return static_cast<T>(tmp);
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> pow(const matxHalf<T> &x,
                                                    const T &y)
{
  auto tmp = cuda::std::pow(static_cast<float>(x.x), static_cast<float>(y));
  return static_cast<T>(tmp);
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> pow(const T &x,
                                                    const matxHalf<T> &y)
{
  auto tmp = cuda::std::pow(x, static_cast<float>(y.x));
  return static_cast<T>(tmp);
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> floor(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hfloor(x.x);
#else
  return static_cast<T>(cuda::std::floor(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
floor(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hfloor(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::floor(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> ceil(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hceil(x.x);
#else
  return static_cast<T>(cuda::std::ceil(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
ceil(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hceil(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::ceil(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> round(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hrint(x.x);
#else
  return static_cast<T>(cuda::std::round(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
round(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hrint(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::round(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> sin(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hsin(x.x);
#else
  return static_cast<T>(cuda::std::sin(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
sin(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hsin(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::sin(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> cos(const matxHalf<T> &x)
{
#ifdef __CUDA_ARCH__
  return hcos(x.x);
#else
  return static_cast<T>(cuda::std::cos(static_cast<float>(x.x)));
#endif
}

template <>
__host__ __device__ __forceinline__ matxHalf<__nv_bfloat16>
cos(const matxHalf<__nv_bfloat16> &x)
{
#if __CUDA_ARCH__ >= 800
  return hcos(x.x);
#else
  return static_cast<__nv_bfloat16>(cuda::std::cos(static_cast<float>(x.x)));
#endif
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> tan(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::tan(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> asin(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::asin(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> acos(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::acos(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> atan(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::atan(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> asinh(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::asinh(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> acosh(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::acosh(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> atanh(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::atanh(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> sinh(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::sinh(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> cosh(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::cosh(static_cast<float>(x.x)));
}

template <class T>
__host__ __device__ __forceinline__ matxHalf<T> tanh(const matxHalf<T> &x)
{
  return static_cast<T>(cuda::std::tanh(static_cast<float>(x.x)));
}

using matxFp16 = matxHalf<__half>;
using matxBf16 = matxHalf<__nv_bfloat16>;

}; // namespace matx