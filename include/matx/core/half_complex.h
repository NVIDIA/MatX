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
#include <complex>
#include <cuda/std/complex>
#include <cuda/std/cmath>
#include <type_traits>

#include "cuComplex.h"
#include "matx/core/half.h"

namespace matx {

/**
 * Template class for half precison complex numbers (__half and __nv_bfloat16).
 * CUDA does not have standardized classes/operators available on both host and
 * device to support these operations, so we define it here.
 *
 */
template <typename T> struct alignas(sizeof(T) * 2) matxHalfComplex {
  using value_type = T; ///< Type trait to get type

  /**
   * @brief Constructor a half complex object with defaults of zero
   * 
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex() : x(0.0f), y(0.0f) {}

  /**
   * @brief Copy constructor from a complex float
   * 
   * @param x_ Object to copy from
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
  matxHalfComplex(const cuda::std::complex<float> &x_) noexcept
      : x(x_.real()), y(x_.imag())
  {
  }

  /**
   * @brief Copy constructor from scalar value. Sets real type to scalar and imaginary to zero
   * 
   * @tparam T2 Type of scalar
   * @param x_ Value of scalar
   */
  template <typename T2>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex(const T2 &x_) noexcept
      : x(static_cast<float>(x_)), y(0.0f)
  {
  }

  /**
   * @brief Construct a half complex from two scalars
   * 
   * @tparam T2 Real scalar type
   * @tparam T3 Imaginary scalar type
   * @param x_ Real value
   * @param y_ Imaginary value
   */
  template <typename T2, typename T3>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex(const T2 &x_,
                                                      const T3 &y_) noexcept
      : x(static_cast<float>(x_)), y(static_cast<float>(y_))
  {
  }

  /**
   * @brief Construct a half complex from two matx half scalars
   * 
   * @tparam T2 Scalar type
   * @param x_ Real value
   * @param y_ Imaginary value
   */
  template <typename T2, std::enable_if_t<std::is_same_v<std::decay<T2>, matxFp16> || 
                                          std::is_same_v<std::decay<T2>, matxBf16>, bool> = true>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex(T2 &&x_, T2 &&y_) noexcept
      : x(static_cast<T>(x_)), 
        y(static_cast<T>(y_))
  {
  }  

  /**
   * @brief Default destructor
   * 
   */
  __MATX_INLINE__ ~matxHalfComplex() = default;

  /**
   * @brief Cast operator to cuCOmplex
   * 
   * @return cuComplex value
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ operator cuComplex()
  {
    return make_cuFloatComplex(x, y);
  }

  /**
   * @brief cuda::std::complex<float> cast operator
   * 
   * @return cuda::std::complex<float> value
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ operator cuda::std::complex<float>()
  {
    return {x, y};
  }

  /**
   * @brief cuda::std::complex<double> cast operator
   * 
   * @return cuda::std::complex<double> value
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ operator cuda::std::complex<double>()
  {
    return {x, y};
  }

  /**
   * @brief std::complex<float> cast operator
   * 
   * @return std::complex<float> value
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ operator std::complex<float>()
  {
    return {x, y};
  }

  /**
   * @brief std::complex<double> cast operator
   * 
   * @return std::complex<double> value
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ operator std::complex<double>()
  {
    return {x, y};
  }

  /**
   * @brief Copy assignment operator
   * 
   * @tparam X Type of complex to copy from
   * @param rhs Value to copy from
   * @return Reference to copied object
   */
  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T> &
  operator=(const matxHalfComplex<X> &rhs)
  {
    x = rhs.x;
    y = rhs.y;
    return *this;
  }

  /**
   * @brief Copy assignment operator
   * 
   * @tparam X Type of complex to copy from
   * @param rhs Value to copy from
   * @return Reference to copied object
   */
  template <typename X, std::enable_if_t< std::is_same_v<std::decay<X>, cuda::std::complex<float>> || 
                                          std::is_same_v<std::decay<X>, cuda::std::complex<double>>, bool> = true>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T> &
  operator=(X rhs)
  {
    matxHalfComplex<X> tmp{rhs};
    x = static_cast<T>(tmp.real());
    y = static_cast<T>(tmp.imag());
    return *this;
  }  

  /**
   * @brief Increment and assign operator
   * 
   * @tparam X Type of source
   * @param rhs Value of source
   * @return Reference to object
   */
  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T> &
  operator+=(const matxHalfComplex<X> &rhs)
  {
    *this = *this + rhs;
    return *this;
  }

  /**
   * @brief Decrement and assign operator
   * 
   * @tparam X Type of source
   * @param rhs Value of source
   * @return Reference to object
   */
  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T> &
  operator-=(const matxHalfComplex<X> &rhs)
  {
    *this = *this - rhs;
    return *this;
  }

  /**
   * @brief Multiply and assign operator
   * 
   * @tparam X Type of source
   * @param rhs Value of source
   * @return Reference to object
   */
  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T> &
  operator*=(const matxHalfComplex<X> &rhs)
  {
    *this = *this * rhs;
    return *this;
  }

  /**
   * @brief Divide and assign operator
   * 
   * @tparam X Type of source
   * @param rhs Value of source
   * @return Reference to object
   */
  template <typename X>
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T> &
  operator/=(const matxHalfComplex<X> &rhs)
  {
    *this = *this / rhs;
    return *this;
  }

  /**
   * @brief Increment and assign operator for floats
   * 
   * @param f Value of source
   * @return Reference to object
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T> &
  operator+=(float f) const
  {
    return {x + f, y + f};
  }

  /**
   * @brief Decrement and assign operator for floats
   * 
   * @param f Value of source
   * @return Reference to object
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T> &
  operator-=(float f) const
  {
    return {x + f, y + f};
  }

  /**
   * @brief Set real part
   * 
   * @param r Value to set 
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void real(T r) { x = r; }

  /**
   * @brief Set imaginary part
   * 
   * @param i Value to set 
   */  
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void imag(T i) { y = i; }

  /**
   * @brief Get real part
   * 
   * @return Real part
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr T real() const { return x; }

  /**
   * @brief Get imaginary part
   * 
   * @return Imaginary part
   */
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr T imag() const { return y; }

  T x; ///< Real part
  T y; ///< Imaginary part
};

/**
 * @brief Addition operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of addition 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator+(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs + tmp;
}

/**
 * @brief Addition operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of addition 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator+(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return tmp + rhs;
}

/**
 * @brief Subtraction operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator-(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs - tmp;
}

/**
 * @brief Subtraction operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator-(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return tmp - rhs;
}

/**
 * @brief Multiply operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator*(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs * tmp;
}


/**
 * @brief Multiply operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator*(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return tmp * rhs;
}

/**
 * @brief Multiply operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T1, typename T2>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T2>
operator*(const T1 &lhs, const matxHalfComplex<T2> &rhs)
{
  matxHalfComplex<T2> tmp{lhs};
  return tmp * rhs;
}

/**
 * @brief Multiply operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T1, typename T2>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T1>
operator*(const matxHalfComplex<T1> &lhs, const T2 &rhs)
{
  matxHalfComplex<T1> tmp{rhs};
  return lhs * tmp;
}

/**
 * @brief Division operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator/(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs / tmp;
}

/**
 * @brief Division operator
 * 
 * @tparam T RHS type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator/(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return tmp / rhs;
}


/**
 * @brief Negation operator
 * 
 * @tparam T Type of complex
 * @param l Value to negate
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator-(const matxHalfComplex<T> &l)
{
  return {-l.x, -l.y};
}

/**
 * @brief Equality operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool
operator==(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

/**
 * @brief Equality operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool
operator==(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return lhs == tmp;
}

/**
 * @brief Equality operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool
operator==(const T &lhs, const matxHalfComplex<T> &rhs)
{
  matxHalfComplex<T> tmp{lhs};
  return lhs == tmp;
}

/**
 * @brief Not equals operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool
operator!=(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return !(lhs == rhs);
}

/**
 * @brief Not equals operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool
operator!=(const matxHalfComplex<T> &lhs, const T &rhs)
{
  matxHalfComplex<T> tmp{rhs};
  return !(lhs == tmp);
}

/**
 * @brief Not equals operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool
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


/**
 * @brief Addition operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator+(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return {lhs.x + rhs.x, lhs.y + rhs.y};
}

/**
 * @brief Subtraction operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator-(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return {lhs.x - rhs.x, lhs.y - rhs.y};
}


/**
 * @brief Multiplication operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
operator*(const matxHalfComplex<T> &lhs, const matxHalfComplex<T> &rhs)
{
  return {lhs.x * rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x};
}

/**
 * @brief Division operator
 * 
 * @tparam T Underlying type
 * @param lhs LHS value
 * @param rhs RHS value
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
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

/**
 * @brief Conjugate operator
 * 
 * @tparam T Underlying type
 * @param x Value to conjugate
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
conj(const matxHalfComplex<T> &x)
{
  return {x.real(), -static_cast<float>(x.imag())};
}

/**
 * @brief Sbsolute value operator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T abs(const matxHalfComplex<T> &x)
{
#ifdef __CUDA_ARCH__
  return sqrt(x.real() * x.real() + x.imag() * x.imag());
#else
  return static_cast<T>(
      hypot(static_cast<float>(x.real()), static_cast<float>(x.imag())));
#endif
}

/**
 * @brief Argument operator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T arg(const matxHalfComplex<T> &x)
{
  return static_cast<T>(cuda::std::atan2(static_cast<float>(x.real()),
                                         static_cast<float>(x.imag())));
}

/**
 * @brief Arctangent operator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T atan2(const matxHalfComplex<T> &x)
{
  return static_cast<T>(cuda::std::atan2(static_cast<float>(x.imag()),
                                         static_cast<float>(x.real())));
}

/**
 * @brief Arctangent operator
 * 
 * @tparam T Underlying type
 * @param x Denominator input
 * @param y Numerator input
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T atan2(const T &x, const T &y)
{
  return static_cast<T>(
      cuda::std::atan2(static_cast<float>(y), static_cast<float>(x)));
}

/**
 * @brief Norm operator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <typename T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T norm(const matxHalfComplex<T> &x)
{
  if (isinf(x.real()))
    return static_cast<T>(cuda::std::abs(static_cast<float>(x.real())));
  if (isinf(x.imag()))
    return static_cast<T>(cuda::std::abs(static_cast<float>(x.imag())));

  return x.real() * x.real() + x.imag() * x.imag();
}

/**
 * @brief Natural log operator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
log(const matxHalfComplex<T> &x)
{
  return {log(abs(x)), arg(x)};
}


/**
 * @brief Log base 10 operator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
log10(const matxHalfComplex<T> &x)
{
  return log(x) / log(static_cast<T>(10.0f));
}

/**
 * @brief Exponential operator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
exp(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::exp(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Power perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @param y value of exponent
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
pow(const matxHalfComplex<T> &x, const matxHalfComplex<T> &y)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  cuda::std::complex<float> tmp2{static_cast<float>(y.real()),
                                 static_cast<float>(y.imag())};
  tmp = cuda::std::pow(tmp, tmp2);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Power perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @param y value of exponent
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
pow(const matxHalfComplex<T> &x, const T &y)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::pow(tmp, y);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Power perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @param y value of exponent
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
pow(const T &x, const matxHalfComplex<T> &y)
{
  cuda::std::complex<float> tmp{static_cast<float>(y.real()),
                                static_cast<float>(y.imag())};
  tmp = cuda::std::pow(y, pow);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Sine perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
sin(const matxHalfComplex<T> &x)
{
  return sinh(matxHalfComplex<T>{-static_cast<float>(x.imag()), x.real()});
}

/**
 * @brief Cosine perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
cos(const matxHalfComplex<T> &x)
{
  return cosh(matxHalfComplex<T>{-static_cast<float>(x.imag()), x.real()});
}

/**
 * @brief Tangent perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
tan(const matxHalfComplex<T> &x)
{
  return tanh(matxHalfComplex<T>{-static_cast<float>(x.imag()), x.real()});
}

/**
 * @brief Hyperbolic arcsine perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
asinh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::asinh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Hyperbolic arccosine perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
acosh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::acosh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Hyperbolic arctangent perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
atanh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::atanh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Arcsin perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
asin(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::asin(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Arccos perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
acos(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::acos(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Arctangent perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
atan(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::atan(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Hyperbolic sine perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
sinh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::sinh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Hyperbolic cosine perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
cosh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::cosh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

/**
 * @brief Hyperbolic tangent perator
 * 
 * @tparam T Underlying type
 * @param x Value of input
 * @return Result of operation 
 */
template <class T>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ matxHalfComplex<T>
tanh(const matxHalfComplex<T> &x)
{
  cuda::std::complex<float> tmp{static_cast<float>(x.real()),
                                static_cast<float>(x.imag())};
  tmp = cuda::std::tanh(tmp);
  return {static_cast<T>(tmp.real()), static_cast<T>(tmp.imag())};
}

using matxFp16Complex = matxHalfComplex<matxFp16>; ///< Alias for a MatX fp16 complex wrapper
using matxBf16Complex = matxHalfComplex<matxBf16>; ///< Alias for a MatXbf16 complex wrapper

}; // namespace matx
