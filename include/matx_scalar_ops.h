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

#include "matx_scalar_diffs.h"
namespace matx {
namespace detail {

// This file defines operators on a scalar

// Utility macro for generating functions that have half precision intrinsics as
// an option. Lots of verbose code in here because of compiler bugs with
// constexpr if
#define MATX_UNARY_OP_GEN(FUNC, OPNAME, DFUNC)                                 \
  template <typename T> struct OPNAME##F {                                     \
    static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T v1)         \
    {                                                                          \
      return FUNC(v1);                                                         \
    }                                                                          \
    static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto deriv(T v1)      \
    {                                                                          \
      return DFUNC(v1);                                                        \
    }                                                                          \
  };                                                                           \
  template <typename T> using OPNAME##Op = UnOp<T, OPNAME##F<T>>;

template <typename T1, typename F> class UnOp {
public:
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(const T1 &v1) { return F::op(v1); }
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto deriv(const T1 &v1) { return F::deriv(v1); }

  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(const T1 &v1) const { return op(v1); }

  using scalar_type = std::invoke_result_t<decltype(op), T1>;
};

template <typename T1, typename T2, typename F> class BinOp {
public:
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(const T1 &v1, const T2 &v2)
  {
    return F::op(v1, v2);
  }
  
  
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(const T1 &v1, const T2 &v2) const
  {
    return op(v1, v2);
  }
  
  static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto deriv(const T1 &v1, const T2 &v2, const T1 &d1, const T2 &d2) 
  {
    return F::deriv(v1, v2, d1, d2);
  }

  using scalar_type = std::invoke_result_t<decltype(op), T1, T2>;
};

template <typename T1, typename T2, typename T3, typename F> class TerOp {
public:
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(const T1 &v1, const T2 &v2,
                                            const T3 &v3) 
  {
    return F::op(v1, v2, v3);
  }

  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(const T1 &v1, const T2 &v2, const T3 &v3) const
  {
    return op(v1, v2, v3);
  }

  using scalar_type = std::invoke_result_t<decltype(op), T1, T2, T3>;
};


MATX_UNARY_OP_GEN(ceil, Ceil, zero);
MATX_UNARY_OP_GEN(floor, Floor, zero);
MATX_UNARY_OP_GEN(round, Round, zero);
MATX_UNARY_OP_GEN(sqrt, Sqrt, dsqrt);
MATX_UNARY_OP_GEN(exp, Exp, exp);

MATX_UNARY_OP_GEN(log10, Log10, dlog10);
MATX_UNARY_OP_GEN(log2, Log2, dlog2);
MATX_UNARY_OP_GEN(log, Log, dlog);
MATX_UNARY_OP_GEN(abs, Abs, dabs);
MATX_UNARY_OP_GEN(norm, Norm, none);

// Trigonometric functions
MATX_UNARY_OP_GEN(sin, Sin, dsin);
MATX_UNARY_OP_GEN(cos, Cos, dcos);
MATX_UNARY_OP_GEN(tan, Tan, dtan);
MATX_UNARY_OP_GEN(asin, Asin, dasin);
MATX_UNARY_OP_GEN(acos, Acos, dacos);
MATX_UNARY_OP_GEN(atan, Atan, datan);
MATX_UNARY_OP_GEN(sinh, Sinh, dacosh);
MATX_UNARY_OP_GEN(cosh, Cosh, dasinh);
MATX_UNARY_OP_GEN(tanh, Tanh, dtanh);
MATX_UNARY_OP_GEN(asinh, Asinh, dasinh);
MATX_UNARY_OP_GEN(acosh, Acosh, dacosh);
MATX_UNARY_OP_GEN(atanh, Atanh, datanh);

template <typename T> struct ExpjF {
  template <typename T2 = T,
            std::enable_if_t<std::is_floating_point_v<T2>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ ExpjF()
  {
  }

  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T v1)
  {
    return cuda::std::complex<T>{cuda::std::cos(v1), cuda::std::sin(v1)};
  }
};
template <typename T> using ExpjOp = UnOp<T, ExpjF<T>>;

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_conj(T v1)
{
  if constexpr (is_cuda_complex_v<T>) {
    return cuda::std::conj(v1);
  }
  else {
    return conj(v1);
  }
  if constexpr (!is_cuda_complex_v<T>) {
    return conj(v1);
  }
  else {
    return cuda::std::conj(v1);
  }
}
template <typename T> struct ConjF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T v1)
  {
    if constexpr (is_complex_v<T>) {
      return _internal_conj(v1);
    }

    return v1;
  }
};

template <typename T> using ConjOp = UnOp<T, ConjF<T>>;


template <typename T> static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_normcdf(T v1)
{
  return normcdf(v1);
}
template <typename T> struct NormCdfF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T v1) { return _internal_normcdf(v1); }
};
template <typename T> using NormCdfOp = UnOp<T, NormCdfF<T>>;


template <typename T> struct RealF {
  static_assert(is_complex_v<T>, "real() must have complex input");
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T v1) { return v1.real(); }
};
template <typename T> using RealOp = UnOp<T, RealF<T>>;

template <typename T> struct ImagF {
  static_assert(is_complex_v<T>, "imag() must have complex input");
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T v1) { return v1.imag(); }
};
template <typename T> using ImagOp = UnOp<T, ImagF<T>>;

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_angle(T v1)
{
  if constexpr (is_cuda_complex_v<T>) {
    return cuda::std::atan2(v1.imag(), v1.real());
  }
  else {
    return atan2(v1.imag(), v1.real());
  }
  if constexpr (!is_cuda_complex_v<T>) {
    return atan2(v1.imag(), v1.real());
  }
  else {
    return cuda::std::atan2(v1.imag(), v1.real());
  }
}
template <typename T>
struct Angle {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T v1)
  {
    static_assert(is_complex_v<T>, "Angle operator must have complex value as input");
    return _internal_angle(v1);
  }
};
template <typename T> using AngleOp = UnOp<T, Angle<T>>;

template<typename T> 
struct SubNegF {
  static __MATX_INLINE__ __MATX_HOST__  __MATX_DEVICE__ auto op(T v1) 
  { 
    return -v1; 
  }
};
template<typename T> using SubNegOp = UnOp<T,SubNegF<T> >;

// Binary Operators

template <typename T1, typename T2> struct AddF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2)
  {
    if constexpr (is_complex_v<T1> && std::is_arithmetic_v<T2>) {
      if constexpr (is_complex_half_v<T1>) {
        return (T1){v1.real() + static_cast<typename T1::value_type>(
                                    static_cast<float>(v2)),
                    v1.imag() + static_cast<typename T1::value_type>(
                                    static_cast<float>(v2))};
      }
      else {
        return (T1){v1.real() + static_cast<typename T1::value_type>(v2),
                    v1.imag() + static_cast<typename T1::value_type>(v2)};
      }
    }
    else if constexpr (is_complex_v<T2> && std::is_arithmetic_v<T1>) {
      if constexpr (is_complex_half_v<T2>) {
        return (T2){v2.real() + static_cast<typename T2::value_type>(
                                    static_cast<float>(v1)),
                    v2.imag() + static_cast<typename T2::value_type>(
                                    static_cast<float>(v1))};
      }
      else {
        return (T2){v2.real() + static_cast<typename T2::value_type>(v1),
                    v2.imag() + static_cast<typename T2::value_type>(v1)};
      }
    }
    else {
      return v1 + v2;
    }

    // Unreachable, but required by the compiler
    return typename std::invoke_result_t<decltype(op), T1, T2>{0};
  }
  
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto deriv(T1 v1, T2 v2, T1 d1, T2 d2) {
    return dadd(v1, v2, d1, d2); 
  }
};
template <typename T1, typename T2> using AddOp = BinOp<T1, T2, AddF<T1, T2>>;


template <typename T1, typename T2> struct SubF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2)
  {
    if constexpr (is_complex_v<T1> && std::is_arithmetic_v<T2>) {
      if constexpr (is_complex_half_v<T1>) {
        return (T1){v1.real() - static_cast<typename T1::value_type>(
                                    static_cast<float>(v2)),
                    v1.imag() - static_cast<typename T1::value_type>(
                                    static_cast<float>(v2))};
      }
      else {
        return (T1){v1.real() - static_cast<typename T1::value_type>(v2),
                    v1.imag() - static_cast<typename T1::value_type>(v2)};
      }
    }
    else if constexpr (is_complex_v<T2> && std::is_arithmetic_v<T1>) {
      if constexpr (is_complex_half_v<T2>) {
        return (T2){v2.real() - static_cast<typename T2::value_type>(
                                    static_cast<float>(v1)),
                    v2.imag() - static_cast<typename T2::value_type>(
                                    static_cast<float>(v1))};
      }
      else {
        return (T2){v2.real() - static_cast<typename T2::value_type>(v1),
                    v2.imag() - static_cast<typename T2::value_type>(v1)};
      }
    }
    else {
      return v1 - v2;
    }

    // Unreachable, but required by the compiler
    return typename std::invoke_result_t<decltype(op), T1, T2>{0};
  }
  
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto deriv(T1 v1, T2 v2, T1 d1, T2 d2) {
    return dsub(v1, v2, d1, d2);
  }
};
template <typename T1, typename T2> using SubOp = BinOp<T1, T2, SubF<T1, T2>>;

template <typename T1, typename T2> struct MulF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2)
  {
    if constexpr (is_complex_v<T1> && std::is_arithmetic_v<T2>) {
      if constexpr (is_complex_half_v<T1>) {
        return (T1){v1.real() * static_cast<typename T1::value_type>(
                                    static_cast<float>(v2)),
                    v1.imag() * static_cast<typename T1::value_type>(
                                    static_cast<float>(v2))};
      }
      else {
        return (T1){v1.real() * static_cast<typename T1::value_type>(v2),
                    v1.imag() * static_cast<typename T1::value_type>(v2)};
      }
    }
    else if constexpr (is_complex_v<T2> && std::is_arithmetic_v<T1>) {
      if constexpr (is_complex_half_v<T2>) {
        return (T2){v2.real() * static_cast<typename T2::value_type>(
                                    static_cast<float>(v1)),
                    v2.imag() * static_cast<typename T2::value_type>(
                                    static_cast<float>(v1))};
      }
      else {
        return (T2){v2.real() * static_cast<typename T2::value_type>(v1),
                    v2.imag() * static_cast<typename T2::value_type>(v1)};
      }
    }
    else {
      return v1 * v2;
    }

    // Unreachable, but required by the compiler
    return typename std::invoke_result_t<decltype(op), T1, T2>{0};
  }
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto deriv(T1 v1, T2 v2, T1 d1, T2 d2) {
    return dproduct(v1, v2, d1, d2);
  }
};
template <typename T1, typename T2> using MulOp = BinOp<T1, T2, MulF<T1, T2>>;

template <typename T1, typename T2> struct DivF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2)
  {
    if constexpr (is_complex_v<T1> && std::is_arithmetic_v<T2>) {
      if constexpr (is_complex_half_v<T1>) {
        return (T1){v1.real() / static_cast<typename T1::value_type>(
                                    static_cast<float>(v2)),
                    v1.imag() / static_cast<typename T1::value_type>(
                                    static_cast<float>(v2))};
      }
      else {
        return (T1){v1.real() / static_cast<typename T1::value_type>(v2),
                    v1.imag() / static_cast<typename T1::value_type>(v2)};
      }
    }
    else if constexpr (is_complex_v<T2> && std::is_arithmetic_v<T1>) {
      if constexpr (is_complex_half_v<T2>) {
        return (T2){v2.real() / static_cast<typename T2::value_type>(
                                    static_cast<float>(v1)),
                    v2.imag() / static_cast<typename T2::value_type>(
                                    static_cast<float>(v1))};
      }
      else {
        return (T2){v2.real() / static_cast<typename T2::value_type>(v1),
                    v2.imag() / static_cast<typename T2::value_type>(v1)};
      }
    }
    else {
      return v1 / v2;
    }

    // Unreachable, but required by the compiler
    return typename std::invoke_result_t<decltype(op), T1, T2>{0};
  }
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto deriv(T1 v1, T2 v2, T1 d1, T2 d2) {
      return dquotient(v1,v2,d1,d2);
  }
};
template <typename T1, typename T2> using DivOp = BinOp<T1, T2, DivF<T1, T2>>;

template <typename T1, typename T2> struct ModF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 % v2; }
};
template <typename T1, typename T2> using ModOp = BinOp<T1, T2, ModF<T1, T2>>;

template <typename T1, typename T2>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_pow(T1 v1, T2 v2)
{
  if constexpr (is_matx_type_v<T1>) {
    return pow(v1, v2);
  }
  else {
    return cuda::std::pow(v1, v2);
  }
  if constexpr (!is_matx_type_v<T1>) { /* Compiler bug WAR */
    return cuda::std::pow(v1, v2);
  }
  else {
    return pow(v1, v2);
  }
}

template <typename T1, typename T2> struct PowF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2)
  {
    return _internal_pow(v1, v2);
  }
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto deriv(T1 v1, T2 v2, T1 d1, T2 d2)
  {

    return dpow(v1, v2, d1, d2);
  }
};
template <typename T1, typename T2> using PowOp = BinOp<T1, T2, PowF<T1, T2>>;

template <typename T1, typename T2> struct MaxF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2)
  {
    return max(v1, v2);
  }
};
template <typename T1, typename T2> using MaxOp = BinOp<T1, T2, MaxF<T1, T2>>;

template <typename T1, typename T2> struct MinF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2)
  {
    return min(v1, v2);
  }
};
template <typename T1, typename T2> using MinOp = BinOp<T1, T2, MinF<T1, T2>>;

// Logical Operators
template <typename T1, typename T2> struct LTF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 < v2; }
};
template <typename T1, typename T2> using LTOp = BinOp<T1, T2, LTF<T1, T2>>;

template <typename T1, typename T2> struct GTF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 > v2; }
};
template <typename T1, typename T2> using GTOp = BinOp<T1, T2, GTF<T1, T2>>;

template <typename T1, typename T2> struct LTEF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 <= v2; }
};
template <typename T1, typename T2> using LTEOp = BinOp<T1, T2, LTEF<T1, T2>>;

template <typename T1, typename T2> struct GTEF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 >= v2; }
};
template <typename T1, typename T2> using GTEOp = BinOp<T1, T2, GTEF<T1, T2>>;

template <typename T1, typename T2> struct EQF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 == v2; }
};
template <typename T1, typename T2> using EQOp = BinOp<T1, T2, EQF<T1, T2>>;

template <typename T1, typename T2> struct NEF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 != v2; }
};
template <typename T1, typename T2> using NEOp = BinOp<T1, T2, NEF<T1, T2>>;

template <typename T1, typename T2> struct AndAndF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 && v2; }
};
template <typename T1, typename T2> using AndAndOp = BinOp<T1, T2, AndAndF<T1, T2>>;

template <typename T1, typename T2> struct OrOrF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 || v2; }
};
template <typename T1, typename T2> using OrOrOp = BinOp<T1, T2, OrOrF<T1, T2>>;

template <typename T1> struct NotF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1) { return !v1; }
};
template <typename T1> using NotOp = UnOp<T1, NotF<T1>>;

template <typename T1, typename T2> struct AndF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 & v2; }
};
template <typename T1, typename T2> using AndOp = BinOp<T1, T2, AndF<T1, T2>>;

template <typename T1, typename T2> struct OrF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 | v2; }
};
template <typename T1, typename T2> using OrOp = BinOp<T1, T2, OrF<T1, T2>>;

template <typename T1, typename T2> struct XorF {
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1 v1, T2 v2) { return v1 ^ v2; }
};
template <typename T1, typename T2> using XorOp = BinOp<T1, T2, XorF<T1, T2>>;

} // end namespace detail
} // end namespace matx
