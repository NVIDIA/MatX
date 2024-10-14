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

#include <type_traits>
#include <cuda/std/cmath>

namespace matx {
namespace detail {

// This file defines operators on a scalar

// Utility macro for generating functions that have half precision intrinsics as
// an option. Lots of verbose code in here because of compiler bugs with
// constexpr if
#define MATX_UNARY_OP_GEN(FUNC, OPNAME)                                        \
  template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>          \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_##FUNC(T v1)                \
  {                                                                            \
    if constexpr (is_matx_type_v<T>) {                                         \
      const auto UnFunc = [&](auto p1) {                                       \
        return FUNC(p1);                                                       \
      };                                                                       \
                                                                               \
      return UnaryVecFunc<InWidth>(UnFunc, v1);                                \
    }                                                                          \
    else {                                                                     \
      const auto UnFunc = [&](auto p1) {                                       \
        return cuda::std::FUNC(p1);                                            \
      };                                                                       \
                                                                               \
      return UnaryVecFunc<InWidth>(UnFunc, v1);                                \
    }                                                                          \
  }                                                                            \
  template <typename T> struct OPNAME##F {                                     \
    static __MATX_INLINE__ std::string str() { return #FUNC; }                 \
                                                                               \
    template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V> \
    static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)       \
    {                                                                          \
      return _internal_##FUNC<T1V, InWidth>(v1);                               \
    }                                                                          \
  };                                                                           \
  template <typename T> using OPNAME##Op = UnOp<T, OPNAME##F<T>>;

#define MATX_BINARY_OP_GEN(FUNC, OPNAME)                                       \
  template <typename T1, typename T2, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_##FUNC(T1 v1, T2 v2)        \
  {                                                                            \
    if constexpr (is_matx_type_v<T1> || is_matx_type_v<T2>) {                  \
      const auto BinFunc = [&](auto p1, auto p2) {                             \
        return FUNC(p1, p2);                                                   \
      };                                                                       \
                                                                               \
      return BinVecFunc<InWidth>(BinFunc, v1, v2);                             \
    }                                                                          \
    else {                                                                     \
      const auto BinFunc = [&](auto p1, auto p2) {                             \
        return cuda::std::FUNC(p1, p2);                                        \
      };                                                                       \
                                                                               \
      return BinVecFunc<InWidth>(BinFunc, v1, v2);                             \
    }                                                                          \
  }                                                                            \
  template <typename T> struct OPNAME##F {                                     \
    template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V> \
    static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)  \
    {                                                                          \
      return _internal_##FUNC<T1V, T2V, InWidth>(v1, v2);                      \
    }                                                                          \
  };                                                                           \
  template <typename T1, typename T2>                                          \
  using OPNAME##Op = BinOp<T1, T2, OPNAME##F<T1, T2>>;



// Helper function to apply a callable binary operator onto two inputs. There are many compile-time
// branches in here because we need to handle both scalar and vector inputs on both sides.
template <matx::detail::VecWidth InWidth, typename BinOpFunc, typename T1, typename T2>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto BinVecFunc(const BinOpFunc &func, const T1 &v1, const T2 &v2) {
  if constexpr (InWidth == VecWidth::SCALAR) {
    return func(v1, v2);
  }
  else if constexpr (InWidth == VecWidth::ONE) {
    if constexpr (is_vector_v<T1>) {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2.data[0])), 1>;
        return res_type{func(v1.data[0], v2.data[0])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2)), 1>;
        return res_type{func(v1.data[0], v2)};
      }
    }
    else {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1, v2.data[0])), 1>;
        return res_type{func(v1, v2.data[0])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1, v2)), 1>;
        return res_type{func(v1, v2)};
      }
    }
  }
  else if constexpr (InWidth == VecWidth::TWO) {
    if constexpr (is_vector_v<T1>) {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2.data[0])), 2>;
        return res_type{  func(v1.data[0], v2.data[0]),
                          func(v1.data[1], v2.data[1])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2)), 2>;
        return res_type{  func(v1.data[0], v2),
                          func(v1.data[1], v2)};
      }
    }
    else {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1, v2.data[0])), 2>;
        return res_type{  func(v1, v2.data[0]),
                          func(v1, v2.data[1])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1, v2)), 2>;
        const auto val = func(v1, v2);
        return res_type{val, val};
      }
    }
  }
  else if constexpr (InWidth == VecWidth::FOUR) {
    if constexpr (is_vector_v<T1>) {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2.data[0])), 4>;
        return res_type{  func(v1.data[0], v2.data[0]),
                          func(v1.data[1], v2.data[1]),
                          func(v1.data[2], v2.data[2]),
                          func(v1.data[3], v2.data[3])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1.data[0], v2)), 4>;
        return res_type{  func(v1.data[0], v2),
                          func(v1.data[1], v2),
                          func(v1.data[2], v2),
                          func(v1.data[3], v2)};
      }
    }
    else {
      if constexpr (is_vector_v<T2>) {
        using res_type = matx::detail::Vector<decltype(func(v1, v2.data[0])), 4>;
        return res_type{  func(v1, v2.data[0]),
                          func(v1, v2.data[1]),
                          func(v1, v2.data[2]),
                          func(v1, v2.data[3])};
      }
      else {
        using res_type = matx::detail::Vector<decltype(func(v1, v2)), 4>;
        const auto val = func(v1, v2);
        return res_type{val, val, val, val};
      }
    }
  }
}

template <matx::detail::VecWidth InWidth, typename UnaryOpFunc, typename T1>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto UnaryVecFunc(const UnaryOpFunc &func, const T1 &v1) {
  if constexpr (InWidth == VecWidth::SCALAR) {
    return func(v1);
  }
  else if constexpr (InWidth == VecWidth::ONE) {
    if constexpr (is_vector_v<T1>) {
      using res_type = matx::detail::Vector<decltype(func(v1.data[0])), 1>;
      return res_type{func(v1.data[0])};
    }
    else {
      using res_type = matx::detail::Vector<decltype(func(v1)), 1>;
      return res_type{func(v1)};
    }
  }
  else if constexpr (InWidth == VecWidth::TWO) {
    if constexpr (is_vector_v<T1>) {
      using res_type = matx::detail::Vector<decltype(func(v1.data[0])), 2>;
      return res_type{  func(v1.data[0]),
                        func(v1.data[1])};
    }
    else {
      using res_type = matx::detail::Vector<decltype(func(v1)), 2>;
      const auto ret = func(v1);
      return res_type{  ret,
                        ret};
    }
  }
  else if constexpr (InWidth == VecWidth::FOUR) {
    if constexpr (is_vector_v<T1>) {
      using res_type = matx::detail::Vector<decltype(func(v1.data[0])), 4>;
      return res_type{  func(v1.data[0]),
                        func(v1.data[1]),
                        func(v1.data[2]),
                        func(v1.data[3])};
    }
    else {
      using res_type = matx::detail::Vector<decltype(func(v1)), 4>;
      const auto ret = func(v1);
      return res_type{ret, ret, ret, ret};
    }
  }
}  

template <typename T1, typename F> class UnOp {
public:
  __MATX_INLINE__ static const std::string str() { return F::str(); }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(const T1V &v1) {
    return F::template op<InWidth, OutWidth, T1V>(v1);
  }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1V &v1) const {
    return op<InWidth, OutWidth, T1V>(v1);
  }

  using value_type = std::invoke_result_t<
    decltype(op<matx::detail::VecWidth::SCALAR, matx::detail::VecWidth::SCALAR, T1>),  T1>;
};

template <typename T1, typename T2, typename F> class BinOp {
public:
  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1S, typename T2S>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(const T1S &v1, const T2S &v2)
  {
    return F::template op<InWidth, OutWidth, T1S, T2S>(v1, v2);
  }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1S, typename T2S>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1S &v1, const T2S &v2) const
  {
    return op<InWidth, OutWidth, T1S, T2S>(v1, v2);
  }

  // Get type that our operator would return
  using value_type = std::invoke_result_t<
    decltype(op<matx::detail::VecWidth::SCALAR, matx::detail::VecWidth::SCALAR, T1, T2>),  T1, T2>;
};



MATX_UNARY_OP_GEN(ceil, Ceil);
MATX_UNARY_OP_GEN(floor, Floor);
MATX_UNARY_OP_GEN(round, Round);
MATX_UNARY_OP_GEN(exp, Exp);
MATX_UNARY_OP_GEN(sqrt, Sqrt);


template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_rsqrt(T v1)
{
  const auto UnFunc = [&](auto p1) {
    if constexpr (is_matx_type_v<T>){
      return rsqrt(p1);
    }
    else {
  #ifdef __CUDACC__
      return ::rsqrt(v1);
  #else
      return static_cast<T>(1) / sqrt(p1);
  #endif
    }
  };

  return UnaryVecFunc<InWidth>(UnFunc, v1);  
}
template <typename T> struct RSqrtF {
  static __MATX_INLINE__ std::string str() { return "rsqrt"; }
  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)
  {
    return _internal_rsqrt<T1V, InWidth>(v1);
  }
};

template <typename T> using RSqrtOp = UnOp<T, RSqrtF<T>>;


template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_csqrt(T v1)
{
  static_assert(std::is_floating_point_v<T>, "csqrt() only supports non-complex floating point inputs");
  const auto UnFunc = [&](auto p1) {
    return sqrt(static_cast<cuda::std::complex<T>>(p1));
  };

  return UnaryVecFunc<InWidth>(UnFunc, v1);
}

template <typename T> struct CSqrtF {
  static __MATX_INLINE__ std::string str() { return "csqrt"; }
  
  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)
  {
    return _internal_csqrt<T1V, InWidth>(v1);
  }
};


template <typename T> using CsqrtOp = UnOp<T, CSqrtF<T>>;

template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_conj(T v1)
{
  if constexpr (is_cuda_complex_v<T>) {
    const auto UnFunc = [&](auto p1) {
      return cuda::std::conj(p1);
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
  else {
    const auto UnFunc = [&](auto p1) {
      return conj(p1);
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
}
template <typename T> struct ConjF {
  static __MATX_INLINE__ std::string str() { return "conj"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)
  {
    if constexpr (is_complex_v<T>) {
      return _internal_conj<T1V, InWidth>(v1);
    }
    else {
      const auto UnFunc = [&](auto p1) {
        return p1;
      };

      return UnaryVecFunc<InWidth>(UnFunc, v1);
    }
  }
};

template <typename T> using ConjOp = UnOp<T, ConjF<T>>;

MATX_UNARY_OP_GEN(log10, Log10);
MATX_UNARY_OP_GEN(log2, Log2);
MATX_UNARY_OP_GEN(log, Log);
MATX_UNARY_OP_GEN(abs, Abs);



// Trigonometric functions
template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_sin(T v1) {
  if constexpr (is_matx_type_v<T>) {
    const auto UnFunc = [&](auto p1) {
      return matx::sin(p1);
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
  else {
    const auto UnFunc = [&](auto p1) {
      return cuda::std::sin(p1);
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
}
template <typename T> struct SinF {
  static __MATX_INLINE__ std::string str() { return "sin"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1) { return _internal_sin<T1V, InWidth>(v1); }
};
template <typename T> using SinOp = UnOp<T, SinF<T>>;

template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_cos(T v1) {
  if constexpr (is_matx_type_v<T>) {
    const auto UnFunc = [&](auto p1) {
      return matx::cos(p1);
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
  else {
    const auto UnFunc = [&](auto p1) {
      return cuda::std::cos(p1);
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
}
template <typename T> struct CosF {
  static __MATX_INLINE__ std::string str() { return "cos"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1) { return _internal_cos<T1V, InWidth>(v1); }
};
template <typename T> using CosOp = UnOp<T, CosF<T>>;


MATX_UNARY_OP_GEN(tan, Tan);
MATX_UNARY_OP_GEN(asin, Asin);
MATX_UNARY_OP_GEN(acos, Acos);
MATX_UNARY_OP_GEN(atan, Atan);
MATX_UNARY_OP_GEN(sinh, Sinh);
MATX_UNARY_OP_GEN(cosh, Cosh);
MATX_UNARY_OP_GEN(tanh, Tanh);
MATX_UNARY_OP_GEN(asinh, Asinh);
MATX_UNARY_OP_GEN(acosh, Acosh);
MATX_UNARY_OP_GEN(atanh, Atanh);


template <typename T> struct ExpjF {
  static __MATX_INLINE__ std::string str() { return "expj"; }
  template <typename T2 = T,
            std::enable_if_t<std::is_floating_point_v<T2>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ ExpjF()
  {
  }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)
  {
    if constexpr (is_matx_type_v<T>) {
      const auto UnFunc = [&](auto p1) {
        return matxHalfComplex<T>{_internal_cos(p1), _internal_sin(p1)};
      };

      return UnaryVecFunc<InWidth>(UnFunc, v1);
    }
    else {
      const auto UnFunc = [&](auto p1) {
        return cuda::std::complex<T>{_internal_cos(p1), _internal_sin(p1)};
      };

      return UnaryVecFunc<InWidth>(UnFunc, v1);
    }
  }
};
template <typename T> using ExpjOp = UnOp<T, ExpjF<T>>;

template <typename T> struct Abs2F {
  static __MATX_INLINE__ std::string str() { return "abs2"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)
  {
    if constexpr (is_complex_v<T>) {
      const auto UnFunc = [&](auto p1) {
        return p1.real() * p1.real() + p1.imag() * p1.imag();
      };

      return UnaryVecFunc<InWidth>(UnFunc, v1);
    }
    else {
      const auto UnFunc = [&](auto p1) {
        return p1 * p1;
      };

      return UnaryVecFunc<InWidth>(UnFunc, v1);
    }
  }
};
template <typename T> using Abs2Op = UnOp<T, Abs2F<T>>;

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_normcdf(T v1)
{
  return normcdf(v1);
}
template <typename T> struct NormCdfF {
  static __MATX_INLINE__ std::string str() { return "normcdf"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1) {
    const auto UnFunc = [&](auto p1) {
      return _internal_normcdf(p1);
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
};
template <typename T> using NormCdfOp = UnOp<T, NormCdfF<T>>;


template <typename T> struct RealF {
  static __MATX_INLINE__ std::string str() { return "real"; }
  static_assert(is_complex_v<T>, "real() must have complex input");

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1) {
    const auto UnFunc = [&](auto p1) {
      return p1.real();
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
};
template <typename T> using RealOp = UnOp<T, RealF<T>>;

template <typename T> struct ImagF {
  static __MATX_INLINE__ std::string str() { return "imag"; }
  static_assert(is_complex_v<T>, "imag() must have complex input");

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T v1) {
    const auto UnFunc = [&](auto p1) {
      return p1.imag();
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
};
template <typename T> using ImagOp = UnOp<T, ImagF<T>>;

template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_angle(T v1)
{
  if constexpr (is_cuda_complex_v<T>) {
    const auto Func = [&](auto p1, auto p2) {
      return cuda::std::atan2(p1.imag(), p1.real());
    };

    return UnaryVecFunc<InWidth>(Func, v1);
  }
  else {
    const auto Func = [&](auto p1, auto p2) {
      return atan2(p1.imag(), p1.real());
    };

    return UnaryVecFunc<InWidth>(Func, v1);
  }
}
template <typename T>
struct Angle {
  static __MATX_INLINE__ std::string str() { return "angle"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)
  {
    static_assert(is_complex_v<T>, "Angle operator must have complex value as input");
    return _internal_angle<T1V, InWidth>(v1);
  }
};
template <typename T> using AngleOp = UnOp<T, Angle<T>>;

template<typename T>
struct SubNegF {
  static __MATX_INLINE__ std::string str() { return "-"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__  __MATX_DEVICE__ auto op(T1V v1)
  {
    const auto Func = [&](auto p1) {
      return -p1;
    };

    return UnaryVecFunc<InWidth>(Func, v1);
  }
};
template<typename T> using SubNegOp = UnOp<T,SubNegF<T> >;

// Binary Operators


template <typename T1, typename T2> struct AddF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "+" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 + p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using AddOp = BinOp<T1, T2, AddF<T1, T2>>;


template <typename T1, typename T2> struct SubF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "-" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 - p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using SubOp = BinOp<T1, T2, SubF<T1, T2>>;

template <typename T1, typename T2> struct MulF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "*" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 * p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using MulOp = BinOp<T1, T2, MulF<T1, T2>>;


template <typename T1, typename T2> struct DivF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "/" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 / p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using DivOp = BinOp<T1, T2, DivF<T1, T2>>;

template <typename T1, typename T2> struct ModF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "%" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 % p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using ModOp = BinOp<T1, T2, ModF<T1, T2>>;

template <typename T1, typename T2, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_fmod(T1 v1, T2 v2) {
  if constexpr (is_matx_half_v<T1> || is_matx_half_v<T2>) {
    const auto BinFunc = [&](auto p1, auto p2) {
      return cuda::std::fmodf(static_cast<float>(p1), static_cast<float>(p2));
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
  else {
    // We should not have to cast here, but libcudacxx doesn't support the double version
    const auto BinFunc = [&](auto p1, auto p2) {
      return cuda::std::fmod(p1, p2);
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
}

template <typename T1, typename T2> struct FModF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "%" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2) {
    return _internal_fmod<T1V, T2V, InWidth>(v1, v2);
  }
};
template <typename T1, typename T2> using FModOp = BinOp<T1, T2, FModF<T1, T2>>;


template <typename T1, typename T2, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_atan2(T1 v1, T2 v2) {
  if constexpr (is_matx_half_v<T1> || is_matx_half_v<T2>) {
    const auto BinFunc = [&](auto p1, auto p2) {
      return atan2(p1, p2);
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
  else {
    // We should not have to cast here, but libcudacxx doesn't support the double version
    const auto BinFunc = [&](auto p1, auto p2) {
      return cuda::std::atan2(p1, p2);
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
}

template <typename T1, typename T2> struct Atan2F {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "%" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2) {
    return _internal_atan2<T1V, T2V, InWidth>(v1, v2);
  }
};
template <typename T1, typename T2> using Atan2Op = BinOp<T1, T2, Atan2F<T1, T2>>;

// MATX_BINARY_OP_GEN(pow, Pow);

template <typename T1, typename T2, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_pow(T1 v1, T2 v2)
{
  if constexpr (is_matx_type_v<T1>) {
    const auto BinFunc = [&](auto p1, auto p2) {
      return pow(p1, p2);
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
  else {
    const auto BinFunc = [&](auto p1, auto p2) {
      return cuda::std::pow(p1, p2);
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
}

template <typename T1, typename T2> struct PowF {
  static std::string str(const std::string &str1, const std::string &str2) { return "pow(" + str1 + "," + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    return _internal_pow<T1V, T2V, InWidth>(v1, v2);
  }
};
template <typename T1, typename T2> using PowOp = BinOp<T1, T2, PowF<T1, T2>>;

template <typename T1, typename T2> struct MaximumF {
  static std::string str(const std::string &str1, const std::string &str2) { return "maximum(" + str1 + "," + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return cuda::std::max(p1, p2);
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using MaximumOp = BinOp<T1, T2, MaximumF<T1, T2>>;

template <typename T1, typename T2> struct MinimumF {
  static std::string str(const std::string &str1, const std::string &str2) { return "minimum(" + str1 + "," + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return cuda::std::min(p1, p2);
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using MinimumOp = BinOp<T1, T2, MinimumF<T1, T2>>;

// Logical Operators
template <typename T1, typename T2> struct LTF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "<" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 < p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using LTOp = BinOp<T1, T2, LTF<T1, T2>>;

template <typename T1, typename T2> struct GTF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + ">" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 > p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using GTOp = BinOp<T1, T2, GTF<T1, T2>>;

template <typename T1, typename T2> struct LTEF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "<=" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 <= p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using LTEOp = BinOp<T1, T2, LTEF<T1, T2>>;

template <typename T1, typename T2> struct GTEF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + ">=" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 >= p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using GTEOp = BinOp<T1, T2, GTEF<T1, T2>>;

template <typename T1, typename T2> struct EQF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "==" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 == p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using EQOp = BinOp<T1, T2, EQF<T1, T2>>;

template <typename T1, typename T2> struct NEF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "!=" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 != p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using NEOp = BinOp<T1, T2, NEF<T1, T2>>;

template <typename T1, typename T2> struct AndAndF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "&&" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 && p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using AndAndOp = BinOp<T1, T2, AndAndF<T1, T2>>;

template <typename T1, typename T2> struct OrOrF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "||" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2)
  {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 || p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using OrOrOp = BinOp<T1, T2, OrOrF<T1, T2>>;

template <typename T1> struct NotF {
  static __MATX_INLINE__ std::string str() { return "!"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1) {
    const auto UnFunc = [&](auto p1) {
      return !p1;
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
};
template <typename T1> using NotOp = UnOp<T1, NotF<T1>>;

template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_isnan(T v1)
{
  using conversionType = typename matx::detail::value_promote_t<T>;
  if constexpr(!std::is_floating_point_v<conversionType>) {
    const auto UnFunc = [&](auto p1) {
      return false;
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }

  using castType = matx::detail::matx_convert_complex_type<T>;
  if constexpr(is_complex_v<T>) {
    const auto UnFunc = [&](auto p1) {
      return cuda::std::isnan(static_cast<typename castType::value_type>(p1.real())) || cuda::std::isnan(static_cast<typename castType::value_type>(p1.imag()));
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  } else {
    const auto UnFunc = [&](auto p1) {
      return cuda::std::isnan(static_cast<castType>(p1));
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
}
template <typename T>
struct IsNan {
  static __MATX_INLINE__ std::string str() { return "isnan"; }
  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)
  {
    return _internal_isnan<T1V, InWidth>(v1);
  }
};
template <typename T> using IsNanOp = UnOp<T, IsNan<T>>;

template <typename T, matx::detail::VecWidth InWidth = matx::detail::VecWidth::SCALAR>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto _internal_isinf(T v1)
{
  using conversionType = typename matx::detail::value_promote_t<T>;
  if constexpr(!std::is_floating_point_v<conversionType>) {
    const auto UnFunc = [&](auto p1) {
      return false;
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }

  using castType = matx::detail::matx_convert_complex_type<T>;
  if constexpr(is_complex_v<T>) {
    const auto UnFunc = [&](auto p1) {
      return cuda::std::isinf(static_cast<typename castType::value_type>(p1.real())) || cuda::std::isinf(static_cast<typename castType::value_type>(p1.imag()));
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  } else {
    const auto UnFunc = [&](auto p1) {
      return cuda::std::isinf(static_cast<castType>(p1));
    };

    return UnaryVecFunc<InWidth>(UnFunc, v1);
  }
}
template <typename T>
struct IsInf {
  static __MATX_INLINE__ std::string str() { return "isinf"; }
  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1)
  {
    return _internal_isinf<T1V, InWidth>(v1);
  }
};
template <typename T> using IsInfOp = UnOp<T, IsInf<T>>;

template <typename T1, typename T2> struct AndF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "&" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2) {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 & p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using AndOp = BinOp<T1, T2, AndF<T1, T2>>;

template <typename T1, typename T2> struct OrF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "|" + str2 + ")"; }

  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2) {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 | p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using OrOp = BinOp<T1, T2, OrF<T1, T2>>;

template <typename T1, typename T2> struct XorF {
  static std::string str(const std::string &str1, const std::string &str2) { return "(" + str1 + "^" + str2 + ")"; }
  template <matx::detail::VecWidth InWidth, matx::detail::VecWidth OutWidth, typename T1V, typename T2V>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto op(T1V v1, T2V v2) {
    const auto BinFunc = [&](auto p1, auto p2) {
      return p1 ^ p2;
    };

    return BinVecFunc<InWidth>(BinFunc, v1, v2);
  }
};
template <typename T1, typename T2> using XorOp = BinOp<T1, T2, XorF<T1, T2>>;

} // end namespace detail
} // end namespace matx