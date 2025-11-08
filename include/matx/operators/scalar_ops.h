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

#include "matx/operators/scalar_internal.h"
#include <cuda/std/cmath>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/max.h>

// Helper macro to conditionally generate code only in non-RTC mode
#ifndef __CUDACC_RTC__
#define MATX_IFNDEF_CUDACC_RTC(...) __VA_ARGS__
#else
#define MATX_IFNDEF_CUDACC_RTC(...)
#endif

namespace matx {
namespace detail {

// This file defines operators on a scalar

// Utility macro for generating functions that have half precision intrinsics as
// an option. Lots of verbose code in here because of compiler bugs with
// constexpr if
#define MATX_UNARY_OP_GEN(FUNC, OPNAME)                                        \
  template <typename T> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_##FUNC(T v1) { \
    if constexpr (is_matx_type_v<T>) {    \
      return FUNC(v1); \
    } \
    else { \
      return cuda::std::FUNC(v1);    \
    } \
  } \
  MATX_IFNDEF_CUDACC_RTC( \
  template <typename T> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_##FUNC(T v1) { \
    if constexpr (is_vector_v<T>) {    \
      return UnaryVecFunc(scalar_internal_##FUNC<typename T::value_type>, v1); \
    } \
    else { \
      return scalar_internal_##FUNC(v1);    \
    } \
  } \
  template <typename T> struct OPNAME##Op {\
    using emits_jit_str = bool; \
    static __MATX_INLINE__ std::string str() { return #FUNC; }                 \
    template <typename CapType, typename T1V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(T1V v1) const { \
      return UnaryVecFunc(internal_##FUNC<T>, v1); \
    } \
    using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_##FUNC<T>), T>; \
    __MATX_INLINE__ std::string get_jit_class_name() const { \
      return std::string("JIT") + #OPNAME + "Op"; \
    } \
    __MATX_INLINE__ auto get_jit_op_str() const { \
      const std::string class_name = get_jit_class_name(); \
      return cuda::std::make_tuple( \
        class_name, \
        std::string( \
          "template <typename T>\n" \
          "static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_" #FUNC "(T v1) {\n" \
          "  if constexpr (is_vector_v<T>) {\n" \
          "    return UnaryVecFunc(scalar_internal_" #FUNC "<typename T::value_type>, v1);\n" \
          "  }\n" \
          "  else {\n" \
          "    return scalar_internal_" #FUNC "(v1);\n" \
          "  }\n" \
          "}\n" \
          "template <typename T> struct " + class_name + " {\n" \
          "  using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_" #FUNC "<T>), T>;\n" \
          "  template <typename CapType, typename T1V>\n" \
          "  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(T1V v1) const {\n" \
          "    return UnaryVecFunc(internal_" #FUNC "<T>, v1);\n" \
          "  }\n" \
          "};\n" \
        ) \
      ); \
    } \
    template <OperatorCapability Cap, typename InType> \
    __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const { \
      if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) { \
        static_assert(std::is_same_v<InType, std::unordered_map<std::string, std::string>>, \
                      "JIT_CLASS_QUERY capability requires std::unordered_map<std::string, std::string> as input type"); \
        const auto [key, value] = get_jit_op_str(); \
        if (in.find(key) == in.end()) { \
          in[key] = value; \
        } \
        return true; \
      } \
      else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) { \
        return get_jit_class_name() + "<" + detail::type_to_string<T>() + ">"; \
      } \
      else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) { \
        return true; \
      } \
      else { \
        return capability_attributes<Cap>::default_value; \
      } \
    } \
  }; \
  )

// Unary operator with a custom function
// The scalar_internal_FUNC implementation should be in scalar_internal.h
#define MATX_UNARY_OP_GEN_NOFUNC(FUNC, OPNAME)                                        \
  MATX_IFNDEF_CUDACC_RTC( \
  template <typename T> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_##FUNC(T v1) { \
    if constexpr (is_vector_v<T>) {    \
      return UnaryVecFunc(scalar_internal_##FUNC<typename T::value_type>, v1); \
    } \
    else { \
      return scalar_internal_##FUNC(v1);    \
    } \
  } \
  template <typename T> struct OPNAME##Op {                                     \
    using emits_jit_str = bool; \
    static __MATX_INLINE__ std::string str() { return #FUNC; }                 \
    template <typename CapType, typename T1V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(T1V v1) const { \
      return UnaryVecFunc(internal_##FUNC<T>, v1); \
    } \
    using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_##FUNC<T>), T>; \
    __MATX_INLINE__ std::string get_jit_class_name() const { \
      return std::string("JIT") + #OPNAME + "Op"; \
    } \
    __MATX_INLINE__ auto get_jit_op_str() const { \
      const std::string class_name = get_jit_class_name(); \
      return cuda::std::make_tuple( \
        class_name, \
        std::string( \
          "template <typename T>\n" \
          "static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_" #FUNC "(T v1) {\n" \
          "  if constexpr (is_vector_v<T>) {\n" \
          "    return UnaryVecFunc(scalar_internal_" #FUNC "<typename T::value_type>, v1);\n" \
          "  }\n" \
          "  else {\n" \
          "    return scalar_internal_" #FUNC "(v1);\n" \
          "  }\n" \
          "}\n" \
          "template <typename T> struct " + class_name + " {\n" \
          "  using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_" #FUNC "<T>), T>;\n" \
          "  template <typename CapType, typename T1V>\n" \
          "  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(T1V v1) const {\n" \
          "    return UnaryVecFunc(internal_" #FUNC "<T>, v1);\n" \
          "  }\n" \
          "};\n" \
        ) \
      ); \
    } \
    template <OperatorCapability Cap, typename InType> \
    __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const { \
      if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) { \
        static_assert(std::is_same_v<InType, std::unordered_map<std::string, std::string>>, \
                      "JIT_CLASS_QUERY capability requires std::unordered_map<std::string, std::string> as input type"); \
        const auto [key, value] = get_jit_op_str(); \
        if (in.find(key) == in.end()) { \
          in[key] = value; \
        } \
        return true; \
      } \
      else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) { \
        return get_jit_class_name() + "<" + detail::type_to_string<T>() + ">"; \
      } \
      else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) { \
        return true; \
      } \
      else { \
        return capability_attributes<Cap>::default_value; \
      } \
    } \
  }; \
  )

// Standard binary function
#define MATX_BINARY_OP_GEN(FUNC, OPNAME)                                       \
  template <typename T1, typename T2> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_##FUNC(T1 v1, T2 v2) { \
    if constexpr (is_matx_type_v<T1> || is_matx_type_v<T2>) {    \
      return FUNC(v1, v2); \
    } \
    else { \
      return cuda::std::FUNC(v1, v2);    \
    } \
  } \
  MATX_IFNDEF_CUDACC_RTC( \
  template <typename T1, typename T2> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_##FUNC(T1 v1, T2 v2) { \
    if constexpr (is_vector_v<T1> || is_vector_v<T2>) {    \
      return BinVecFunc(scalar_internal_##FUNC<typename T1::value_type, typename T2::value_type>, v1, v2); \
    } \
    else { \
      return scalar_internal_##FUNC(v1, v2);    \
    } \
  } \
  template <typename T1, typename T2> struct OPNAME##Op {                                     \
    using emits_jit_str = bool; \
    static __MATX_INLINE__ std::string str(const std::string &in1, const std::string &in2) { return std::string(#FUNC) + "(" + in1 + "," + in2 + ")"; } \
    template <typename CapType, typename T1V, typename T2V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(T1V v1, T2V v2) const { \
      return BinVecFunc(internal_##FUNC<T1, T2>, v1, v2); \
    } \
    using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_##FUNC<T1, T2>), T1, T2>; \
    __MATX_INLINE__ std::string get_jit_class_name() const { \
      return std::string("JIT") + #OPNAME + "Op"; \
    } \
    __MATX_INLINE__ auto get_jit_op_str() const { \
      const std::string class_name = get_jit_class_name(); \
      return cuda::std::make_tuple( \
        class_name, \
        std::string( \
          "template <typename T1, typename T2>\n" \
          "static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_" #FUNC "(T1 v1, T2 v2) {\n" \
          "  if constexpr (is_vector_v<T1> || is_vector_v<T2>) {\n" \
          "    return BinVecFunc(scalar_internal_" #FUNC "<typename T1::value_type, typename T2::value_type>, v1, v2);\n" \
          "  }\n" \
          "  else {\n" \
          "    return scalar_internal_" #FUNC "(v1, v2);\n" \
          "  }\n" \
          "}\n" \
          "template <typename T1, typename T2> struct " + class_name + " {\n" \
          "  using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_" #FUNC "<T1, T2>), T1, T2>;\n" \
          "  template <typename CapType, typename T1V, typename T2V>\n" \
          "  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(T1V v1, T2V v2) const {\n" \
          "    return BinVecFunc(internal_" #FUNC "<T1, T2>, v1, v2);\n" \
          "  }\n" \
          "};\n" \
        ) \
      ); \
    } \
    template <OperatorCapability Cap, typename InType> \
    __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const { \
      if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) { \
        static_assert(std::is_same_v<InType, std::unordered_map<std::string, std::string>>, \
                      "JIT_CLASS_QUERY capability requires std::unordered_map<std::string, std::string> as input type"); \
        const auto [key, value] = get_jit_op_str(); \
        if (in.find(key) == in.end()) { \
          in[key] = value; \
        } \
        return true; \
      } \
      else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) { \
        return get_jit_class_name() + "<" + detail::type_to_string<T1>() + "," + detail::type_to_string<T2>() + ">"; \
      } \
      else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) { \
        return true; \
      } \
      else { \
        return capability_attributes<Cap>::default_value; \
      } \
    } \
  }; \
  )

#define MATX_BINARY_OP_GEN_OPERATOR(FUNC, OPNAME, OPSYM)                                       \
  template <typename T1, typename T2> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_##FUNC(T1 v1, T2 v2) { \
    return v1 OPSYM v2; \
  } \
  MATX_IFNDEF_CUDACC_RTC( \
  template <typename T1, typename T2> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_##FUNC(T1 v1, T2 v2) { \
    if constexpr (is_vector_v<T1> || is_vector_v<T2>) {    \
      return BinVecFunc(scalar_internal_##FUNC<typename T1::value_type, typename T2::value_type>, v1, v2); \
    } \
    else { \
      return scalar_internal_##FUNC(v1, v2);    \
    } \
  } \
  template <typename T1, typename T2> struct OPNAME##Op {                   \
    using emits_jit_str = bool; \
    static __MATX_INLINE__ std::string str(const std::string &in1, const std::string &in2) { return std::string(#FUNC) + "(" + in1 + "," + in2 + ")"; } \
    template <typename CapType, typename T1V, typename T2V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1V &v1, const T2V &v2) const { \
      return BinVecFunc(internal_##FUNC<T1, T2>, v1, v2); \
    } \
    using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_##FUNC<T1, T2>), T1, T2>; \
    __MATX_INLINE__ std::string get_jit_class_name() const { \
      return std::string("JIT") + #OPNAME + "Op"; \
    } \
    __MATX_INLINE__ auto get_jit_op_str() const { \
      const std::string class_name = get_jit_class_name(); \
      return cuda::std::make_tuple( \
        class_name, \
        std::string( \
          "template <typename T1, typename T2>\n" \
          "static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_" #FUNC "(T1 v1, T2 v2) {\n" \
          "  if constexpr (is_vector_v<T1> || is_vector_v<T2>) {\n" \
          "    return BinVecFunc(scalar_internal_" #FUNC "<typename T1::value_type, typename T2::value_type>, v1, v2);\n" \
          "  }\n" \
          "  else {\n" \
          "    return scalar_internal_" #FUNC "(v1, v2);\n" \
          "  }\n" \
          "}\n" \
          "template <typename T1, typename T2> struct " + class_name + " {\n" \
          "  using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_" #FUNC "<T1, T2>), T1, T2>;\n" \
          "  template <typename CapType, typename T1V, typename T2V>\n" \
          "  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(const T1V &v1, const T2V &v2) const {\n" \
          "    return BinVecFunc(internal_" #FUNC "<T1, T2>, v1, v2);\n" \
          "  }\n" \
          "};\n" \
        ) \
      ); \
    } \
    template <OperatorCapability Cap, typename InType> \
    __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const { \
      if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) { \
        static_assert(std::is_same_v<InType, std::unordered_map<std::string, std::string>>, \
                      "JIT_CLASS_QUERY capability requires std::unordered_map<std::string, std::string> as input type"); \
        const auto [key, value] = get_jit_op_str(); \
        if (in.find(key) == in.end()) { \
          in[key] = value; \
        } \
        return true; \
      } \
      else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) { \
        return get_jit_class_name() + "<" + detail::type_to_string<T1>() + "," + detail::type_to_string<T2>() + ">"; \
      } \
      else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) { \
        return true; \
      } \
      else { \
        return capability_attributes<Cap>::default_value; \
      } \
    } \
  }; \
  )

// Binary operator with a custom function
// The scalar_internal_FUNC implementation should be in scalar_internal.h
#define MATX_BINARY_OP_NOFUNC(FUNC, OPNAME)                                       \
  MATX_IFNDEF_CUDACC_RTC( \
  template <typename T1, typename T2> \
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_##FUNC(T1 v1, T2 v2) { \
    if constexpr (is_vector_v<T1> || is_vector_v<T2>) {    \
      return BinVecFunc(scalar_internal_##FUNC<typename T1::value_type, typename T2::value_type>, v1, v2); \
    } \
    else { \
      return scalar_internal_##FUNC(v1, v2);    \
    } \
  } \
  template <typename T1, typename T2> struct OPNAME##Op {                                     \
    using emits_jit_str = bool; \
    static __MATX_INLINE__ std::string str(const std::string &in1, const std::string &in2) { return std::string(#FUNC) + "(" + in1 + "," + in2 + ")"; } \
    template <typename CapType, typename T1V, typename T2V> \
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(T1V v1, T2V v2) const { \
      return BinVecFunc(internal_##FUNC<T1, T2>, v1, v2); \
    } \
    using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_##FUNC<T1, T2>), T1, T2>; \
    __MATX_INLINE__ std::string get_jit_class_name() const { \
      return std::string("JIT") + #OPNAME + "Op"; \
    } \
    __MATX_INLINE__ auto get_jit_op_str() const { \
      const std::string class_name = get_jit_class_name(); \
      return cuda::std::make_tuple( \
        class_name, \
        std::string( \
          "template <typename T1, typename T2>\n" \
          "static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto internal_" #FUNC "(T1 v1, T2 v2) {\n" \
          "  if constexpr (is_vector_v<T1> || is_vector_v<T2>) {\n" \
          "    return BinVecFunc(scalar_internal_" #FUNC "<typename T1::value_type, typename T2::value_type>, v1, v2);\n" \
          "  }\n" \
          "  else {\n" \
          "    return scalar_internal_" #FUNC "(v1, v2);\n" \
          "  }\n" \
          "}\n" \
          "template <typename T1, typename T2> struct " + class_name + " {\n" \
          "  using value_type = cuda::std::invoke_result_t<decltype(scalar_internal_" #FUNC "<T1, T2>), T1, T2>;\n" \
          "  template <typename CapType, typename T1V, typename T2V>\n" \
          "  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(T1V v1, T2V v2) const {\n" \
          "    return BinVecFunc(internal_" #FUNC "<T1, T2>, v1, v2);\n" \
          "  }\n" \
          "};\n" \
        ) \
      ); \
    } \
    template <OperatorCapability Cap, typename InType> \
    __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const { \
      if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) { \
        static_assert(std::is_same_v<InType, std::unordered_map<std::string, std::string>>, \
                      "JIT_CLASS_QUERY capability requires std::unordered_map<std::string, std::string> as input type"); \
        const auto [key, value] = get_jit_op_str(); \
        if (in.find(key) == in.end()) { \
          in[key] = value; \
        } \
        return true; \
      } \
      else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) { \
        return get_jit_class_name() + "<" + detail::type_to_string<T1>() + "," + detail::type_to_string<T2>() + ">"; \
      } \
      else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) { \
        return true; \
      } \
      else { \
        return capability_attributes<Cap>::default_value; \
      } \
    } \
  }; \
  )



MATX_UNARY_OP_GEN(ceil, Ceil);
MATX_UNARY_OP_GEN(floor, Floor);
MATX_UNARY_OP_GEN(round, Round);
MATX_UNARY_OP_GEN(exp, Exp);
MATX_UNARY_OP_GEN(sqrt, Sqrt);
MATX_UNARY_OP_GEN(log10, Log10);
MATX_UNARY_OP_GEN(log2, Log2);
MATX_UNARY_OP_GEN(log, Log);
MATX_UNARY_OP_GEN(abs, Abs);
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


MATX_UNARY_OP_GEN_NOFUNC(rsqrt, RSqrt);
MATX_UNARY_OP_GEN_NOFUNC(csqrt, CSqrt);
MATX_UNARY_OP_GEN_NOFUNC(conj, Conj);

// Trigonometric functions
MATX_UNARY_OP_GEN_NOFUNC(sin, Sin);
MATX_UNARY_OP_GEN_NOFUNC(cos, Cos);
MATX_UNARY_OP_GEN_NOFUNC(expj, Expj);
MATX_UNARY_OP_GEN_NOFUNC(abs2, Abs2);
MATX_UNARY_OP_GEN_NOFUNC(normcdf, NormCdf);
MATX_UNARY_OP_GEN_NOFUNC(real, Real);
MATX_UNARY_OP_GEN_NOFUNC(imag, Imag);
MATX_UNARY_OP_GEN_NOFUNC(angle, Angle);
MATX_UNARY_OP_GEN_NOFUNC(subneg, SubNeg);

// Binary Operators

MATX_UNARY_OP_GEN_NOFUNC(not, Not);
MATX_UNARY_OP_GEN_NOFUNC(isnan, IsNan);
MATX_UNARY_OP_GEN_NOFUNC(isinf, IsInf);


// Binary Operators
MATX_BINARY_OP_GEN_OPERATOR(add, Add, +);
MATX_BINARY_OP_GEN_OPERATOR(sub, Sub, -);
MATX_BINARY_OP_GEN_OPERATOR(mul, Mul, *);
MATX_BINARY_OP_GEN_OPERATOR(div, Div, /);
MATX_BINARY_OP_GEN_OPERATOR(mod, Mod, %);

MATX_BINARY_OP_NOFUNC(fmod, FMod);
MATX_BINARY_OP_NOFUNC(atan2, Atan2);

MATX_BINARY_OP_GEN(pow, Pow);

MATX_BINARY_OP_NOFUNC(max, Maximum);
MATX_BINARY_OP_NOFUNC(min, Minimum);

// Logical Operators
MATX_BINARY_OP_GEN_OPERATOR(LT, LT, <);
MATX_BINARY_OP_GEN_OPERATOR(GT, GT, >);
MATX_BINARY_OP_GEN_OPERATOR(LTE, LTE, <=);
MATX_BINARY_OP_GEN_OPERATOR(GTE, GTE, >=);
MATX_BINARY_OP_GEN_OPERATOR(EQ, EQ, ==);
MATX_BINARY_OP_GEN_OPERATOR(NE, NE, !=);
MATX_BINARY_OP_GEN_OPERATOR(andand, AndAnd, &&);
MATX_BINARY_OP_GEN_OPERATOR(oror, OrOr, ||);
MATX_BINARY_OP_GEN_OPERATOR(bitand, And, &);
MATX_BINARY_OP_GEN_OPERATOR(bitor, Or, |);
MATX_BINARY_OP_GEN_OPERATOR(bitxor, Xor, ^);


} // end namespace detail
} // end namespace matx
