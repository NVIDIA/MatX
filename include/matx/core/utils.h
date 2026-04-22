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
#include <string>
#include <cuda_fp16.h>

#include "matx/core/defines.h"
#include "matx/core/error.h"

namespace matx {

constexpr int HOPPER_CC = 9;
constexpr int AMPERE_CC = 8;
constexpr int VOLTA_CC = 7;
constexpr int PASCAL_CC = 6;

namespace detail {
__MATX_INLINE__ int GetDeviceAttr(cudaDeviceAttr attr) {
    int val;
    int dev;
    [[maybe_unused]] auto err = cudaGetDevice(&dev);
    MATX_ASSERT(err == cudaSuccess, matxCudaError);
    err = cudaDeviceGetAttribute(&val, attr, dev);
    MATX_ASSERT(err == cudaSuccess, matxCudaError);
    return val;
}

__MATX_INLINE__ int GetComputeCapabilityMajor() {
    return GetDeviceAttr(cudaDevAttrComputeCapabilityMajor);
}

__MATX_INLINE__ int GetComputeCapabilityMinor() {
    return GetDeviceAttr(cudaDevAttrComputeCapabilityMinor);
}

__MATX_INLINE__ int GetComputeCapability() {
    return GetComputeCapabilityMajor() * 100 + GetComputeCapabilityMinor();
}

__MATX_INLINE__ bool IsHopperOrAbove() {
    return GetComputeCapabilityMajor() >= HOPPER_CC;
}

__MATX_INLINE__ bool IsAmpereOrAbove() {
    return GetComputeCapabilityMajor() >= AMPERE_CC;
}

template <typename Op1, typename Op2>
bool SizesMatch(const Op1 &op1, const Op2 &op2) {
  if constexpr (Op1::Rank() != Op2::Rank()) {
    return false;
  }

  for (int r = 0; r < Op1::Rank(); r++) {
    if (op1.Size(r) != op2.Size(r)) {
      return false;
    }
  }

  return true;
}


template <int RANK, typename T>
  requires (!std::is_array_v<remove_cvref_t<T>>)
auto __MATX_INLINE__ getPermuteDims(T dims) {
  constexpr auto D = dims.size();
  cuda::std::array<int, RANK> perm;
  cuda::std::array<bool, RANK> visited;

  visited.fill(false);

  // construct permutation array by moving fastest changing dims to end
  int j = RANK-1;
  MATX_LOOP_UNROLL
  for(int i = D-1; i>= 0; i--) {
    int a = dims[i];
    MATX_ASSERT_STR(a >= 0 && a < RANK, matxInvalidDim, "Reduction dim out of range\n");
    MATX_ASSERT_STR(visited[a] == false, matxInvalidDim, "Reduction Dim repeated");

    visited[a] = true;

    perm[j--] = a;
  }

  // now add remaning dims to front
  j = 0;
  for(int i = 0; i < RANK;  i++) {
    if(!visited[i]) {
      perm[j++] = i;
    }
  }

  return perm;
}

template <int RANK, int D>
auto __MATX_INLINE__ getPermuteDims( const int (&dims)[D] ) {
  return getPermuteDims<RANK>(detail::to_array(dims));
}

// Runtime-rank variant for dynamic tensors. Uses MATX_MAX_DYNAMIC_RANK-sized
// array but only fills entries [0, rt_rank).
template <int D>
auto __MATX_INLINE__ getPermuteDims(int rt_rank, const int (&dims)[D]) {
  constexpr int MAX_RANK = MATX_MAX_DYNAMIC_RANK;
  cuda::std::array<int, MAX_RANK> perm;
  cuda::std::array<bool, MAX_RANK> visited;

  perm.fill(0);
  visited.fill(false);

  int j = rt_rank - 1;
  for (int i = D - 1; i >= 0; i--) {
    int a = dims[i];
    MATX_ASSERT_STR(a >= 0 && a < rt_rank, matxInvalidDim, "Reduction dim out of range\n");
    MATX_ASSERT_STR(visited[a] == false, matxInvalidDim, "Reduction Dim repeated");
    visited[a] = true;
    perm[j--] = a;
  }

  j = 0;
  for (int i = 0; i < rt_rank; i++) {
    if (!visited[i]) {
      perm[j++] = i;
    }
  }

  return perm;
}


template <typename T1, typename T2, typename T3>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto madd( const T1 &x, const T2 &y, const T3 &z) {
  // CUDA 12.6 with gcc 13 is reporting a parsing bug with the expression below. Use an alternative form.
  // using T4 = decltype(x*y+z);
  using T4 = std::invoke_result_t<decltype(std::plus<>{}), decltype(std::multiplies<>{}(x, y)), decltype(z)>;
  if constexpr (is_complex_v<T4> && !is_complex_half_v<T4>) {

    using value_type = typename T4::value_type;

    value_type xr, xi;
    value_type yr, yi;
    value_type zr, zi;

    if constexpr (is_complex_v<T1>) {
      xr = x.real();
      xi = x.imag();
    } else {
      xr = x;
      xi = value_type(0);
    }

    if constexpr (is_complex_v<T2>) {
      yr = y.real();
      yi = y.imag();
    } else {
      yr = y;
      yi = value_type(0);
    }

    if constexpr (is_complex_v<T3>) {
      zr = z.real();
      zi = z.imag();
    } else {
      zr = z;
      zi = value_type(0);
    }

    T4 Z(zr,zi);

    Z.real(Z.real() + xr*yr);
    Z.real(Z.real() - xi*yi);

    Z.imag(Z.imag() + xi*yr);
    Z.imag(Z.imag() + xr*yi);

    return Z;
  } else if constexpr (std::is_same_v<T4, matxFp16Complex>) {
    //__half2 X = make_half2(x.real(), x.imag());
    //__half2 Y = make_half2(y.real(), y.imag());
    //__half2 Z = make_half2(z.real(), z.imag());

    [[maybe_unused]] const __half2 &X = *reinterpret_cast<const __half2*>(&x);
    [[maybe_unused]] const __half2 &Y = *reinterpret_cast<const __half2*>(&y);
    [[maybe_unused]] const __half2 &Z = *reinterpret_cast<const __half2*>(&z);

#if 1
#ifdef __CUDA_ARCH__
    auto v = __hcmadd(X,Y,Z);
    return T4(v.x, v.y);
#else
    return x*y+z;
#endif
#else
    // In theory this could be faster but compiler is not folding broadcast/swap into HFMAs

    __half2 ari = make_half2(X.x, X.y);
    // negate and swap supported in hardware sm_8.6+
    __half2 air = make_half2(X.y, __hneg(X.x));
    // broadcast supported in hardware
    __half2 brr = make_half2(Y.x, Y.x);
    // broadcast supported in hardware
    __half2 bii = make_half2(Y.y, Y.y);
    __half2 c = Z;
    __half2 d;

    // HFMA2 RD, RA.H1_H0, RB.H1_H1, RC.H1_H0
    d = __hfma2(ari, brr, c);
    // HFMA2 RD, RB.H0_H0, -RA.H0_NH1, RC.H1_H0
    d = __hfma2(bii, -air, d);

    return T4(d.x, d.y);
#endif
  } else {
    return x*y+z;
  }
}


template <int RANK>
auto __MATX_INLINE__ invPermute(const cuda::std::array<int, RANK> &perm) {
  cuda::std::array<int, RANK> inv_perm;
  for (int i = 0; i < RANK; i++) {
    inv_perm[perm[i]] = i;
  }
  return inv_perm;
}

template <typename Container>
__MATX_INLINE__ std::string array_to_string(const Container& container) {
  std::string s;
  for (size_t i = 0; i < container.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(container[i]);
  }
  return s;
}

template <typename T, size_t N>
__MATX_INLINE__ std::string array_to_string(const cuda::std::array<T, N>& arr) {
  if constexpr (N == 0) {
    return std::string("");
  }
  else {
    std::string s;
    for (size_t i = 0; i < N; ++i) {
      if (i != 0) s += ", ";
      s += std::to_string(arr[i]);
    }
    return s;
  }
}

template <typename T, size_t N>
__MATX_INLINE__ std::string array_to_string(const cuda::std::array<T, N>& arr, int count) {
  std::string s;
  for (int i = 0; i < count; ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(arr[i]);
  }
  return s;
}


template <typename T>
__MATX_INLINE__ __MATX_HOST__  std::string type_to_string()
{
  // Handle standard POD types
  if constexpr (std::is_same_v<T, float>) {
    return "float";
  }
  else if constexpr (std::is_same_v<T, double>) {
    return "double";
  }
  else if constexpr (std::is_same_v<T, int>) {
    return "int";
  }
  else if constexpr (std::is_same_v<T, unsigned int>) {
    return "unsigned int";
  }
  else if constexpr (std::is_same_v<T, long>) {
    return "long";
  }
  else if constexpr (std::is_same_v<T, unsigned long>) {
    return "unsigned long";
  }
  else if constexpr (std::is_same_v<T, long long>) {
    return "long long";
  }
  else if constexpr (std::is_same_v<T, unsigned long long>) {
    return "unsigned long long";
  }
  else if constexpr (std::is_same_v<T, short>) {
    return "short";
  }
  else if constexpr (std::is_same_v<T, unsigned short>) {
    return "unsigned short";
  }
  else if constexpr (std::is_same_v<T, char>) {
    return "char";
  }
  else if constexpr (std::is_same_v<T, signed char>) {
    return "signed char";
  }
  else if constexpr (std::is_same_v<T, unsigned char>) {
    return "unsigned char";
  }
  else if constexpr (std::is_same_v<T, bool>) {
    return "bool";
  }
  else if constexpr (std::is_same_v<T, __half>) {
    return "__half";
  }
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return "__nv_bfloat16";
  }
  else if constexpr (std::is_same_v<T, matxFp16>) {
    return "matx::matxFp16";
  }
  else if constexpr (std::is_same_v<T, matxBf16>) {
    return "matx::matxBf16";
  }
  else if constexpr (std::is_same_v<T, matxFp16Complex>) {
    return "matx::matxFp16Complex";
  }
  else if constexpr (std::is_same_v<T, matxBf16Complex>) {
    return "matx::matxBf16Complex";
  }
  else if constexpr (std::is_same_v<T, matxFp16ComplexPlanar>) {
    return "matx::matxFp16ComplexPlanar";
  }
  else if constexpr (std::is_same_v<T, matxBf16ComplexPlanar>) {
    return "matx::matxBf16ComplexPlanar";
  }
  // CCCL complex types
  else if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
    return "cuda::std::complex<float>";
  }
  else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
    return "cuda::std::complex<double>";
  }
  else if constexpr (std::is_same_v<T, index_t>) {
    return "index_t";
  }
  // fallback: use typeid if available, or unknown
  else {
    return "unknown";
  }
}

// Unique type names compatible with C naming conventions
template <typename T>
__MATX_INLINE__ __MATX_HOST__  std::string type_to_string_c_name()
{
  if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
    return "cuda_std_complex_float";
  }
  else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
    return "cuda_std_complex_double";
  }
  else if constexpr (std::is_same_v<T, unsigned int>) {
    return "unsigned_int";
  }
  else if constexpr (std::is_same_v<T, unsigned long>) {
    return "unsigned_long";
  }
  else if constexpr (std::is_same_v<T, unsigned long long>) {
    return "unsigned_long_long";
  }
  else if constexpr (std::is_same_v<T, unsigned short>) {
    return "unsigned_short";
  }
  else if constexpr (std::is_same_v<T, unsigned char>) {
    return "unsigned_char";
  }
  else if constexpr (std::is_same_v<T, signed char>) {
    return "signed_char";
  }
  else if constexpr (std::is_same_v<T, long long>) {
    return "long_long";
  }
  else if constexpr (std::is_same_v<T, matxFp16>) {
    return "matx_matxFp16";
  }
  else if constexpr (std::is_same_v<T, matxBf16>) {
    return "matx_matxBf16";
  }  
  else if constexpr (std::is_same_v<T, matxFp16Complex>) {
    return "matx_matxFp16Complex";
  }
  else if constexpr (std::is_same_v<T, matxBf16Complex>) {
    return "matx_matxBf16Complex";
  }
  else if constexpr (std::is_same_v<T, matxFp16ComplexPlanar>) {
    return "matx_matxFp16ComplexPlanar";
  }
  else if constexpr (std::is_same_v<T, matxBf16ComplexPlanar>) {
    return "matx_matxBf16ComplexPlanar";
  }
  else {
    return type_to_string<T>();
  }
}


template <typename T>
auto get_jit_class_or_pod_name(const T& op)
{
  if constexpr (is_matx_op<T>()) {
    return op.get_jit_class_name();
  }
  else {
    return type_to_string<T>();
  }
}

/**
 * @brief Convert a number to a valid C++ symbol/identifier string
 * 
 * Formats a numeric value as a string that can be used in C++ variable names.
 * For complex numbers, the format is "r{real}_i{imag}".
 * For non-complex numbers, the format is the string representation of the value.
 * Special characters like '.' and '-' are replaced with 'p' (for point) and 
 * 'n' (for negative) respectively.
 * 
 * @tparam T Numeric type (can be complex or non-complex)
 * @param val Numeric value to convert
 * @return String representation safe for use in C++ identifiers
 * 
 * @example
 * number_to_symbol(cuda::std::complex<float>{1.5, -2.3}) returns "r1p5_in2p3"
 * number_to_symbol(3.14f) returns "3p14"
 * number_to_symbol(-5) returns "n5"
 */
template <typename T>
__MATX_INLINE__ std::string number_to_symbol(const T& val)
{
  // Helper lambda to sanitize floating point values for variable names
  auto sanitize_float = [](auto v) -> std::string {
    std::ostringstream oss;
    oss << v;
    std::string str = oss.str();
    for (auto& c : str) {
      if (c == '.') c = 'p';
      else if (c == '-') c = 'n';
    }
    return str;
  };

  if constexpr (is_complex_v<T>) {
    // Format complex numbers as r{real}_i{imag}
    auto real_val = val.real();
    auto imag_val = val.imag();
    return std::string("r") + sanitize_float(real_val) + "_i" + sanitize_float(imag_val);
  } else {
    // Format non-complex numbers directly
    return sanitize_float(val);
  }
}


#ifdef MATX_EN_JIT
/**
 * @brief Build a JIT tensor class name from string components.
 *
 * Format: JITTensorImpl_<type_c_name>_R<rank>_SI_<s0>_<s1>_...ST_<st0>_<st1>_...
 * The shape/stride values use '_' as separator.
 */
__MATX_INLINE__ std::string make_jit_tensor_class_name_str(
    const std::string &type_c_name,
    int rank,
    const index_t *shape,
    const index_t *strides)
{
  std::string name = "JITTensorImpl_" + type_c_name + "_";
  name += "R" + std::to_string(rank) + "_";
  name += "SI_";
  for (int i = 0; i < rank; ++i) {
    name += std::to_string(shape[i]);
    if (i != rank - 1) name += "_";
  }
  name += "ST_";
  for (int i = 0; i < rank; ++i) {
    name += std::to_string(strides[i]);
    if (i != rank - 1) name += "_";
  }
  return name;
}

/**
 * @brief Generate the common JIT tensor struct body string.
 *
 * Both tensor_impl_t and dynamic_tensor_t produce identical JIT struct code;
 * this function is the single source of truth for that struct.
 */
__MATX_INLINE__ std::string make_jit_tensor_struct_str(
    const std::string &class_name,
    const std::string &type_str,
    const std::string &stride_type_str,
    const std::string &rank_str,
    const std::string &sizes_str,
    const std::string &strides_str)
{
  return
    std::string("struct " + class_name + "  {\n") +
        "  static constexpr int RANK = " + rank_str + ";\n" +
        "  using T = " + type_str + ";\n" +
        "  using value_type = T;\n" +
        "  using matxop = bool;\n" +
        "  using stride_type = " + stride_type_str + ";\n" +
        "  T *ldata_;\n" +
        "  constexpr static cuda::std::array<index_t, " + rank_str + "> strides_ = { " + strides_str + " };\n" +
        "  constexpr static cuda::std::array<index_t, " + rank_str + "> sizes_ = { " + sizes_str + " };\n" +
        "  template <detail::ElementsPerThread EPT, int I = 0, typename ...Is>\n" +
        "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetVal([[maybe_unused]] cuda::std::tuple<Is...> tup)  {\n" +
        "    if constexpr (I < sizeof...(Is)) {\n" +
        "      if constexpr (EPT != detail::ElementsPerThread::ONE && I == sizeof...(Is) - 1) {\n" +
        "        return GetVal<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(strides_[I] * static_cast<index_t>(EPT));\n" +
        "      }\n" +
        "      else {\n" +
        "        return GetVal<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(strides_[I]);\n" +
        "      }\n" +
        "    }\n" +
        "    else {\n" +
        "      return 0;\n" +
        "    }\n" +
        "  }\n" +
        "  template <detail::ElementsPerThread EPT, int I = 0, typename ...Is>\n" +
        "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetValC([[maybe_unused]] const cuda::std::tuple<Is...> tup) const {\n" +
        "    if constexpr (I < sizeof...(Is)) {\n" +
        "      if constexpr (EPT != detail::ElementsPerThread::ONE && I == sizeof...(Is) - 1) {\n" +
        "        return GetValC<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(strides_[I] * static_cast<index_t>(EPT));\n" +
        "      }\n" +
        "      else {\n" +
        "        return GetValC<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(strides_[I]);\n" +
        "      }\n" +
        "    }\n" +
        "    else {\n" +
        "      return 0;\n" +
        "    }\n" +
        "  }\n" +
        "  template <detail::ElementsPerThread EPT, typename... Is>\n" +
        "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetOffsetOptimized(Is... indices) const {\n" +
        "     constexpr size_t rank = sizeof...(Is);\n" +
        "     constexpr int EPT_int = static_cast<int>(EPT);\n" +
        "     const cuda::std::array<index_t, rank> idx{indices...};\n" +
        "    \n" +
        "    if constexpr (rank == 1) {\n" +
        "        if constexpr (EPT != detail::ElementsPerThread::ONE) {\n" +
        "          return idx[0] * (strides_[0] * EPT_int);\n" +
        "      } else {\n" +
        "        return idx[0] * strides_[0];\n" +
        "      }\n" +
        "    }\n" +
        "    else if constexpr (rank == 2) {\n" +
        "      if constexpr (EPT != detail::ElementsPerThread::ONE) {\n" +
        "        return idx[0] * strides_[0] + idx[1] * (strides_[1] * EPT_int);\n" +
        "      } else {\n" +
        "        return idx[0] * strides_[0] + idx[1] * strides_[1];\n" +
        "      }\n" +
        "    }\n" +
        "    else if constexpr (rank == 3) {\n" +
        "      if constexpr (EPT != detail::ElementsPerThread::ONE) {\n" +
        "        return idx[0] * strides_[0] + idx[1] * strides_[1] + idx[2] * (strides_[2] * EPT_int);\n" +
        "      } else {\n" +
        "        return idx[0] * strides_[0] + idx[1] * strides_[1] + idx[2] * strides_[2];\n" +
        "      }\n" +
        "    }\n" +
        "    else if constexpr (rank == 4) {\n" +
        "      if constexpr (EPT != detail::ElementsPerThread::ONE) {\n" +
        "        return idx[0] * strides_[0] + idx[1] * strides_[1] + idx[2] * strides_[2] + idx[3] * (strides_[3] * EPT_int);\n" +
        "      } else {\n" +
        "        return idx[0] * strides_[0] + idx[1] * strides_[1] + idx[2] * strides_[2] + idx[3] * strides_[3];\n" +
        "      }\n" +
        "    }\n" +
        "    else {\n" +
        "      return GetValC<EPT, 0, Is...>(cuda::std::make_tuple(indices...));\n" +
        "    }\n" +
        "  }\n" +
        "  template <typename CapType, int I = 0, typename... Is>\n" +
        "  __MATX_INLINE__ __MATX_DEVICE__ bool CheckBounds(cuda::std::tuple<Is...> tup) const {\n" +
        "    if constexpr (I < sizeof...(Is)) {\n" +
        "      constexpr int EPT_int = static_cast<int>(CapType::ept);\n" +
        "      if constexpr (I == sizeof...(Is) - 1 && EPT_int > 1) {\n" +
        "        if ((cuda::std::get<I>(tup) + 1) * EPT_int > sizes_[I]) return false;\n" +
        "      } else {\n" +
        "        if (cuda::std::get<I>(tup) >= sizes_[I]) return false;\n" +
        "      }\n" +
        "      return CheckBounds<CapType, I+1>(tup);\n" +
        "    }\n" +
        "    return true;\n" +
        "  }\n" +
        "  template <typename CapType, int M = RANK, typename... Is,\n" +
        "            cuda::std::enable_if_t<cuda::std::conjunction_v<cuda::std::is_integral<Is>...>, bool> = true>\n" +
        "  __MATX_INLINE__  __MATX_DEVICE__ auto operator()(Is... indices) const noexcept {\n" +
        "    static_assert(sizeof...(Is) == M, \"Number of indices of operator() must match rank of tensor\");\n" +
        "    constexpr int EPT_int = static_cast<int>(CapType::ept);\n" +
        "    using ReturnType = cuda::std::conditional_t<CapType::ept == detail::ElementsPerThread::ONE, T, detail::Vector<T, EPT_int>>;\n" +
        "    if constexpr (CapType::pass_through_threads) {\n" +
        "      if (!CheckBounds<CapType, 0>(cuda::std::make_tuple(indices...))) {\n" +
        "        return ReturnType{};\n" +
        "      }\n" +
        "    }\n" +
        "    const index_t offset = GetOffsetOptimized<CapType::ept>(indices...);\n" +
        "    if constexpr (CapType::ept == detail::ElementsPerThread::ONE) {\n" +
        "      return ldata_[offset];\n" +
        "    } else if constexpr (EPT_int * sizeof(T) <= MAX_VEC_WIDTH_BYTES ) {\n" +
        "      return *reinterpret_cast<detail::Vector<T, EPT_int>*>(ldata_ + offset);\n" +
        "    } else {\n" +
        "      detail::Vector<T, EPT_int> vec;\n" +
        "      vec.template load<EPT_int>(ldata_ + offset);\n" +
        "      return vec;\n" +
        "    }\n" +
        "  }\n" +
        "  template <typename CapType, int M = RANK, typename... Is,\n" +
        "            cuda::std::enable_if_t<cuda::std::conjunction_v<cuda::std::is_integral<Is>...>, bool> = true>\n" +
        "  __MATX_INLINE__  __MATX_DEVICE__ auto operator()(Is... indices) noexcept\n" +
        "    -> cuda::std::conditional_t<CapType::ept == detail::ElementsPerThread::ONE, T&, detail::Vector<T, static_cast<int>(CapType::ept)>&>\n" +
        "  {\n" +
        "    static_assert(sizeof...(Is) == M, \"Number of indices of operator() must match rank of tensor\");\n" +
        "    constexpr int EPT_int = static_cast<int>(CapType::ept);\n" +
        "    if constexpr (CapType::pass_through_threads) {\n" +
        "      using ReturnType = cuda::std::conditional_t<CapType::ept == detail::ElementsPerThread::ONE, T, detail::Vector<T, EPT_int>>;\n" +
        "      __align__(alignof(ReturnType)) __shared__ unsigned char dummy_storage[sizeof(ReturnType)];\n" +
        "      auto &dummy_ = *reinterpret_cast<ReturnType *>(dummy_storage);\n" +
        "      if (!CheckBounds<CapType, 0>(cuda::std::make_tuple(indices...))) {\n" +
        "        return dummy_;\n" +
        "      }\n" +
        "    }\n" +
        "    const index_t offset = GetOffsetOptimized<CapType::ept>(indices...);\n" +
        "    if constexpr (CapType::ept == detail::ElementsPerThread::ONE) {\n" +
        "      return ldata_[offset];\n" +
        "    } else {\n" +
        "      return *reinterpret_cast<detail::Vector<T, EPT_int>*>(ldata_ + offset);\n" +
        "    }\n" +
        "  }\n" +
        "  template <typename CapType>\n" +
        "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) const noexcept\n" +
        "  {\n" +
        "    return cuda::std::apply([&](auto &&...args) {\n" +
        "      return this->operator()<CapType>(args...);\n" +
        "    }, idx);\n" +
        "  }\n" +
        "  template <typename CapType>\n" +
        "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) noexcept\n" +
        "  {\n" +
        "    return cuda::std::apply([&](auto &&...args) -> T& {\n" +
        "      return this->operator()<CapType>(args...);\n" +
        "    }, idx);\n" +
        "  }\n" +
        "  template <typename CapType, int M = RANK, typename... Is>\n" +
        "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T* data_ptr(index_t block_idx, index_t ttl_threads) const noexcept\n" +
        "  {\n" +
        "    return ldata_ + block_idx * ttl_threads * static_cast<index_t>(CapType::ept);\n" +
        "  }\n" +
        "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n" +
        "  {\n" +
        "    return " + rank_str + ";\n" +
        "  }\n" +
        "  constexpr __MATX_INLINE__  __MATX_DEVICE__ index_t Size(int dim) const\n" +
        "  {\n" +
        "    return sizes_[dim];\n " +
        "  }\n" +
        "};\n";
}
#endif

}
}

