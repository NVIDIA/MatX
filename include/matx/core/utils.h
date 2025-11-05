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
  std::string s;
  for (size_t i = 0; i < N; ++i) {
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
    return "matxFp16";
  }
  else if constexpr (std::is_same_v<T, matxBf16>) {
    return "matxBf16";
  }
  else if constexpr (std::is_same_v<T, matxFp16Complex>) {
    return "matxFp16Complex";
  }
  else if constexpr (std::is_same_v<T, matxBf16Complex>) {
    return "matxBf16Complex";
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
  else if constexpr (std::is_same_v<T, long long>) {
    return "long_long";
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



}
}



