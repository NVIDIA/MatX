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
#include <cuda_fp16.h>

#include "matx/core/defines.h"
#include "matx/core/error.h"

#define HOPPER_CC 9
#define AMPERE_CC 8
#define VOLTA_CC 7
#define PASCAL_CC 6

namespace matx {
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

template <typename T1, typename T2, typename T3>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto madd( const T1 &x, const T2 &y, const T3 &z) {
  using T4 = decltype(x*y+z); 
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

    const __half2 &X = *reinterpret_cast<const __half2*>(&x);
    const __half2 &Y = *reinterpret_cast<const __half2*>(&y);
    const __half2 &Z = *reinterpret_cast<const __half2*>(&z);

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



template <int RANK, typename T, std::enable_if_t<!std::is_array_v<typename remove_cvref<T>::type>, bool> = true>
auto __MATX_INLINE__ getPermuteDims(T dims) {
  constexpr auto D = dims.size();
  cuda::std::array<int, RANK> perm;
  cuda::std::array<bool, RANK> visited;

  visited.fill(false);

  // construct permutation array by moving fastest changing dims to end
  int j = RANK-1;
  #pragma unroll
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

};
};
