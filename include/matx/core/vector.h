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

#include <cuda/std/array>
#include <cuda/std/__algorithm/fill.h>
#include <matx/core/type_utils.h>

namespace matx{
namespace detail {


template <typename T, int N>
struct alignas(sizeof(T) * N) Vector {
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Vector() {}
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Vector(T v) { Fill(v); }

  template <typename T2, std::enable_if_t<std::is_same_v<typename T2::matx_vec, bool> && T2::width == N, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Vector& operator=(const T2& v) {
    for (int i = 0; i < N; i++) {
      data[i] = v.data[i];
    }
    return *this;
  }

  // Load elements from a pointer into the vector. This function is only used when multiple
  // loads need to be issued to fill the vector
  template <int EPT>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ void load(T* ptr) {
    constexpr int elements_per_load = (MAX_VEC_WIDTH_BYTES / sizeof(T));
    constexpr int num_iterations = EPT / elements_per_load;
    #pragma unroll
    for (int i = 0; i < num_iterations; i++) {
      *reinterpret_cast<float4*>(&data[i*elements_per_load]) = *reinterpret_cast<float4*>(ptr + i * elements_per_load);
    }
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ void Fill(T val) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
      data[i] = val;
    }
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T* Data() { return data.data(); }

  static constexpr size_t width = N;
  using type = T;
  using value_type = T;
  using matx_vec = bool;
  cuda::std::array<T, N> data;
};


template <typename T, typename = void> struct is_vector : std::false_type {};
template <typename T>
struct is_vector<T, std::void_t<typename T::matx_vec>>
    : std::true_type {
};


template< class T >
inline constexpr bool is_vector_v = detail::is_vector<typename remove_cvref<T>::type>::value;


template <typename T, int EPT, typename = void>
struct vector_or_scalar_impl {
  using type = Vector<T, EPT>;
};

template <typename T, int EPT>
struct vector_or_scalar_impl<T, EPT, std::enable_if_t<is_vector_v<T> && EPT != 1>> {
  using type = T;
};

template <typename T, int EPT>
struct vector_or_scalar_impl<T, EPT, std::enable_if_t<!is_vector_v<T> && EPT == 1>> {
  using type = T;
};

template <typename T, int EPT>
using vector_or_scalar_t = typename vector_or_scalar_impl<T, EPT>::type;




template <typename V>
__MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto GetVectorVal(const V& v, int idx) {
  if constexpr (is_vector_v<V>) {
    return v.data[idx];
  } else {
    return v;
  }
}


}
}
