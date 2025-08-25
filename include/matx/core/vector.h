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

template <typename T>
constexpr size_t __MATX_HOST__ __MATX_DEVICE__ alignment_by_type() {
  // See alignment requirements for CUDA vector types at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types-alignment-requirements-in-device-code
  // Some vector types allow alignments lower than sizeof(T) to maintain power-of-two alignments. This is particularly
  // important for types where sizeof(T) > MAX_VEC_WIDTH_BYTES as we would not be able to load a single element
  // with one load instruction if we assume that the alignment is sizeof(T).
  if constexpr (std::is_same_v<T, char3> || std::is_same_v<T, uchar3>) {
    return 1;
  } else if constexpr (std::is_same_v<T, short3> || std::is_same_v<T, ushort3>) {
    return 2;
  } else if constexpr (std::is_same_v<T, int3> || std::is_same_v<T, uint3>) {
    return 4;
  } else if constexpr (std::is_same_v<T, long3> || std::is_same_v<T, ulong3>) {
    return sizeof(long) == sizeof(int) ? 4 : 8;
  } else if constexpr (std::is_same_v<T, long4> || std::is_same_v<T, ulong4>) {
    return 16;
  } else if constexpr (std::is_same_v<T, longlong3> || std::is_same_v<T, ulonglong3>) {
    return 8;
  } else if constexpr (std::is_same_v<T, longlong4> || std::is_same_v<T, ulonglong4>) {
    return 16;
  } else if constexpr (std::is_same_v<T, float3>) {
    return 4;
  } else if constexpr (std::is_same_v<T, double3>) {
    return 8;
  } else if constexpr (std::is_same_v<T, double4>) {
    return 16;
  } else {
    return sizeof(T);
  }
}

template <typename T, int N>
struct alignas(alignment_by_type<T>() * N) Vector {
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
    if constexpr (sizeof(T) == alignment_by_type<T>()) {
      constexpr int elements_per_load = (MAX_VEC_WIDTH_BYTES / sizeof(T));
      constexpr int num_iterations = EPT / elements_per_load;
      MATX_LOOP_UNROLL
      for (int i = 0; i < num_iterations; i++) {
        *reinterpret_cast<float4*>(&data[i*elements_per_load]) = *reinterpret_cast<float4*>(ptr + i * elements_per_load);
      }
    } else {
      // If the alignment of the type does not match the types size, then we load it as
      // a scalar. This is true for vector types -- e.g. float3 has an alignment of 4B,
      // but a size of 12B. The vectorization logic would need to be updated to handle
      // such cases, but we now we load them as scalars.
      MATX_LOOP_UNROLL
      for (int i = 0; i < EPT; i++) {
        data[i] = ptr[i];
      }
    }
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ void Fill(T val) {
    MATX_LOOP_UNROLL
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
