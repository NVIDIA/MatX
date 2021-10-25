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

#include "cublas_v2.h"
#include "matx_dim.h"
#include "matx_error.h"
#include "matx_tensor.h"
#include <cstdio>
#include <numeric>

namespace matx {

typedef enum {
  MAT_INVERSE_ALGO_LU,
} MatInverseAlgo_t;

/**
 * Parameters needed to execute a matrix inverse. Since the matrix inverse
 * reuses internally-allocated memory and overwrites the A array with the
 * factored rows, we distinguish unique inverses mostly by the data pointer in A
 * and A inverse. This ensures that two inverses of the same size, but two
 * different input tensors don't reuse the same plan.
 */
struct InverseParams_t {
  MatInverseAlgo_t algo;
  index_t n;
  void *A;
  void *A_inv;
  MatXDataType_t dtype;
  size_t batch_size;
  cudaStream_t stream;
};

template <typename T1, int RANK, MatInverseAlgo_t ALGO = MAT_INVERSE_ALGO_LU>
class matxInversePlan_t {
public:
  /**
   * Construct a matrix inverse handle
   *
   * Creates a handle for executing a matrix inverse. There are several methods
   * of performing a matrix inverse with various tradeoffs, so an algorithm type
   * is supplied to give flexibility. To perform a matrix inversion the input
   * matrix must be square, and non-singular.
   *
   * @tparam T1
   *    Data type of A matrix
   * @tparam RANK
   *    Rank of A matrix
   * @tparam ALGO
   *    Inverse algorithm to use
   *
   * @param a
   *   Input tensor view
   * @param a_inv
   *   Inverse of A (if it exists)
   *
   */
#ifdef DOXYGEN_ONLY
  matxInversePlan_t(tensor_t a_inv, tensor_t a)
  {
#else
  matxInversePlan_t(tensor_t<T1, RANK> a_inv, tensor_t<T1, RANK> a)
  {
#endif
    static_assert(RANK >= 2);

    // Ensure matrix is square
    MATX_ASSERT(a.Size(RANK - 1) == a.Size(RANK - 2), matxInvalidSize);

    for (int i = 0; i < a.Rank(); i++) {
      MATX_ASSERT(a.Size(i) == a_inv.Size(i), matxInvalidSize);
    }

    MATX_ASSERT(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS,
                matxInverseError);

    params = GetInverseParams(a_inv, a);

    if constexpr (ALGO == MAT_INVERSE_ALGO_LU) {
      // cuBLAS requires a list of pointers to each matrix. Construct that list
      // here as our batch dims
      std::vector<T1 *> in_pointers;
      std::vector<T1 *> out_pointers;
      if constexpr (RANK == 2) {
        in_pointers.push_back(&a(0, 0));
        out_pointers.push_back(&a_inv(0, 0));
      }
      else if constexpr (RANK == 3) {
        for (int i = 0; i < a.Size(0); i++) {
          in_pointers.push_back(&a(i, 0, 0));
          out_pointers.push_back(&a_inv(i, 0, 0));
        }
      }
      else {
        for (int i = 0; i < a.Size(0); i++) {
          for (int j = 0; j < a.Size(1); j++) {
            in_pointers.push_back(&a(i, j, 0, 0));
            out_pointers.push_back(&a_inv(i, j, 0, 0));
          }
        }
      }

      // Allocate any workspace needed by inverse
      matxAlloc((void **)&d_A_array, in_pointers.size() * sizeof(T1 *),
                MATX_DEVICE_MEMORY);
      matxAlloc((void **)&d_A_inv_array, in_pointers.size() * sizeof(T1 *),
                MATX_DEVICE_MEMORY);
      matxAlloc((void **)&d_pivot,
                a.Size(RANK - 1) * in_pointers.size() * sizeof(*d_info),
                MATX_DEVICE_MEMORY);
      matxAlloc((void **)&d_info, in_pointers.size() * sizeof(*d_info),
                MATX_DEVICE_MEMORY);
      cudaMemcpy(d_A_array, in_pointers.data(),
                 in_pointers.size() * sizeof(T1 *), cudaMemcpyHostToDevice);
      cudaMemcpy(d_A_inv_array, out_pointers.data(),
                 out_pointers.size() * sizeof(T1 *), cudaMemcpyHostToDevice);
    }
    else {
      MATX_THROW(matxInvalidType, "Invalid inverse algorithm");
    }
  }

  static InverseParams_t GetInverseParams(tensor_t<T1, RANK> a_inv,
                                          tensor_t<T1, RANK> a)
  {
    InverseParams_t params;
    params.A = a.Data();
    params.A_inv = a_inv.Data();
    params.algo = ALGO;
    params.n = a.Size(RANK - 1);
    params.dtype = TypeToInt<T1>();

    if constexpr (ALGO == MAT_INVERSE_ALGO_LU) {
      if constexpr (RANK == 2) {
        params.batch_size = 1;
      }
      else if constexpr (RANK == 3) {
        params.batch_size = a.Size(0);
      }
      else if constexpr (RANK == 4) {
        params.batch_size = a.Size(0) * a.Size(1);
      }
    }

    return params;
  }

  /**
   * Inverse handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxInversePlan_t()
  {
    matxFree(d_A_array);
    matxFree(d_A_inv_array);
    matxFree(d_pivot);
    matxFree(d_info);

    cublasDestroy(handle);
  }

  /**
   * Execute a matrix inverse
   *
   * Execute a matrix inverse operation on matrix A with the chosen algorithm.
   *
   * @note Views being passed to matxInverse_t must be column-major order for
   * now
   *
   * @tparam T1
   *   Type of matrix A
   * @param stream
   *   CUDA stream
   *
   */
  inline void Exec(cudaStream_t stream)
  {

    cublasSetStream(handle, stream);

    if constexpr (ALGO == MAT_INVERSE_ALGO_LU) {
      if constexpr (std::is_same_v<T1, float>) {
        ret =
            cublasSgetrfBatched(handle, params.n, d_A_array, params.n, d_pivot,
                                d_info, static_cast<int>(params.batch_size));
      }
      else if constexpr (std::is_same_v<T1, double>) {
        ret =
            cublasDgetrfBatched(handle, params.n, d_A_array, params.n, d_pivot,
                                d_info, static_cast<int>(params.batch_size));
      }
      else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
        ret =
            cublasCgetrfBatched(handle, static_cast<int>(params.n),
                                reinterpret_cast<cuComplex *const *>(d_A_array),
                                static_cast<int>(params.n), d_pivot, d_info,
                                static_cast<int>(params.batch_size));
        cudaDeviceSynchronize();
      }
      else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
        ret = cublasZgetrfBatched(
            handle, params.n,
            reinterpret_cast<cuDoubleComplex *const *>(d_A_array),
            static_cast<int>(params.n), d_pivot, d_info,
            static_cast<int>(params.batch_size));
      }

      MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxLUError);

      int h_info = 0;
      cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

      MATX_ASSERT(h_info == 0, matxLUError);

      if constexpr (std::is_same_v<T1, float>) {
        ret = cublasSgetriBatched(handle, params.n, d_A_array,
                                  static_cast<int>(params.n), d_pivot,
                                  d_A_inv_array, static_cast<int>(params.n),
                                  d_info, static_cast<int>(params.batch_size));
      }
      else if constexpr (std::is_same_v<T1, double>) {
        ret = cublasDgetriBatched(handle, static_cast<int>(params.n), d_A_array,
                                  static_cast<int>(params.n), d_pivot,
                                  d_A_inv_array, static_cast<int>(params.n),
                                  d_info, static_cast<int>(params.batch_size));
      }
      else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
        ret = cublasCgetriBatched(
            handle, static_cast<int>(params.n),
            reinterpret_cast<cuComplex *const *>(d_A_array),
            static_cast<int>(params.n), d_pivot,
            reinterpret_cast<cuComplex *const *>(d_A_inv_array),
            static_cast<int>(params.n), d_info,
            static_cast<int>(params.batch_size));
      }
      else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
        ret = cublasZgetriBatched(
            handle, static_cast<int>(params.n),
            reinterpret_cast<cuDoubleComplex *const *>(d_A_array),
            static_cast<int>(params.n), d_pivot,
            reinterpret_cast<cuDoubleComplex *const *>(d_A_inv_array),
            static_cast<int>(params.n), d_info,
            static_cast<int>(params.batch_size));
      }

      MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxInverseError);

      cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
      MATX_ASSERT(h_info == 0, matxInverseError);
    }
  }

private:
  // Member variables
  cublasStatus_t ret = CUBLAS_STATUS_SUCCESS;

  InverseParams_t params;
  cublasHandle_t handle;
  int *d_pivot;
  int *d_info;
  T1 **d_A_array;
  T1 **d_A_inv_array;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common inverse parameters change
 */
struct InverseParamsKeyHash {
  std::size_t operator()(const InverseParams_t &k) const noexcept
  {
    return (std::hash<index_t>()(k.n)) + (std::hash<index_t>()(k.batch_size)) +
           (std::hash<index_t>()((uint64_t)k.A)) +
           (std::hash<index_t>()((uint64_t)k.A_inv)) +
           (std::hash<index_t>()((uint64_t)k.stream));
  }
};

/**
 * Test inverse parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct InverseParamsKeyEq {
  bool operator()(const InverseParams_t &l, const InverseParams_t &t) const
      noexcept
  {
    return l.n == t.n && l.A == t.A && l.A_inv == t.A_inv &&
           l.stream == t.stream && l.algo == t.algo &&
           l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

// Static caches of inverse handles
static matxCache_t<InverseParams_t, InverseParamsKeyHash, InverseParamsKeyEq>
    inv_cache;

#ifdef DOXYGEN_ONLY
void inv(tensor_t a_inv, tensor_t a, cudaStream_t stream = 0)
#else
template <typename T1, int RANK, MatInverseAlgo_t ALGO = MAT_INVERSE_ALGO_LU>
void inv(tensor_t<T1, RANK> a_inv, tensor_t<T1, RANK> a,
         cudaStream_t stream = 0)
#endif
{
  // Get parameters required by these tensors
  auto params = matxInversePlan_t<T1, RANK, ALGO>::GetInverseParams(a_inv, a);
  params.stream = stream;

  // Get cache or new inverse plan if it doesn't exist
  auto ret = inv_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new matxInversePlan_t{a_inv, a};
    inv_cache.Insert(params, static_cast<void *>(tmp));
    tmp->Exec(stream);
  }
  else {
    auto inv_type =
        static_cast<matxInversePlan_t<T1, RANK, ALGO> *>(ret.value());
    inv_type->Exec(stream);
  }
}

} // end namespace matx
