////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "matx/core/cache.h"
#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/transforms/solve/solve_common.h"

#include <limits>
#include <vector>

namespace matx {
namespace detail {

template <typename T>
__MATX_INLINE__ cublasStatus_t DenseSolveGetrfBatched(cublasHandle_t handle,
                                                      int n,
                                                      T **a_array,
                                                      int lda,
                                                      int *piv,
                                                      int *info,
                                                      int batch_size)
{
  if constexpr (std::is_same_v<T, float>) {
    return cublasSgetrfBatched(handle, n, a_array, lda, piv, info, batch_size);
  }
  else if constexpr (std::is_same_v<T, double>) {
    return cublasDgetrfBatched(handle, n, a_array, lda, piv, info, batch_size);
  }
  else if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
    return cublasCgetrfBatched(handle, n,
                               reinterpret_cast<cuComplex *const *>(a_array),
                               lda, piv, info, batch_size);
  }
  else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
    return cublasZgetrfBatched(handle, n,
                               reinterpret_cast<cuDoubleComplex *const *>(a_array),
                               lda, piv, info, batch_size);
  }
  else {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
}

template <typename T>
__MATX_INLINE__ cublasStatus_t DenseSolveGetrsBatched(cublasHandle_t handle,
                                                      int n,
                                                      int nrhs,
                                                      T **a_array,
                                                      int lda,
                                                      int *piv,
                                                      T **b_array,
                                                      int ldb,
                                                      int *info,
                                                      int batch_size)
{
  if constexpr (std::is_same_v<T, float>) {
    return cublasSgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, a_array, lda,
                               piv, b_array, ldb, info, batch_size);
  }
  else if constexpr (std::is_same_v<T, double>) {
    return cublasDgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, a_array, lda,
                               piv, b_array, ldb, info, batch_size);
  }
  else if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
    return cublasCgetrsBatched(
        handle, CUBLAS_OP_N, n, nrhs,
        reinterpret_cast<const cuComplex *const *>(a_array), lda, piv,
        reinterpret_cast<cuComplex *const *>(b_array), ldb, info, batch_size);
  }
  else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
    return cublasZgetrsBatched(
        handle, CUBLAS_OP_N, n, nrhs,
        reinterpret_cast<const cuDoubleComplex *const *>(a_array), lda, piv,
        reinterpret_cast<cuDoubleComplex *const *>(b_array), ldb, info,
        batch_size);
  }
  else {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
}

template <typename ATensor, typename BTensor>
__MATX_INLINE__ bool DenseSolveCanUseCublasBatched(const ATensor &a,
                                                  const BTensor &b)
{
  const auto n = a.Size(remove_cvref_t<ATensor>::Rank() - 1);
  const auto nrhs = GetDenseSolveNumRhs<ATensor, BTensor>(b);
  const auto batches = GetNumBatches(a);
  return batches > 1 &&
         n <= std::numeric_limits<int>::max() &&
         nrhs <= std::numeric_limits<int>::max() &&
         batches <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

template <typename ATensor, typename BTensor>
void DenseSolveCublasBatched(ATensor &a_col,
                             BTensor &b_col,
                             const cudaExecutor &exec)
{
  using T = typename remove_cvref_t<ATensor>::value_type;
  constexpr int ARANK = remove_cvref_t<ATensor>::Rank();

  const auto stream = exec.getStream();
  const int n = static_cast<int>(a_col.Size(ARANK - 1));
  const int nrhs = static_cast<int>(GetDenseSolveNumRhs<ATensor, BTensor>(b_col));
  const int batch_size = static_cast<int>(GetNumBatches(a_col));

  std::vector<T *> h_a_array;
  std::vector<T *> h_b_array;
  SetBatchPointers<BatchType::MATRIX>(a_col, h_a_array);
  if constexpr (IsDenseSolveVectorRHS<ATensor, BTensor>()) {
    SetBatchPointers<BatchType::VECTOR>(b_col, h_b_array);
  }
  else {
    SetBatchPointers<BatchType::MATRIX>(b_col, h_b_array);
  }

  T **d_a_array = nullptr;
  T **d_b_array = nullptr;
  int *d_piv = nullptr;
  int *d_info = nullptr;
  matxAlloc(reinterpret_cast<void **>(&d_a_array),
            h_a_array.size() * sizeof(T *), MATX_ASYNC_DEVICE_MEMORY, stream);
  matxAlloc(reinterpret_cast<void **>(&d_b_array),
            h_b_array.size() * sizeof(T *), MATX_ASYNC_DEVICE_MEMORY, stream);
  matxAlloc(reinterpret_cast<void **>(&d_piv),
            static_cast<size_t>(n) * batch_size * sizeof(int),
            MATX_ASYNC_DEVICE_MEMORY, stream);
  matxAlloc(reinterpret_cast<void **>(&d_info),
            static_cast<size_t>(batch_size) * sizeof(int),
            MATX_ASYNC_DEVICE_MEMORY, stream);

  cudaMemcpyAsync(d_a_array, h_a_array.data(), h_a_array.size() * sizeof(T *),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_b_array, h_b_array.data(), h_b_array.size() * sizeof(T *),
                  cudaMemcpyHostToDevice, stream);

  cublasHandle_t handle;
  auto ret = cublasCreate(&handle);
  MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxSolverError);
  ret = cublasSetStream(handle, stream);
  MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxSolverError);

  ret = DenseSolveGetrfBatched(handle, n, d_a_array, n, d_piv, d_info,
                               batch_size);
  MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxSolverError);

  std::vector<int> h_info(batch_size);
  cudaMemcpyAsync(h_info.data(), d_info, sizeof(int) * batch_size,
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CheckDenseSolveInfos(h_info, "cuBLAS", "getrfBatched");

  int h_getrs_info = 0;
  ret = DenseSolveGetrsBatched(handle, n, nrhs, d_a_array, n, d_piv,
                               d_b_array, n, &h_getrs_info, batch_size);
  MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxSolverError);
  CheckDenseSolveInfo(h_getrs_info, "cuBLAS", "getrsBatched");

  cublasDestroy(handle);
  matxFree(d_a_array);
  matxFree(d_b_array);
  matxFree(d_piv);
  matxFree(d_info);
}

template <typename ATensor, typename BTensor>
void DenseSolveCusolverLoop(ATensor &a_col,
                            BTensor &b_col,
                            const cudaExecutor &exec)
{
  using T = typename remove_cvref_t<ATensor>::value_type;
  constexpr int ARANK = remove_cvref_t<ATensor>::Rank();

  const auto stream = exec.getStream();
  const int64_t n = static_cast<int64_t>(a_col.Size(ARANK - 1));
  const int64_t nrhs = static_cast<int64_t>(GetDenseSolveNumRhs<ATensor, BTensor>(b_col));
  const auto batch_size = GetNumBatches(a_col);

  std::vector<T *> h_a_array;
  std::vector<T *> h_b_array;
  SetBatchPointers<BatchType::MATRIX>(a_col, h_a_array);
  if constexpr (IsDenseSolveVectorRHS<ATensor, BTensor>()) {
    SetBatchPointers<BatchType::VECTOR>(b_col, h_b_array);
  }
  else {
    SetBatchPointers<BatchType::MATRIX>(b_col, h_b_array);
  }

  cusolverDnHandle_t handle;
  cusolverDnParams_t dn_params;
  auto ret = cusolverDnCreate(&handle);
  MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  ret = cusolverDnCreateParams(&dn_params);
  MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  ret = cusolverDnSetStream(handle, stream);
  MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

  size_t dspace = 0;
  size_t hspace = 0;
  ret = cusolverDnXgetrf_bufferSize(handle, dn_params, n, n,
                                    MatXTypeToCudaType<T>(), h_a_array[0], n,
                                    MatXTypeToCudaType<T>(), &dspace, &hspace);
  MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

  void *d_workspace = nullptr;
  void *h_workspace = nullptr;
  int64_t *d_piv = nullptr;
  int *d_info = nullptr;
  if (dspace > 0) {
    matxAlloc(&d_workspace, batch_size * dspace, MATX_ASYNC_DEVICE_MEMORY,
              stream);
    cudaMemsetAsync(d_workspace, 0, batch_size * dspace, stream);
  }
  if (hspace > 0) {
    matxAlloc(&h_workspace, batch_size * hspace, MATX_HOST_MEMORY);
  }
  matxAlloc(reinterpret_cast<void **>(&d_piv),
            static_cast<size_t>(n) * batch_size * sizeof(int64_t),
            MATX_ASYNC_DEVICE_MEMORY, stream);
  matxAlloc(reinterpret_cast<void **>(&d_info),
            static_cast<size_t>(batch_size) * sizeof(int),
            MATX_ASYNC_DEVICE_MEMORY, stream);

  for (uint32_t i = 0; i < batch_size; i++) {
    ret = cusolverDnXgetrf(
        handle, dn_params, n, n, MatXTypeToCudaType<T>(), h_a_array[i], n,
        d_piv + static_cast<size_t>(i) * n, MatXTypeToCudaType<T>(),
        reinterpret_cast<uint8_t *>(d_workspace) + i * dspace, dspace,
        reinterpret_cast<uint8_t *>(h_workspace) + i * hspace, hspace,
        d_info + i);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  std::vector<int> h_info(batch_size);
  cudaMemcpyAsync(h_info.data(), d_info, sizeof(int) * batch_size,
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CheckDenseSolveInfos(h_info, "cuSolver", "Xgetrf");

  for (uint32_t i = 0; i < batch_size; i++) {
    ret = cusolverDnXgetrs(
        handle, dn_params, CUBLAS_OP_N, n, nrhs, MatXTypeToCudaType<T>(),
        h_a_array[i], n, d_piv + static_cast<size_t>(i) * n,
        MatXTypeToCudaType<T>(), h_b_array[i], n, d_info + i);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  cudaMemcpyAsync(h_info.data(), d_info, sizeof(int) * batch_size,
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CheckDenseSolveInfos(h_info, "cuSolver", "Xgetrs");

  cusolverDnDestroyParams(dn_params);
  cusolverDnDestroy(handle);
  matxFree(d_workspace);
  matxFree(h_workspace);
  matxFree(d_piv);
  matxFree(d_info);
}

} // end namespace detail

template <typename OutputTensor, typename ATensor, typename BTensor>
void dense_solve_impl(OutputTensor &&out,
                      const ATensor &a,
                      const BTensor &b,
                      const cudaExecutor &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  using T = typename remove_cvref_t<OutputTensor>::value_type;

  detail::ValidateDenseSolve(out, a, b);

  auto a_new = OpToTensor(a, exec);
  auto b_new = OpToTensor(b, exec);

  if constexpr (!is_matx_transform_op<ATensor>()) {
    if (!a_new.isSameView(a)) {
      (a_new = a).run(exec);
    }
  }

  if constexpr (!is_matx_transform_op<BTensor>()) {
    if (!b_new.isSameView(b)) {
      (b_new = b).run(exec);
    }
  }

  const auto stream = exec.getStream();

  T *a_ptr = nullptr;
  matxAlloc(reinterpret_cast<void **>(&a_ptr), a_new.Bytes(),
            MATX_ASYNC_DEVICE_MEMORY, stream);
  auto a_work_t = detail::TransposeCopy(a_ptr, a_new, exec);
  auto a_col = a_work_t.PermuteMatrix();

  if constexpr (detail::IsDenseSolveVectorRHS<decltype(a_new), decltype(b_new)>()) {
    T *b_ptr = nullptr;
    matxAlloc(reinterpret_cast<void **>(&b_ptr), b_new.Bytes(),
              MATX_ASYNC_DEVICE_MEMORY, stream);
    auto b_work = make_tensor<T>(b_ptr, b_new.Shape());
    (b_work = b_new).run(exec);

    if (detail::DenseSolveCanUseCublasBatched<decltype(a_col), decltype(b_work)>(a_col, b_work)) {
      detail::DenseSolveCublasBatched(a_col, b_work, exec);
    }
    else {
      detail::DenseSolveCusolverLoop(a_col, b_work, exec);
    }

    matx::copy(out, b_work, exec);
    matxFree(b_ptr);
  }
  else {
    T *b_ptr = nullptr;
    matxAlloc(reinterpret_cast<void **>(&b_ptr), b_new.Bytes(),
              MATX_ASYNC_DEVICE_MEMORY, stream);
    auto b_work_t = detail::TransposeCopy(b_ptr, b_new, exec);
    auto b_col = b_work_t.PermuteMatrix();

    if (detail::DenseSolveCanUseCublasBatched<decltype(a_col), decltype(b_col)>(a_col, b_col)) {
      detail::DenseSolveCublasBatched(a_col, b_col, exec);
    }
    else {
      detail::DenseSolveCusolverLoop(a_col, b_col, exec);
    }

    matx::copy(out, b_work_t.PermuteMatrix(), exec);
    matxFree(b_ptr);
  }

  matxFree(a_ptr);
}

} // end namespace matx
