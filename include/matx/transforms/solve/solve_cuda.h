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

#include <any>
#include <limits>
#include <memory>
#include <unordered_map>
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
  if constexpr (cuda::std::is_same_v<T, float>) {
    return cublasSgetrfBatched(handle, n, a_array, lda, piv, info, batch_size);
  }
  else if constexpr (cuda::std::is_same_v<T, double>) {
    return cublasDgetrfBatched(handle, n, a_array, lda, piv, info, batch_size);
  }
  else if constexpr (cuda::std::is_same_v<T, cuda::std::complex<float>>) {
    return cublasCgetrfBatched(handle, n,
                               reinterpret_cast<cuComplex *const *>(a_array),
                               lda, piv, info, batch_size);
  }
  else if constexpr (cuda::std::is_same_v<T, cuda::std::complex<double>>) {
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
  if constexpr (cuda::std::is_same_v<T, float>) {
    return cublasSgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, a_array, lda,
                               piv, b_array, ldb, info, batch_size);
  }
  else if constexpr (cuda::std::is_same_v<T, double>) {
    return cublasDgetrsBatched(handle, CUBLAS_OP_N, n, nrhs, a_array, lda,
                               piv, b_array, ldb, info, batch_size);
  }
  else if constexpr (cuda::std::is_same_v<T, cuda::std::complex<float>>) {
    return cublasCgetrsBatched(
        handle, CUBLAS_OP_N, n, nrhs,
        reinterpret_cast<const cuComplex *const *>(a_array), lda, piv,
        reinterpret_cast<cuComplex *const *>(b_array), ldb, info, batch_size);
  }
  else if constexpr (cuda::std::is_same_v<T, cuda::std::complex<double>>) {
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
         batches <= static_cast<index_t>(std::numeric_limits<int>::max());
}

struct DenseSolveCUDAParams_t {
  int64_t n;
  int64_t nrhs;
  index_t batch_size;
  MatXDataType_t dtype;
  cudaStream_t stream;
};

template <typename ATensor, typename BTensor>
__MATX_INLINE__ DenseSolveCUDAParams_t
GetDenseSolveCUDAParams(const ATensor &a, const BTensor &b,
                        const cudaExecutor &exec)
{
  DenseSolveCUDAParams_t params;
  params.n = static_cast<int64_t>(a.Size(remove_cvref_t<ATensor>::Rank() - 1));
  params.nrhs = static_cast<int64_t>(GetDenseSolveNumRhs<ATensor, BTensor>(b));
  params.batch_size = GetNumBatches(a);
  params.dtype = TypeToInt<typename remove_cvref_t<ATensor>::value_type>();
  params.stream = exec.getStream();
  return params;
}

struct DenseSolveCUDAParamsKeyHash {
  std::size_t operator()(const DenseSolveCUDAParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(static_cast<uint64_t>(k.n))) +
           (std::hash<uint64_t>()(static_cast<uint64_t>(k.nrhs))) +
           (std::hash<uint64_t>()(static_cast<uint64_t>(k.batch_size))) +
           (std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(k.stream))) +
           (std::hash<int>()(static_cast<int>(k.dtype)));
  }
};

struct DenseSolveCUDAParamsKeyEq {
  bool operator()(const DenseSolveCUDAParams_t &l,
                  const DenseSolveCUDAParams_t &r) const noexcept
  {
    return l.n == r.n &&
           l.nrhs == r.nrhs &&
           l.batch_size == r.batch_size &&
           l.dtype == r.dtype &&
           l.stream == r.stream;
  }
};

class DenseSolveCublasHandleGuard {
public:
  DenseSolveCublasHandleGuard() = default;
  DenseSolveCublasHandleGuard(const DenseSolveCublasHandleGuard &) = delete;
  DenseSolveCublasHandleGuard &operator=(const DenseSolveCublasHandleGuard &) = delete;

  ~DenseSolveCublasHandleGuard()
  {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  void Create()
  {
    // Create into a local so a failed call leaves handle_ null for the destructor
    cublasHandle_t handle = nullptr;
    CheckSolverCublasStatus(cublasCreate(&handle), "cublasCreate");
    handle_ = handle;
  }

  __MATX_INLINE__ cublasHandle_t get() const noexcept
  {
    return handle_;
  }

private:
  cublasHandle_t handle_ = nullptr;
};

class DenseSolveCusolverHandleGuard {
public:
  DenseSolveCusolverHandleGuard() = default;
  DenseSolveCusolverHandleGuard(const DenseSolveCusolverHandleGuard &) = delete;
  DenseSolveCusolverHandleGuard &operator=(const DenseSolveCusolverHandleGuard &) = delete;

  ~DenseSolveCusolverHandleGuard()
  {
    if (handle_ != nullptr) {
      cusolverDnDestroy(handle_);
    }
  }

  void Create()
  {
    // Create into a local so a failed call leaves handle_ null for the destructor
    cusolverDnHandle_t handle = nullptr;
    CheckSolverLibraryStatus(cusolverDnCreate(&handle), "cusolverDnCreate");
    handle_ = handle;
  }

  __MATX_INLINE__ cusolverDnHandle_t get() const noexcept
  {
    return handle_;
  }

private:
  cusolverDnHandle_t handle_ = nullptr;
};

class DenseSolveCusolverParamsGuard {
public:
  DenseSolveCusolverParamsGuard() = default;
  DenseSolveCusolverParamsGuard(const DenseSolveCusolverParamsGuard &) = delete;
  DenseSolveCusolverParamsGuard &operator=(const DenseSolveCusolverParamsGuard &) = delete;

  ~DenseSolveCusolverParamsGuard()
  {
    if (params_ != nullptr) {
      cusolverDnDestroyParams(params_);
    }
  }

  void Create()
  {
    // Create into a local so a failed call leaves params_ null for the destructor
    cusolverDnParams_t params = nullptr;
    CheckSolverLibraryStatus(cusolverDnCreateParams(&params), "cusolverDnCreateParams");
    params_ = params;
  }

  __MATX_INLINE__ cusolverDnParams_t get() const noexcept
  {
    return params_;
  }

private:
  cusolverDnParams_t params_ = nullptr;
};

template <typename T>
class DenseSolveCublasBatchedPlan_t {
public:
  DenseSolveCublasBatchedPlan_t(const DenseSolveCUDAParams_t &params,
                                const cudaExecutor &exec)
      : params_(params)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    handle_.Create();

    const auto stream = exec.getStream();
    const auto batch_size = static_cast<size_t>(params_.batch_size);
    d_a_array_.Alloc(batch_size * sizeof(T *), MATX_ASYNC_DEVICE_MEMORY,
                     stream);
    d_b_array_.Alloc(batch_size * sizeof(T *), MATX_ASYNC_DEVICE_MEMORY,
                     stream);
    d_piv_.Alloc(static_cast<size_t>(params_.n) * batch_size * sizeof(int),
                 MATX_ASYNC_DEVICE_MEMORY, stream);
    d_info_.Alloc(batch_size * sizeof(int), MATX_ASYNC_DEVICE_MEMORY, stream);
  }

  /**
   * Run getrfBatched and getrsBatched.
   *
   * @param status_info If non-null, getrf info for matrix i is written to
   * status_info[i] and stage 1 (getrs) is zeroed on the stream, since
   * getrsBatched reports only one host-side value, which throws here if
   * nonzero. The call then does not synchronize, and getrs runs whatever
   * getrf reported. Otherwise getrf info is copied back after a stream
   * synchronization and checked in debug builds only.
   */
  template <typename ATensor, typename BTensor>
  void Exec(ATensor &a_col, BTensor &b_col, const cudaExecutor &exec, int *status_info = nullptr)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    const auto stream = exec.getStream();
    const int n = static_cast<int>(params_.n);
    const int nrhs = static_cast<int>(params_.nrhs);
    const int batch_size = static_cast<int>(params_.batch_size);

    std::vector<T *> h_a_array;
    std::vector<T *> h_b_array;
    SetBatchPointers<BatchType::MATRIX>(a_col, h_a_array);
    if constexpr (IsDenseSolveVectorRHS<ATensor, BTensor>()) {
      SetBatchPointers<BatchType::VECTOR>(b_col, h_b_array);
    }
    else {
      SetBatchPointers<BatchType::MATRIX>(b_col, h_b_array);
    }

    MATX_CUDA_CHECK(cudaMemcpyAsync(d_a_array_.get(), h_a_array.data(),
                    h_a_array.size() * sizeof(T *), cudaMemcpyHostToDevice,
                    stream));
    MATX_CUDA_CHECK(cudaMemcpyAsync(d_b_array_.get(), h_b_array.data(),
                    h_b_array.size() * sizeof(T *), cudaMemcpyHostToDevice,
                    stream));

    CheckSolverCublasStatus(cublasSetStream(handle_.get(), stream), "cublasSetStream");

    int *getrf_info = status_info != nullptr ? status_info : d_info_.get();
    CheckSolverCublasStatus(DenseSolveGetrfBatched(handle_.get(), n, d_a_array_.get(), n,
                                                   d_piv_.get(), getrf_info, batch_size),
                            "cuBLAS getrfBatched");

    if (status_info == nullptr) {
      std::vector<int> h_info(static_cast<size_t>(batch_size));
      CheckSolverInfosOnHost(d_info_.get(), h_info, stream, "cuBLAS getrfBatched", DescribeSingularU);
    }

    int h_getrs_info = 0;
    CheckSolverCublasStatus(DenseSolveGetrsBatched(handle_.get(), n, nrhs, d_a_array_.get(), n,
                                                   d_piv_.get(), d_b_array_.get(), n,
                                                   &h_getrs_info, batch_size),
                            "cuBLAS getrsBatched");
    if (status_info != nullptr) {
      // getrsBatched checks its arguments on the host and reports one value for
      // the whole batch, so a failure throws here, and stage 1 holds no
      // per-matrix failures
      if (h_getrs_info != 0) {
        MATX_THROW(matxSolverError, "cuBLAS getrsBatched: " + DescribeSolverInfo(h_getrs_info));
      }
      MATX_CUDA_CHECK(cudaMemsetAsync(status_info + batch_size, 0, sizeof(int) * static_cast<size_t>(batch_size), stream));
    }
    else {
      CheckDenseSolveInfo(h_getrs_info, "cuBLAS", "getrsBatched");
    }
  }

private:
  DenseSolveCUDAParams_t params_;
  DenseSolveCublasHandleGuard handle_;
  DenseSolveAllocGuard<T *> d_a_array_;
  DenseSolveAllocGuard<T *> d_b_array_;
  DenseSolveAllocGuard<int> d_piv_;
  DenseSolveAllocGuard<int> d_info_;
};

template <typename T>
class DenseSolveCusolverLoopPlan_t {
public:
  template <typename ATensor>
  DenseSolveCusolverLoopPlan_t(const DenseSolveCUDAParams_t &params,
                               const ATensor &a_col,
                               const cudaExecutor &exec)
      : params_(params)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    const auto stream = exec.getStream();
    handle_.Create();
    dn_params_.Create();

    CheckSolverLibraryStatus(cusolverDnSetStream(handle_.get(), stream), "cusolverDnSetStream");
    CheckSolverLibraryStatus(cusolverDnXgetrf_bufferSize(
        handle_.get(), dn_params_.get(), params_.n, params_.n,
        MatXTypeToCudaType<T>(), a_col.Data(), params_.n,
        MatXTypeToCudaType<T>(), &dspace_, &hspace_), "cuSolver Xgetrf_bufferSize");

    const auto batch_size = static_cast<size_t>(params_.batch_size);
    if (dspace_ > 0) {
      d_workspace_.Alloc(batch_size * dspace_, MATX_ASYNC_DEVICE_MEMORY,
                         stream);
      cudaMemsetAsync(d_workspace_.get(), 0, batch_size * dspace_, stream);
    }
    if (hspace_ > 0) {
      h_workspace_.Alloc(batch_size * hspace_, MATX_HOST_MEMORY);
    }
    d_piv_.Alloc(static_cast<size_t>(params_.n) * batch_size * sizeof(int64_t),
                 MATX_ASYNC_DEVICE_MEMORY, stream);
    d_info_.Alloc(batch_size * sizeof(int), MATX_ASYNC_DEVICE_MEMORY, stream);
  }

  /**
   * Run Xgetrf, then Xgetrs, on every matrix.
   *
   * @param status_info If non-null, getrf info for matrix i is written to
   * status_info[i] and getrs info to status_info[batches + i], and the call
   * does not synchronize; getrs runs whatever getrf reported. Otherwise each
   * stage's info is copied back after a stream synchronization and checked in
   * debug builds only. cuSolver return-status errors throw in every build type.
   */
  template <typename ATensor, typename BTensor>
  void Exec(ATensor &a_col, BTensor &b_col, const cudaExecutor &exec, int *status_info = nullptr)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    const auto stream = exec.getStream();
    CheckSolverLibraryStatus(cusolverDnSetStream(handle_.get(), stream), "cusolverDnSetStream");

    std::vector<T *> h_a_array;
    std::vector<T *> h_b_array;
    SetBatchPointers<BatchType::MATRIX>(a_col, h_a_array);
    if constexpr (IsDenseSolveVectorRHS<ATensor, BTensor>()) {
      SetBatchPointers<BatchType::VECTOR>(b_col, h_b_array);
    }
    else {
      SetBatchPointers<BatchType::MATRIX>(b_col, h_b_array);
    }

    const auto batch_size = static_cast<size_t>(params_.batch_size);
    // With a status, getrf info is its stage 0 and getrs info its stage 1
    int *getrf_info = status_info != nullptr ? status_info : d_info_.get();
    int *getrs_info = status_info != nullptr ? status_info + batch_size : d_info_.get();
    for (size_t i = 0; i < batch_size; i++) {
      auto *d_work = dspace_ > 0 ? d_workspace_.get() + i * dspace_ : nullptr;
      auto *h_work = hspace_ > 0 ? h_workspace_.get() + i * hspace_ : nullptr;
      const auto ret = cusolverDnXgetrf(
          handle_.get(), dn_params_.get(), params_.n, params_.n,
          MatXTypeToCudaType<T>(), h_a_array[i], params_.n,
          d_piv_.get() + static_cast<size_t>(i) * params_.n,
          MatXTypeToCudaType<T>(), d_work, dspace_, h_work, hspace_,
          getrf_info + i);
      CheckSolverLibraryStatus(ret, "cuSolver Xgetrf");
    }

    std::vector<int> h_info(status_info != nullptr ? 0 : batch_size);
    if (status_info == nullptr) {
      CheckSolverInfosOnHost(d_info_.get(), h_info, stream, "cuSolver Xgetrf", DescribeSingularU);
    }

    for (size_t i = 0; i < batch_size; i++) {
      const auto ret = cusolverDnXgetrs(
          handle_.get(), dn_params_.get(), CUBLAS_OP_N, params_.n,
          params_.nrhs, MatXTypeToCudaType<T>(), h_a_array[i], params_.n,
          d_piv_.get() + static_cast<size_t>(i) * params_.n,
          MatXTypeToCudaType<T>(), h_b_array[i], params_.n,
          getrs_info + i);
      CheckSolverLibraryStatus(ret, "cuSolver Xgetrs");
    }

    if (status_info != nullptr) {
      return;
    }

    CheckSolverInfosOnHost(d_info_.get(), h_info, stream, "cuSolver Xgetrs");
  }

private:
  DenseSolveCUDAParams_t params_;
  DenseSolveCusolverHandleGuard handle_;
  DenseSolveCusolverParamsGuard dn_params_;
  DenseSolveAllocGuard<uint8_t> d_workspace_;
  DenseSolveAllocGuard<uint8_t> h_workspace_;
  DenseSolveAllocGuard<int64_t> d_piv_;
  DenseSolveAllocGuard<int> d_info_;
  size_t dspace_ = 0;
  size_t hspace_ = 0;
};

using dense_solve_cublas_cuda_cache_t =
    std::unordered_map<DenseSolveCUDAParams_t, std::any,
                       DenseSolveCUDAParamsKeyHash, DenseSolveCUDAParamsKeyEq>;
using dense_solve_cusolver_cuda_cache_t =
    std::unordered_map<DenseSolveCUDAParams_t, std::any,
                       DenseSolveCUDAParamsKeyHash, DenseSolveCUDAParamsKeyEq>;

template <typename ATensor, typename BTensor>
void DenseSolveCublasBatched(ATensor &a_col,
                             BTensor &b_col,
                             const cudaExecutor &exec,
                             int *status_info)
{
  using T = typename remove_cvref_t<ATensor>::value_type;
  auto params = GetDenseSolveCUDAParams(a_col, b_col, exec);
  using cache_val_type = DenseSolveCublasBatchedPlan_t<T>;
  auto cache_id = GetCacheIdFromType<dense_solve_cublas_cuda_cache_t>();
  MATX_LOG_DEBUG("Dense solve cuBLAS transform: cache_id={}", cache_id);
  GetCache().LookupAndExec<dense_solve_cublas_cuda_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(params, exec);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->Exec(a_col, b_col, exec, status_info);
      },
      exec);
}

template <typename ATensor, typename BTensor>
void DenseSolveCusolverLoop(ATensor &a_col,
                            BTensor &b_col,
                            const cudaExecutor &exec,
                            int *status_info)
{
  using T = typename remove_cvref_t<ATensor>::value_type;
  auto params = GetDenseSolveCUDAParams(a_col, b_col, exec);
  using cache_val_type = DenseSolveCusolverLoopPlan_t<T>;
  auto cache_id = GetCacheIdFromType<dense_solve_cusolver_cuda_cache_t>();
  MATX_LOG_DEBUG("Dense solve cuSolver transform: cache_id={}", cache_id);
  GetCache().LookupAndExec<dense_solve_cusolver_cuda_cache_t>(
      cache_id,
      params,
      [&]() {
        return std::make_shared<cache_val_type>(params, a_col, exec);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->Exec(a_col, b_col, exec, status_info);
      },
      exec);
}

} // end namespace detail

/**
 * Solve A * X = B for dense square A on a CUDA executor.
 *
 * @param out
 *   Output tensor X, shaped like B
 * @param a
 *   Square coefficient matrices A of shape `... x n x n`
 * @param b
 *   Right-hand sides B of shape `... x n` or `... x n x nrhs`
 * @param exec
 *   CUDA executor
 * @param status
 *   Optional solver status. When given, getrf info is written to stage 0 and
 *   getrs info to stage 1 on the stream, and the call does not synchronize the
 *   stream.
 */
template <typename OutputTensor, typename ATensor, typename BTensor>
void dense_solve_impl(OutputTensor &&out,
                      const ATensor &a,
                      const BTensor &b,
                      const cudaExecutor &exec,
                      detail::SolverStatusState *status = nullptr)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  using T = typename remove_cvref_t<OutputTensor>::value_type;

  detail::ValidateDenseSolve(out, a, b);

  // Validate the status before any work, so a rejected status allocates nothing.
  // Two stages: getrf, then getrs.
  const auto status_run = detail::BeginStatusRun(detail::ExecutorKind::Cuda, status, 2, a);
  int *status_info = status_run.Info();

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

  detail::DenseSolveAllocGuard<T> a_ptr;
  a_ptr.Alloc(a_new.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
  auto a_work_t = detail::TransposeCopy(a_ptr.get(), a_new, exec);
  auto a_col = a_work_t.PermuteMatrix();

  if constexpr (detail::IsDenseSolveVectorRHS<decltype(a_new), decltype(b_new)>()) {
    detail::DenseSolveAllocGuard<T> b_ptr;
    b_ptr.Alloc(b_new.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
    auto b_work = make_tensor<T>(b_ptr.get(), b_new.Shape());
    (b_work = b_new).run(exec);

    if (detail::DenseSolveCanUseCublasBatched<decltype(a_col), decltype(b_work)>(a_col, b_work)) {
      detail::DenseSolveCublasBatched(a_col, b_work, exec, status_info);
    }
    else {
      detail::DenseSolveCusolverLoop(a_col, b_work, exec, status_info);
    }

    matx::copy(out, b_work, exec);
  }
  else {
    detail::DenseSolveAllocGuard<T> b_ptr;
    b_ptr.Alloc(b_new.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
    auto b_work_t = detail::TransposeCopy(b_ptr.get(), b_new, exec);
    auto b_col = b_work_t.PermuteMatrix();

    if (detail::DenseSolveCanUseCublasBatched<decltype(a_col), decltype(b_col)>(a_col, b_col)) {
      detail::DenseSolveCublasBatched(a_col, b_col, exec, status_info);
    }
    else {
      detail::DenseSolveCusolverLoop(a_col, b_col, exec, status_info);
    }

    matx::copy(out, b_work_t.PermuteMatrix(), exec);
  }
}

} // end namespace matx
