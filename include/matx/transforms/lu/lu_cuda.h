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

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/cache.h"
#include "matx/transforms/solver_common.h"

#include <cstdio>
#include <numeric>
#include <cuda/std/__algorithm/min.h>

namespace matx {

namespace detail {

/**
 * Parameters needed to execute an LU factorization. We distinguish unique
 * factorizations mostly by the data pointer in A
 */
struct DnLUCUDAParams_t {
  int64_t m;
  int64_t n;
  void *A;
  void *piv;
  size_t batch_size;
  MatXDataType_t dtype;
  cudaExecutor exec;
};

template <typename OutputTensor, typename PivotTensor, typename ATensor>
class matxDnLUCUDAPlan_t : matxDnCUDASolver_t {
  using OutTensor_t = remove_cvref_t<OutputTensor>;
  using T1 = typename ATensor::value_type;
  using T2 = typename PivotTensor::value_type;
  static constexpr int RANK = OutTensor_t::Rank();
  static_assert(RANK >= 2, "Input/Output tensor must be rank 2 or higher");

public:
  /**
   * Plan for factoring A such that \f$\textbf{P} * \textbf{A} = \textbf{L} *
   * \textbf{U}\f$
   *
   * Creates a handle for factoring matrix A into the format above. Matrix must
   * not be singular.
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam T2
   *  Data type of Pivot vector
   * @tparam RANK
   *  Rank of A matrix
   *
   * @param piv
   *   Pivot indices
   * @param a
   *   Input tensor view
   *
   */
  matxDnLUCUDAPlan_t(PivotTensor &piv,
                       const ATensor &a,
                       const cudaExecutor &exec)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(RANK-1 == PivotTensor::Rank(), matxInvalidDim, "Pivot tensor rank must be one less than output");
    MATX_STATIC_ASSERT_STR(RANK == ATensor::Rank(), matxInvalidDim, "Output tensor must match A tensor rank in LU");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "LU solver does not support half precision");
    MATX_STATIC_ASSERT_STR((cuda::std::is_same_v<T1, typename OutTensor_t::value_type>), matxInavlidType, "Input and Output types must match");
    MATX_STATIC_ASSERT_STR((cuda::std::is_same_v<T2, int64_t>), matxInavlidType, "Pivot tensor type must be int64_t");

    params = GetLUParams(piv, a, exec);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size, false, exec);
  }

  void GetWorkspaceSize() override
  {
    CheckSolverLibraryStatus(cusolverDnXgetrf_bufferSize(this->handle, this->dn_params, params.m,
                                            params.n, MatXTypeToCudaType<T1>(),
                                            params.A, params.m,
                                            MatXTypeToCudaType<T1>(), &this->dspace,
                                            &this->hspace),
                             "cuSolver Xgetrf_bufferSize");
  }

  static DnLUCUDAParams_t GetLUParams(PivotTensor &piv,
                                      const ATensor &a,
                                      const cudaExecutor &exec) noexcept
  {
    DnLUCUDAParams_t params;
    params.batch_size = GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.piv = piv.Data();
    params.dtype = TypeToInt<T1>();
    params.exec = exec;
    return params;
  }

  /**
   * Run getrf on every matrix.
   *
   * @param status_info If non-null, each matrix's info is written to
   * status_info[i], and the call does not synchronize. Otherwise info is copied
   * back after a stream synchronization and checked in debug builds only.
   * cuSolver return-status errors throw in every build type.
   */
  void Exec(OutputTensor &out, PivotTensor &piv,
            const ATensor &a, const cudaExecutor &exec, int *status_info = nullptr)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Batch size checks
    for(int i = 0 ; i < RANK-2; i++) {
      MATX_ASSERT_STR(out.Size(i) == a.Size(i), matxInvalidDim, "Out and A must have the same batch sizes");
      MATX_ASSERT_STR(piv.Size(i) == a.Size(i), matxInvalidDim, "Piv and A must have the same batch sizes");
    }

    // Inner size checks
    MATX_ASSERT_STR((out.Size(RANK-2) == params.m) && (out.Size(RANK-1) == params.n), matxInvalidSize, "Out and A shapes do not match");
    MATX_ASSERT_STR(piv.Size(RANK-2) == cuda::std::min(params.m, params.n), matxInvalidSize, "Piv must be ... x min(m,n)");

    SetBatchPointers<BatchType::MATRIX>(out, this->batch_a_ptrs);
    SetBatchPointers<BatchType::VECTOR>(piv, this->batch_piv_ptrs);

    if (out.Data() != a.Data()) {
      (out = a).run(exec);
    }

    const auto stream = exec.getStream();
    CheckSolverLibraryStatus(cusolverDnSetStream(this->handle, stream), "cusolverDnSetStream");

    // With a status, info goes into it instead of being checked here
    int *infos = status_info != nullptr ? status_info : this->d_info;

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      const auto ret = cusolverDnXgetrf(
          this->handle, this->dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
          this->batch_a_ptrs[i], params.m, this->batch_piv_ptrs[i],
          MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(this->d_workspace) + i * this->dspace, this->dspace,
          reinterpret_cast<uint8_t *>(this->h_workspace) + i * this->hspace, this->hspace,
          infos + i);
      CheckSolverLibraryStatus(ret, "cuSolver Xgetrf");
    }

    if (status_info != nullptr) {
      return;
    }

    std::vector<int> h_info(this->batch_a_ptrs.size());
    CheckSolverInfosOnHost(this->d_info, h_info, stream, "cuSolver Xgetrf", DescribeSingularU);
  }

  /**
   * LU solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnLUCUDAPlan_t() {}

private:
  std::vector<T2 *> batch_piv_ptrs;
  DnLUCUDAParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnLUCUDAParamsKeyHash {
  std::size_t operator()(const DnLUCUDAParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size)) + (std::hash<uint64_t>()((uint64_t)(k.exec.getStream())));
  }
};

/**
 * Test LU parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnLUCUDAParamsKeyEq {
  bool operator()(const DnLUCUDAParams_t &l, const DnLUCUDAParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size &&
           l.dtype == t.dtype && l.exec.getStream() == t.exec.getStream();
  }
};

// Static caches of LU this->handles
using lu_cuda_cache_t = std::unordered_map<DnLUCUDAParams_t, std::any, DnLUCUDAParamsKeyHash, DnLUCUDAParamsKeyEq>;

} // end namespace detail


/**
 * Perform an LU decomposition
 *
 * See documentation of matxDnLUCUDAPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
 * library by deducing all parameters needed to perform an LU decomposition from
 * only the matrix A. The input and output parameters may be the same tensor. In
 * that case, the input is destroyed and the output is stored in-place.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor view
 * @param piv
 *   Output of pivot indices
 * @param a
 *   Input matrix A
 * @param exec
 *   CUDA executor
 * @param status
 *   Optional solver status. When given, getrf info is written into it on the
 *   stream, and the call does not synchronize the stream.
 */
template <typename OutputTensor, typename PivotTensor, typename ATensor>
void lu_impl(OutputTensor &&out, PivotTensor &&piv,
        const ATensor &a, const cudaExecutor &exec, detail::SolverStatusState *status = nullptr)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  using T1 = typename remove_cvref_t<OutputTensor>::value_type;

  // Validate the status before any work, so a rejected status allocates nothing.
  // One stage: getrf.
  const auto status_run = detail::BeginStatusRun(detail::ExecutorKind::Cuda, status, 1, a);
  int *status_info = status_run.Info();

  auto piv_new = OpToTensor(piv, exec);
  auto a_new = OpToTensor(a, exec);

  if(!is_matx_transform_op<ATensor>() && !a_new.isSameView(a)) {
    (a_new = a).run(exec);
  }

  /* Temporary WAR
     cuSolver doesn't support row-major layouts. Since we want to make the
     library appear as though everything is row-major, we take a performance hit
     to transpose in and out of the function. Eventually this may be fixed in
     cuSolver.
     The tensor owns the transpose buffer, so an exception frees it.
  */
  auto tp_tensor = make_tensor<T1>(a_new.Shape(), MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
  T1 *tp = tp_tensor.Data();
  auto tv = TransposeCopy(tp, a_new, exec);
  auto tvt = tv.PermuteMatrix();

  // Get parameters required by these tensors
  auto params = detail::matxDnLUCUDAPlan_t<OutputTensor, decltype(piv_new), decltype(a_new)>::GetLUParams(piv_new, tvt, exec);

  // Get cache or new LU plan if it doesn't exist
  using cache_val_type = detail::matxDnLUCUDAPlan_t<OutputTensor, decltype(piv_new), decltype(a_new)>;
  auto cache_id = detail::GetCacheIdFromType<detail::lu_cuda_cache_t>();
  MATX_LOG_DEBUG("LU transform: cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::lu_cuda_cache_t>(
    cache_id,
    params,
    [&]() {
      return std::make_shared<cache_val_type>(piv_new, tvt, exec);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tvt, piv_new, tvt, exec, status_info);
    },
    exec
  );

  /* Temporary WAR
   * Copy out of the transpose buffer, which tp_tensor frees on return */
  matx::copy(out, tv.PermuteMatrix(), exec);
}

} // end namespace matx