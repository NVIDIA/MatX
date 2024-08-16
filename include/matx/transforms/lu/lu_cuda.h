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
#include "cusolverDn.h"

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/cache.h"
#include "matx/transforms/solver_common.h"

#include <cstdio>
#include <numeric>

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
                       const ATensor &a)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(RANK-1 == PivotTensor::Rank(), matxInvalidDim, "Pivot tensor rank must be one less than output");
    MATX_STATIC_ASSERT_STR(RANK == ATensor::Rank(), matxInvalidDim, "Output tensor must match A tensor rank in LU");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "LU solver does not support half precision");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T1, typename OutTensor_t::value_type>), matxInavlidType, "Input and Output types must match");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T2, int64_t>), matxInavlidType, "Pivot tensor type must be int64_t");

    params = GetLUParams(piv, a);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize() override
  {
    cusolverStatus_t ret = cusolverDnXgetrf_bufferSize(this->handle, this->dn_params, params.m,
                                            params.n, MatXTypeToCudaType<T1>(),
                                            params.A, params.m,
                                            MatXTypeToCudaType<T1>(), &this->dspace,
                                            &this->hspace);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnLUCUDAParams_t GetLUParams(PivotTensor &piv,
                                  const ATensor &a) noexcept
  {
    DnLUCUDAParams_t params;
    params.batch_size = GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.piv = piv.Data();
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(OutputTensor &out, PivotTensor &piv,
            const ATensor &a, const cudaExecutor &exec)
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
    cusolverDnSetStream(this->handle, stream);

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXgetrf(
          this->handle, this->dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
          this->batch_a_ptrs[i], params.m, this->batch_piv_ptrs[i],
          MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(this->d_workspace) + i * this->dspace, this->dspace,
          reinterpret_cast<uint8_t *>(this->h_workspace) + i * this->hspace, this->hspace,
          this->d_info + i);

      MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
    }

    std::vector<int> h_info(this->batch_a_ptrs.size());
    cudaMemcpyAsync(h_info.data(), this->d_info, sizeof(int) * this->batch_a_ptrs.size(), cudaMemcpyDeviceToHost, stream);

    // This will block. Figure this out later
    cudaStreamSynchronize(stream);

    for (const auto& info : h_info) {
      if (info < 0) {
        MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
          ("Parameter " + std::to_string(-info) + " had an illegal value in cuSolver Xgetrf").c_str());
      } else {
        MATX_ASSERT_STR_EXP(info, 0, matxSolverError, 
          ("U is singular: U(" + std::to_string(info) + "," + std::to_string(info) + ") = 0 in cuSolver Xgetrf").c_str());
      }
    }
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
           (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test LU parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnLUCUDAParamsKeyEq {
  bool operator()(const DnLUCUDAParams_t &l, const DnLUCUDAParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size &&
           l.dtype == t.dtype;
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
 */
template <typename OutputTensor, typename PivotTensor, typename ATensor>
void lu_impl(OutputTensor &&out, PivotTensor &&piv,
        const ATensor &a, const cudaExecutor &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
    
  using T1 = typename remove_cvref_t<OutputTensor>::value_type;

  auto piv_new = OpToTensor(piv, exec);
  auto a_new = OpToTensor(a, exec);

  if(!a_new.isSameView(a)) {
    (a_new = a).run(exec);
  }

  /* Temporary WAR
     cuSolver doesn't support row-major layouts. Since we want to make the
     library appear as though everything is row-major, we take a performance hit
     to transpose in and out of the function. Eventually this may be fixed in
     cuSolver.
  */
  T1 *tp;
  matxAlloc(reinterpret_cast<void **>(&tp), a_new.Bytes(), MATX_ASYNC_DEVICE_MEMORY,
            exec.getStream());
  auto tv = TransposeCopy(tp, a_new, exec);
  auto tvt = tv.PermuteMatrix();

  // Get parameters required by these tensors
  auto params = detail::matxDnLUCUDAPlan_t<OutputTensor, decltype(piv_new), decltype(a_new)>::GetLUParams(piv_new, tvt);

  // Get cache or new LU plan if it doesn't exist
  using cache_val_type = detail::matxDnLUCUDAPlan_t<OutputTensor, decltype(piv_new), decltype(a_new)>;
  detail::GetCache().LookupAndExec<detail::lu_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::lu_cuda_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(piv_new, tvt);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tvt, piv_new, tvt, exec);
    }
  );

  /* Temporary WAR
   * Copy and free async buffer for transpose */
  matx::copy(out, tv.PermuteMatrix(), exec);
  matxFree(tp);
}

} // end namespace matx