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
 * Parameters needed to execute eigenvalue decomposition. We distinguish
 * unique factorizations mostly by the data pointer in A.
 */
struct DnEigCUDAParams_t {
  int64_t n;
  cusolverEigMode_t jobz;
  cublasFillMode_t uplo;
  void *A;
  void *out;
  void *W;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename OutputTensor, typename WTensor, typename ATensor>
class matxDnEigCUDAPlan_t : matxDnCUDASolver_t {
public:
  using OutTensor_t = remove_cvref_t<OutputTensor>;
  using T1 = typename ATensor::value_type;
  using T2 = typename WTensor::value_type;
  static constexpr int RANK = OutTensor_t::Rank();
  static_assert(RANK >= 2, "Input/Output tensor must be rank 2 or higher");

  /**
   * Plan computing eigenvalues/vectors on square Hermitian A such that:
   *
   * \f$\textbf{A} * textbf{V} = \textbf{V} * \textbf{\Lambda}\f$
   *
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam T2
   *  Data type of W matrix
   * @tparam RANK
   *  Rank of A matrix
   *
   * @param w
   *   Eigenvalues of A
   * @param a
   *   Input tensor view
   * @param jobz
   *   CUSOLVER_EIG_MODE_VECTOR to compute eigenvectors or
   * CUSOLVER_EIG_MODE_NOVECTOR to not compute
   * @param uplo
   *   Where to store data in A
   *
   */
  matxDnEigCUDAPlan_t(WTensor &w,
                        const ATensor &a,
                        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR,
                        cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(RANK == ATensor::Rank(), matxInvalidDim, "Output and A tensor ranks must match for eigen solver");
    MATX_STATIC_ASSERT_STR(RANK - 1 == WTensor::Rank(), matxInvalidDim, "W tensor must be one rank lower than output for eigen solver");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "Eigen solver does not support half precision");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T1, typename OutTensor_t::value_type>), matxInavlidType, "Input and output types must match");
    MATX_STATIC_ASSERT_STR(!is_complex_v<T2>, matxInvalidType, "W type must be real");
    MATX_STATIC_ASSERT_STR((std::is_same_v<typename inner_op_type_t<T1>::type, T2>), matxInvalidType, "Out and W inner types must match");

    params = GetEigParams(w, a, jobz, uplo);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize() override
  {
    // Use vector mode for a larger workspace size that works for both modes
    cusolverStatus_t ret = cusolverDnXsyevd_bufferSize(
                    this->handle, this->dn_params, CUSOLVER_EIG_MODE_VECTOR, 
                    params.uplo, params.n, MatXTypeToCudaType<T1>(), params.A,
                    params.n, MatXTypeToCudaType<T2>(), params.W,
                    MatXTypeToCudaType<T1>(), &this->dspace,
                    &this->hspace);

    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnEigCUDAParams_t GetEigParams(WTensor &w,
                                    const ATensor &a,
                                    cusolverEigMode_t jobz,
                                    cublasFillMode_t uplo)
  {
    DnEigCUDAParams_t params;
    params.batch_size = GetNumBatches(a);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.W = w.Data();
    params.jobz = jobz;
    params.uplo = uplo;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(OutputTensor &out, WTensor &w,
            const ATensor &a,
            const cudaExecutor &exec,
            cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR,
            cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(a.Size(RANK - 1) == a.Size(RANK - 2), matxInvalidSize, "Input to eigen must be a square matrix");

    // Ensure output & w size matches input
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT(out.Size(i) == a.Size(i), matxInvalidSize);
      if (i < RANK - 1) {
        MATX_ASSERT(out.Size(i) == w.Size(i), matxInvalidSize);
      }
    }

    SetBatchPointers<BatchType::MATRIX>(out, this->batch_a_ptrs);
    SetBatchPointers<BatchType::VECTOR>(w, this->batch_w_ptrs);

    if (out.Data() != a.Data()) {
      (out = a).run(exec);
    }

    const auto stream = exec.getStream();
    cusolverDnSetStream(this->handle, stream);

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXsyevd(
          this->handle, this->dn_params, jobz, uplo, params.n, MatXTypeToCudaType<T1>(),
          this->batch_a_ptrs[i], params.n, MatXTypeToCudaType<T2>(), this->batch_w_ptrs[i],
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
          ("Parameter " + std::to_string(-info) + " had an illegal value in cuSolver Xsyevd").c_str());
      } else {
        MATX_ASSERT_STR_EXP(info, 0, matxSolverError, 
            (std::to_string(info) + " off-diagonal elements of an intermediate tridiagonal form did not converge to zero in cuSolver Xsyevd").c_str());
      }
    }
  }

  /**
   * Eigen solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnEigCUDAPlan_t() {}

private:
  std::vector<T2 *> batch_w_ptrs;
  DnEigCUDAParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnEigCUDAParamsKeyHash {
  std::size_t operator()(const DnEigCUDAParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.n)) + (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test Eigen parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct DnEigCUDAParamsKeyEq {
  bool operator()(const DnEigCUDAParams_t &l, const DnEigCUDAParams_t &t) const noexcept
  {
    return l.n == t.n && l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

using eig_cuda_cache_t = std::unordered_map<DnEigCUDAParams_t, std::any, DnEigCUDAParamsKeyHash, DnEigCUDAParamsKeyEq>;

} // end namespace detail


/**
 * Perform a Eig decomposition using a cached plan
 *
 * See documentation of matxDnEigCUDAPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
 * library by deducing all parameters needed to perform a eigen decomposition
 * from only the matrix A. The input and output parameters may be the same
 * tensor. In that case, the input is destroyed and the output is stored
 * in-place. Input must be a Hermitian or real symmetric matrix.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor view
 * @param w
 *   Eigenvalues output
 * @param a
 *   Input matrix A
 * @param exec
 *   CUDA executor
 * @param jobz
 *   EigenMode::VECTOR to compute eigenvectors or
 *   EigenMode::NO_VECTOR to not compute
 * @param uplo
 *   Where to store data in A
 */
template <typename OutputTensor, typename WTensor, typename ATensor>
void eig_impl(OutputTensor &&out, WTensor &&w,
         const ATensor &a, const cudaExecutor &exec,
         EigenMode jobz = EigenMode::VECTOR,
         SolverFillMode uplo = SolverFillMode::UPPER)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using T1 = typename remove_cvref_t<OutputTensor>::value_type;

  auto w_new = OpToTensor(w, exec);
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
  
  cusolverEigMode_t jobz_cusolver = (jobz == EigenMode::VECTOR) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  cublasFillMode_t uplo_cusolver = (uplo == SolverFillMode::UPPER) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  // Get parameters required by these tensors
  auto params = detail::matxDnEigCUDAPlan_t<OutputTensor, decltype(w_new), decltype(a_new)>::
      GetEigParams(w_new, tv, jobz_cusolver, uplo_cusolver);

  // Get cache or new eigen plan if it doesn't exist
  using cache_val_type = detail::matxDnEigCUDAPlan_t<OutputTensor, decltype(w_new), decltype(a_new)>;
  detail::GetCache().LookupAndExec<detail::eig_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::eig_cuda_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(w_new, tv, jobz_cusolver, uplo_cusolver);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tv, w_new, tv, exec, jobz_cusolver, uplo_cusolver);
    }
  );

  /* Copy and free async buffer for transpose */
  matx::copy(out, tv.PermuteMatrix(), exec);
  matxFree(tp);
}

} // end namespace matx