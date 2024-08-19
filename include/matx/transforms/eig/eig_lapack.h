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

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/cache.h"
#include "matx/executors/host.h"
#include "matx/executors/support.h"
#include "matx/transforms/solver_common.h"

#include <cstdio>
#include <numeric>

namespace matx {

namespace detail {

#if MATX_EN_CPU_SOLVER
/**
 * Parameters needed to execute eigenvalue decomposition. We distinguish
 * unique factorizations mostly by the data pointer in A.
 */
struct DnEigHostParams_t {
  lapack_int_t n;
  char jobz;
  char uplo;
  void *A;
  void *out;
  void *W;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename OutputTensor, typename WTensor, typename ATensor>
class matxDnEigHostPlan_t : matxDnHostSolver_t<typename ATensor::value_type> {
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
   *   'V' to compute eigenvectors or
   *   'N' to not compute
   * @param uplo
   *   Where to store data in A: {'U' or 'L'}
   *
   */
  matxDnEigHostPlan_t(WTensor &w,
                        const ATensor &a,
                        const char jobz = 'V',
                        const char uplo = 'U')
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
    // Perform a workspace query with lwork = -1.
    lapack_int_t info;
    T1 work_query;
    T2 rwork_query;
    lapack_int_t iwork_query;

    // Use vector mode for a larger workspace size that works for both modes
    syevd_dispatch("V", &params.uplo, &params.n, nullptr, &params.n,
                    nullptr, &work_query, &this->lwork, &rwork_query,
                    &this->lrwork, &iwork_query, &this->liwork, &info);
    
    MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
      ("Parameter " + std::to_string(-info) + " had an illegal value in LAPACK syevd workspace query").c_str());

    // the real part of the first elem of work holds the optimal lwork.
    if constexpr (is_complex_v<T1>) {
      this->lwork = static_cast<lapack_int_t>(work_query.real());
      this->lrwork = static_cast<lapack_int_t>(rwork_query);
    } else {
      this->lwork = static_cast<lapack_int_t>(work_query);
      this->lrwork = 0; // Complex variants do not use rwork.
    }
    this->liwork = static_cast<lapack_int_t>(iwork_query);
  }

  static DnEigHostParams_t GetEigParams(WTensor &w,
                                    const ATensor &a,
                                    char jobz,
                                    char uplo)
  {
    DnEigHostParams_t params;
    params.batch_size = GetNumBatches(a);
    params.n = static_cast<lapack_int_t>(a.Size(RANK - 1));
    params.A = a.Data();
    params.W = w.Data();
    params.jobz = jobz;
    params.uplo = uplo;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  template <ThreadsMode MODE>
  void Exec(OutputTensor &out, WTensor &w,
            const ATensor &a,
            const HostExecutor<MODE> &exec,
            const char jobz = 'V',
            const char uplo = 'U')
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

    lapack_int_t info;
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      syevd_dispatch(&jobz, &uplo, &params.n,
                      reinterpret_cast<T1*>(this->batch_a_ptrs[i]),
                      &params.n, reinterpret_cast<T2*>(this->batch_w_ptrs[i]),
                      reinterpret_cast<T1*>(this->work), &this->lwork,
                      reinterpret_cast<T2*>(this->rwork), &this->lrwork,
                      reinterpret_cast<lapack_int_t*>(this->iwork), &this->liwork, &info);

      MATX_ASSERT_STR_EXP(info, 0, matxSolverError, 
          (std::to_string(info) + " off-diagonal elements of an intermediate tridiagonal form did not converge to zero in LAPACK syevd").c_str());
    }
  }

  /**
   * Eigen solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnEigHostPlan_t() {}

private:
  void syevd_dispatch(const char* jobz, const char* uplo, const lapack_int_t* n,
            T1* a, const lapack_int_t* lda, T2* w, T1* work_in,
            const lapack_int_t* lwork_in, [[maybe_unused]] T2* rwork_in,
            [[maybe_unused]] const lapack_int_t* lrwork_in, lapack_int_t* iwork_in,
            const lapack_int_t* liwork_in, lapack_int_t* info)
  {
    // TODO: remove warning suppression once syevd is optimized in NVPL LAPACK
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if constexpr (std::is_same_v<T1, float>) {
      LAPACK_CALL(ssyevd)(jobz, uplo, n, a, lda, w, work_in, lwork_in, iwork_in, liwork_in, info);
    } else if constexpr (std::is_same_v<T1, double>) {
      LAPACK_CALL(dsyevd)(jobz, uplo, n, a, lda, w, work_in, lwork_in, iwork_in, liwork_in, info);
    } else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
      LAPACK_CALL(cheevd)(jobz, uplo, n, a, lda, w, work_in, lwork_in, rwork_in, lrwork_in, iwork_in, liwork_in, info);
    } else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
      LAPACK_CALL(zheevd)(jobz, uplo, n, a, lda, w, work_in, lwork_in, rwork_in, lrwork_in, iwork_in, liwork_in, info);
    }
#pragma GCC diagnostic pop
  }

  std::vector<T2 *> batch_w_ptrs;
  DnEigHostParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnEigHostParamsKeyHash {
  std::size_t operator()(const DnEigHostParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.n)) + (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test Eigen parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct DnEigHostParamsKeyEq {
  bool operator()(const DnEigHostParams_t &l, const DnEigHostParams_t &t) const noexcept
  {
    return l.n == t.n && l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

using eig_Host_cache_t = std::unordered_map<DnEigHostParams_t, std::any, DnEigHostParamsKeyHash, DnEigHostParamsKeyEq>;
#endif

} // end namespace detail


/**
 * Perform a Eig decomposition using a cached plan
 *
 * See documentation of matxDnEigHostPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to a LAPACK
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
 *   Host executor
 * @param jobz
 *   EigenMode::VECTOR to compute eigenvectors or
 *   EigenMode::NO_VECTOR to not compute
 * @param uplo
 *   Where to store data in A
 */
template <typename OutputTensor, typename WTensor, typename ATensor, ThreadsMode MODE>
void eig_impl([[maybe_unused]] OutputTensor &&out,
              [[maybe_unused]] WTensor &&w,
              [[maybe_unused]] const ATensor &a,
              [[maybe_unused]] const HostExecutor<MODE> &exec,
              [[maybe_unused]] EigenMode jobz = EigenMode::VECTOR,
              [[maybe_unused]] SolverFillMode uplo = SolverFillMode::UPPER)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(MATX_EN_CPU_SOLVER, matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured");
#if MATX_EN_CPU_SOLVER

  using T1 = typename remove_cvref_t<OutputTensor>::value_type;

  auto w_new = OpToTensor(w, exec);
  auto a_new = OpToTensor(a, exec);

  if(!a_new.isSameView(a)) {
    (a_new = a).run(exec);
  }

  /* Temporary WAR
     LAPACK doesn't support row-major layouts. Since we want to make the
     library appear as though everything is row-major, we take a performance hit
     to transpose in and out of the function. LAPACKE, however, supports both formats.
  */
  T1 *tp;
  matxAlloc(reinterpret_cast<void **>(&tp), a_new.Bytes(), MATX_HOST_MALLOC_MEMORY);
  auto tv = TransposeCopy(tp, a_new, exec);

  const char jobz_lapack = (jobz == EigenMode::VECTOR) ? 'V' : 'N';
  const char uplo_lapack = (uplo == SolverFillMode::UPPER) ? 'U': 'L';

  // Get parameters required by these tensors
  auto params = detail::matxDnEigHostPlan_t<OutputTensor, decltype(w_new), decltype(a_new)>::
      GetEigParams(w_new, tv, jobz_lapack, uplo_lapack);

  // Get cache or new eigen plan if it doesn't exist
  using cache_val_type = detail::matxDnEigHostPlan_t<OutputTensor, decltype(w_new), decltype(a_new)>;
  detail::GetCache().LookupAndExec<detail::eig_Host_cache_t>(
    detail::GetCacheIdFromType<detail::eig_Host_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(w_new, tv, jobz_lapack, uplo_lapack);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tv, w_new, tv, exec, jobz_lapack, uplo_lapack);
    }
  );

  /* Copy and free async buffer for transpose */
  matx::copy(out, tv.PermuteMatrix(), exec);
  matxFree(tp);
#endif
}

} // end namespace matx