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
#include "matx/operators/slice.h"
#include "matx/executors/host.h"
#include "matx/executors/support.h"
#include "matx/transforms/solver_common.h"

#include <cstdio>
#include <numeric>

namespace matx {

namespace detail {

#if MATX_EN_CPU_SOLVER
/**
 * Parameters needed to execute singular value decomposition. We distinguish
 * unique factorizations mostly by the data pointer in A.
 */
struct DnSVDHostParams_t {
  lapack_int_t m;
  lapack_int_t n;
  char jobu;
  char jobvt;
  void *A;
  void *U;
  void *VT;
  void *S;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename UTensor, typename STensor, typename VtTensor, typename ATensor>
class matxDnSVDHostPlan_t : matxDnHostSolver_t<typename ATensor::value_type> {
  using T1 = typename ATensor::value_type;
  using T2 = typename UTensor::value_type;
  using T3 = typename STensor::value_type;
  using T4 = typename VtTensor::value_type;
  static constexpr int RANK = UTensor::Rank();
  static_assert(RANK >= 2, "Input/Output tensor must be rank 2 or higher");

public:
  /**
   * Plan for factoring A such that \f$\textbf{A} = \textbf{U} * \textbf{\Sigma}
   * * \textbf{V^{H}}\f$
   *
   * Creates a handle for decomposing matrix A into the format above. LAPACK destroys
   * the contents of A, so a copy of the user input should be passed here.
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam T2
   *  Data type of U matrix
   * @tparam T3
   *  Data type of S vector
   * @tparam T4
   *  Data type of VT matrix
   * @tparam RANK
   *  Rank of A, U, and VT matrices, and RANK-1 of S
   *
   * @param u
   *   Output tensor view for U matrix
   * @param s
   *   Output tensor view for S matrix
   * @param vt
   *   Output tensor view for VT matrix
   * @param a
   *   Input tensor view for A matrix
   * @param jobu
   *   Specifies options for computing all or part of the matrix U: = 'A'. See
   *  SVDJob documentation for more info
   * @param jobvt
   *   specifies options for computing all or part of the matrix V**T. See
   * SVDJob documentation for more info
   *
   */
  matxDnSVDHostPlan_t(UTensor &u,
                      STensor &s,
                      VtTensor &vt,
                      const ATensor &a, const char jobu = 'A',
                      const char jobvt = 'A')
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(UTensor::Rank()-1 == STensor::Rank(), matxInvalidDim, "S tensor must be 1 rank lower than U tensor in SVD");
    MATX_STATIC_ASSERT_STR(UTensor::Rank() == ATensor::Rank(), matxInvalidDim, "U tensor must match A tensor rank in SVD");
    MATX_STATIC_ASSERT_STR(UTensor::Rank() == VtTensor::Rank(), matxInvalidDim, "U tensor must match VT tensor rank in SVD");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "SVD solver does not support half precision");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T1, T2>), matxInavlidType, "A and U types must match");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T1, T4>), matxInavlidType, "A and VT types must match");
    MATX_STATIC_ASSERT_STR(!is_complex_v<T3>, matxInvalidType, "S type must be real");
    MATX_STATIC_ASSERT_STR((std::is_same_v<typename inner_op_type_t<T1>::type, T3>), matxInvalidType, "A and S inner types must match");

    params = GetSVDParams(u, s, vt, a, jobu, jobvt);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize() override
  {
    // Perform a workspace query with lwork = -1.

    lapack_int_t info;
    T1 work_query;
    // Use all mode for a larger workspace size that works for all modes
    gesvd_dispatch("A", "A", &params.m, &params.n, nullptr,
                  &params.m, nullptr, nullptr, &params.m, nullptr, &params.n,
                  &work_query, &this->lwork, nullptr, &info);

    MATX_ASSERT(info == 0, matxSolverError);

    // the real part of the first elem of work holds the optimal lwork.
    // rwork has size 5*min(M,N) and is only used for complex types
    if constexpr (is_complex_v<T1>) {
      this->lwork = static_cast<lapack_int_t>(work_query.real());
      this->lrwork = 5 * cuda::std::min(params.m, params.n);
    } else {
      this->lwork = static_cast<lapack_int_t>(work_query);
      this->lrwork = 0; // rwork is not used for real types
    }
  }

  static DnSVDHostParams_t
  GetSVDParams(UTensor &u, STensor &s,
               VtTensor &vt, const ATensor &a,
               const char jobu = 'A', const char jobvt = 'A')
  {
    DnSVDHostParams_t params;
    params.batch_size = GetNumBatches(a);
    params.m = static_cast<lapack_int_t>(a.Size(RANK - 2));
    params.n = static_cast<lapack_int_t>(a.Size(RANK - 1));
    params.A = a.Data();
    params.U = u.Data();
    params.VT = vt.Data();
    params.S = s.Data();
    params.jobu = jobu;
    params.jobvt = jobvt;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  template<ThreadsMode MODE>
  void Exec(UTensor &u, STensor &s, VtTensor &vt,
            const ATensor &a, [[maybe_unused]] const HostExecutor<MODE> &exec,
            const char jobu = 'A', const char jobvt = 'A')
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Batch size checks
    for(int i = 0 ; i < RANK-2; i++) {
      MATX_ASSERT_STR(u.Size(i) == a.Size(i), matxInvalidDim, "U and A must have the same batch sizes");
      MATX_ASSERT_STR(vt.Size(i) == a.Size(i), matxInvalidDim, "VT and A must have the same batch sizes");
      MATX_ASSERT_STR(s.Size(i) == a.Size(i), matxInvalidDim, "S and A must have the same batch sizes");
    }

    // Inner size checks
    MATX_ASSERT_STR((u.Size(RANK-1) == params.m) && (u.Size(RANK-2) == u.Size(RANK-1)), matxInvalidSize, "U must be ... x m x m");
    MATX_ASSERT_STR((vt.Size(RANK-1) == params.n) && (vt.Size(RANK-2) == vt.Size(RANK-1)), matxInvalidSize, "VT must be ... x n x n");
    MATX_ASSERT_STR(s.Size(RANK-2) == cuda::std::min(params.m, params.n), matxInvalidSize, "S must be ... x min(m,n)");

    SetBatchPointers<BatchType::MATRIX>(a, this->batch_a_ptrs);
    SetBatchPointers<BatchType::MATRIX>(u, this->batch_u_ptrs);
    SetBatchPointers<BatchType::MATRIX>(vt, this->batch_vt_ptrs);
    SetBatchPointers<BatchType::VECTOR>(s, this->batch_s_ptrs);

    lapack_int_t info;
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      gesvd_dispatch(&jobu, &jobvt, &params.m, &params.n,
                      reinterpret_cast<T1*>(this->batch_a_ptrs[i]),
                      &params.m, reinterpret_cast<T3*>(this->batch_s_ptrs[i]),
                      reinterpret_cast<T1*>(this->batch_u_ptrs[i]), &params.m,
                      reinterpret_cast<T1*>(this->batch_vt_ptrs[i]), &params.n,
                      reinterpret_cast<T1*>(this->work), &this->lwork,
                      reinterpret_cast<T3*>(this->rwork), &info);

      MATX_ASSERT(info == 0, matxSolverError);
    }
  }

  /**
   * SVD solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnSVDHostPlan_t() {}

private:
  void gesvd_dispatch(const char *jobu, const char *jobvt, const lapack_int_t *m,
                      const lapack_int_t *n, T1 *a,
                      const lapack_int_t *lda, T3 *s, T1 *u,
                      const lapack_int_t *ldu, T1 *vt,
                      const lapack_int_t *ldvt, T1 *work_in,
                      const lapack_int_t *lwork_in, [[maybe_unused]] T3 *rwork_in, lapack_int_t *info)
  {
    // TODO: remove warning suppression once gesvd is optimized in NVPL LAPACK
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if constexpr (std::is_same_v<T1, float>) {
      LAPACK_CALL(sgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work_in, lwork_in, info);
    } else if constexpr (std::is_same_v<T1, double>) {
      LAPACK_CALL(dgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work_in, lwork_in, info);
    } else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
      LAPACK_CALL(cgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work_in, lwork_in, rwork_in, info);
    } else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
      LAPACK_CALL(zgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work_in, lwork_in, rwork_in, info);
    }
#pragma GCC diagnostic pop
  }
  
  std::vector<T2 *> batch_u_ptrs;
  std::vector<T3 *> batch_s_ptrs;
  std::vector<T4 *> batch_vt_ptrs;
  DnSVDHostParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnSVDHostParamsKeyHash {
  std::size_t operator()(const DnSVDHostParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test SVD parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnSVDHostParamsKeyEq {
  bool operator()(const DnSVDHostParams_t &l, const DnSVDHostParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

using svd_Host_cache_t = std::unordered_map<DnSVDHostParams_t, std::any, DnSVDHostParamsKeyHash, DnSVDHostParamsKeyEq>;
#endif

}

/**
 * Perform a SVD decomposition using a cached plan
 *
 * See documentation of matxDnSVDHostPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the LAPACK
 * library by deducing all parameters needed to perform a SVD decomposition from
 * only the matrix A.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param u
 *   U matrix output
 * @param s
 *   Sigma matrix output
 * @param vt
 *   VT matrix output
 * @param a
 *   Input matrix A
 * @param exec
 *   Host Executor
 * @param jobu
 *   Specifies options for computing all or part of the matrix U: = 'A'. See
 * SVDJob documentation for more info
 * @param jobvt
 *   specifies options for computing all or part of the matrix V**T. See
 * SVDJob documentation for more info
 *
 */
template <typename UTensor, typename STensor, typename VtTensor, typename ATensor, ThreadsMode MODE>
void svd_impl([[maybe_unused]] UTensor &&u,
              [[maybe_unused]] STensor &&s,
              [[maybe_unused]] VtTensor &&vt,
              [[maybe_unused]] const ATensor &a,
              [[maybe_unused]] const HostExecutor<MODE> &exec,
              [[maybe_unused]] const SVDJob jobu = SVDJob::ALL,
              [[maybe_unused]] const SVDJob jobvt = SVDJob::ALL)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(MATX_EN_CPU_SOLVER, matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured");
#if MATX_EN_CPU_SOLVER

  using T1 = typename ATensor::value_type;
  constexpr int RANK = ATensor::Rank();

  auto u_new = OpToTensor(u, exec);
  auto s_new = OpToTensor(s, exec);
  auto vt_new = OpToTensor(vt, exec);
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
  auto tvt = tv.PermuteMatrix();
  
  auto u_col_maj = make_tensor<T1>(u_new.Shape(), MATX_HOST_MALLOC_MEMORY);
  auto vt_col_maj = make_tensor<T1>(vt_new.Shape(), MATX_HOST_MALLOC_MEMORY);

  const char jobu_lapack = detail::SVDJobToChar(jobu);
  const char jobvt_lapack = detail::SVDJobToChar(jobvt);

  // Get parameters required by these tensors
  auto params = detail::matxDnSVDHostPlan_t<decltype(u_new), decltype(s_new), decltype(vt_new), decltype(tvt)>::
    GetSVDParams(u_col_maj, s_new, vt_col_maj, tvt, jobu_lapack, jobvt_lapack);

  // Get cache or new QR plan if it doesn't exist
  using cache_val_type = detail::matxDnSVDHostPlan_t<decltype(u_col_maj), decltype(s_new), decltype(vt_col_maj), decltype(tvt)>;
  detail::GetCache().LookupAndExec<detail::svd_Host_cache_t>(
    detail::GetCacheIdFromType<detail::svd_Host_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(u_col_maj, s_new, vt_col_maj, tvt, jobu_lapack, jobvt_lapack);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(u_col_maj, s_new, vt_col_maj, tvt, exec, jobu_lapack, jobvt_lapack);
    }
  );

  // LAPACK writes to them in col-major format, so we need to transpose them back.
  (u = transpose_matrix(u_col_maj)).run(exec);
  (vt = transpose_matrix(vt_col_maj)).run(exec);

  matxFree(tp);
#endif
}

} // end namespace matx

