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
  SVDHostAlgo algo;
  lapack_int_t m;
  lapack_int_t n;
  char jobz;
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
   * @param jobz
   *   Specifies options for computing all, part, or none of the matrices U and VT. See
   *  SVDMode documentation for more info
   * @param algo
   *   Specifies the algorithm to use for computing SVD. Either QR based 'gesvd' or
   *  divide-and-conquer based 'gesdd'. See SVDHostAlgo documentation for more info
   */
  matxDnSVDHostPlan_t(UTensor &u,
                      STensor &s,
                      VtTensor &vt,
                      const ATensor &a,
                      const char jobz = 'A',
                      SVDHostAlgo algo = SVDHostAlgo::DC)
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

    params = GetSVDParams(u, s, vt, a, jobz, algo);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize() override
  {
    // Perform a workspace query with lwork = -1, using all mode for a
    // larger workspace size that works for all modes

    lapack_int_t info;
    T1 work_query;
    lapack_int_t mn = cuda::std::min(params.m, params.n);
    lapack_int_t mx = cuda::std::max(params.m, params.n);

    if (params.algo == SVDHostAlgo::QR) {
      gesvd_dispatch("A", "A", &params.m, &params.n, nullptr,
                    &params.m, nullptr, nullptr, &params.m, nullptr, &params.n,
                    &work_query, &this->lwork, nullptr, &info);

      MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
        ("Parameter " + std::to_string(-info) + " had an illegal value in LAPACK gesvd workspace query").c_str());

      // the real part of the first elem of work holds the optimal lwork.
      // rwork has size 5*min(M,N) and is only used for complex types
      if constexpr (is_complex_v<T1>) {
        this->lwork = static_cast<lapack_int_t>(work_query.real());
        this->lrwork = 5 * mn;
      } else {
        this->lwork = static_cast<lapack_int_t>(work_query);
        this->lrwork = 0;
      }
    } else if (params.algo == SVDHostAlgo::DC) {
      gesdd_dispatch("A", &params.m, &params.n, nullptr, &params.m, nullptr,
                    nullptr, &params.m, nullptr, &params.n, &work_query,
                    &this->lwork, nullptr, nullptr, &info);

      MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
        ("Parameter " + std::to_string(-info) + " had an illegal value in LAPACK gesdd workspace query").c_str());

      this->liwork = 8 * mn; // iwork has size 8*min(M,N) and is used for all types

      // the real part of the first elem of work holds the optimal lwork.
      if constexpr (is_complex_v<T1>) {
        this->lwork = static_cast<lapack_int_t>(work_query.real());
        
        lapack_int_t mnthr = (mn * 5) / 3;
        if (mx >= mnthr) {
          this->lrwork = 5*mn*mn + 5*mn;
        } else {
          this->lrwork = cuda::std::max(5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn);
        }
      } else {
        this->lwork = static_cast<lapack_int_t>(work_query);
        this->lrwork = 0; // rwork is not used for real types
      }
    } else {
      MATX_THROW(matxInvalidType, "Invalid SVD host algorithm");
    }
  }

  static DnSVDHostParams_t
  GetSVDParams(UTensor &u, STensor &s,
               VtTensor &vt, const ATensor &a,
               const char jobz = 'A', const SVDHostAlgo algo = SVDHostAlgo::DC)
  {
    DnSVDHostParams_t params;
    params.batch_size = GetNumBatches(a);
    params.m = static_cast<lapack_int_t>(a.Size(RANK - 2));
    params.n = static_cast<lapack_int_t>(a.Size(RANK - 1));
    params.algo = algo;
    params.A = a.Data();
    params.U = u.Data();
    params.VT = vt.Data();
    params.S = s.Data();
    params.jobz = jobz;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  template<ThreadsMode MODE>
  void Exec(UTensor &u, STensor &s, VtTensor &vt,
            const ATensor &a, [[maybe_unused]] const HostExecutor<MODE> &exec,
            const char jobz = 'A')
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
    if (params.algo == SVDHostAlgo::QR) {
      for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
        gesvd_dispatch(&jobz, &jobz, &params.m, &params.n,
                        reinterpret_cast<T1*>(this->batch_a_ptrs[i]),
                        &params.m, reinterpret_cast<T3*>(this->batch_s_ptrs[i]),
                        reinterpret_cast<T1*>(this->batch_u_ptrs[i]), &params.m,
                        reinterpret_cast<T1*>(this->batch_vt_ptrs[i]), &params.n,
                        reinterpret_cast<T1*>(this->work), &this->lwork,
                        reinterpret_cast<T3*>(this->rwork), &info);

        MATX_ASSERT_STR_EXP(info, 0, matxSolverError, 
          (std::to_string(info) + " superdiagonals of an intermediate bidiagonal form did not converge to zero in LAPACK").c_str());
      }
    } else if (params.algo == SVDHostAlgo::DC) {
      for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
        gesdd_dispatch(&jobz, &params.m, &params.n,
                        reinterpret_cast<T1*>(this->batch_a_ptrs[i]),
                        &params.m, reinterpret_cast<T3*>(this->batch_s_ptrs[i]),
                        reinterpret_cast<T1*>(this->batch_u_ptrs[i]), &params.m,
                        reinterpret_cast<T1*>(this->batch_vt_ptrs[i]), &params.n,
                        reinterpret_cast<T1*>(this->work), &this->lwork,
                        reinterpret_cast<T3*>(this->rwork),
                        reinterpret_cast<lapack_int_t*>(this->iwork), &info);

        MATX_ASSERT_STR_EXP(info, 0, matxSolverError, "gesdd error in LAPACK");     
      }
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

  void gesdd_dispatch(const char *jobz, const lapack_int_t *m, const lapack_int_t *n,
                      T1 *a, const lapack_int_t *lda, T3 *s, T1 *u, const lapack_int_t *ldu,
                      T1 *vt, const lapack_int_t *ldvt, T1 *work_in, const lapack_int_t *lwork_in,
                      [[maybe_unused]] T3 *rwork_in, lapack_int_t *iwork_in, lapack_int_t *info)
  {
    if constexpr (std::is_same_v<T1, float>) {
      LAPACK_CALL(sgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work_in, lwork_in, iwork_in, info);
    } else if constexpr (std::is_same_v<T1, double>) {
      LAPACK_CALL(dgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work_in, lwork_in, iwork_in, info);
    } else if constexpr  (std::is_same_v<T1, cuda::std::complex<float>>) {
      LAPACK_CALL(cgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work_in, lwork_in, rwork_in, iwork_in, info);
    } else if constexpr   (std::is_same_v<T1, cuda::std::complex<double>>)  {
      LAPACK_CALL(zgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work_in, lwork_in, rwork_in, iwork_in, info);
    }
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
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size && l.dtype == t.dtype &&
           l.algo == t.algo;
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
 * @param jobz
 *   Specifies options for computing all, part, or none of the matrices U and VT. See
 * SVDMode documentation for more info
 * @param algo
 *   Specifies the algorithm to use for computing SVD. Either QR based 'gesvd' or
 * divide-and-conquer based 'gesdd'. See SVDHostAlgo documentation for more info
 */
template <typename UTensor, typename STensor, typename VtTensor, typename ATensor, ThreadsMode MODE>
void svd_impl([[maybe_unused]] UTensor &&u,
              [[maybe_unused]] STensor &&s,
              [[maybe_unused]] VtTensor &&vt,
              [[maybe_unused]] const ATensor &a,
              [[maybe_unused]] const HostExecutor<MODE> &exec,
              [[maybe_unused]] const SVDMode jobz = SVDMode::ALL,
              [[maybe_unused]] const SVDHostAlgo algo = SVDHostAlgo::DC)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(MATX_EN_CPU_SOLVER, matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured");
#if MATX_EN_CPU_SOLVER

  using T1 = typename ATensor::value_type;
  constexpr int RANK = ATensor::Rank();

  auto s_new = getSolverSupportedTensor(s, exec);
  auto u_new = getSolverSupportedTensor(u, exec);
  auto vt_new = getSolverSupportedTensor(vt, exec);

  /* Temporary WAR
     LAPACK assumes column-major matrices and MatX uses row-major matrices.
     One way to address this is to create a transposed copy of the input to
     use with the factorization, followed by transposing the outputs. For SVD,
     we can skip this by passing in a permuted view of row-major A, which is
     equivalent to col-major AT, and swapping the inputs U and VT.
  */

  // LAPACK destroys the input, so we need to make a copy of A regardless  
  auto a_copy = make_tensor<T1>(a.Shape(), MATX_HOST_MALLOC_MEMORY);
  (a_copy = a).run(exec);
  auto at_col_maj = transpose_matrix(a_copy);

  // swap U and VT
  auto u_in = vt_new;
  auto vt_in = u_new;

  const char job_lapack = detail::SVDModeToChar(jobz);

  // Get parameters required by these tensors
  auto params = detail::matxDnSVDHostPlan_t<decltype(u_in), decltype(s_new), decltype(vt_in), decltype(at_col_maj)>::
    GetSVDParams(u_in, s_new, vt_in, at_col_maj, job_lapack, algo);

  // Get cache or new SVD plan if it doesn't exist
  using cache_val_type = detail::matxDnSVDHostPlan_t<decltype(u_in), decltype(s_new), decltype(vt_in), decltype(at_col_maj)>;
  detail::GetCache().LookupAndExec<detail::svd_Host_cache_t>(
    detail::GetCacheIdFromType<detail::svd_Host_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(u_in, s_new, vt_in, at_col_maj, job_lapack, algo);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(u_in, s_new, vt_in, at_col_maj, exec, job_lapack);
    }
  );


  if(!s_new.isSameView(s)) {
    (s = s_new).run(exec);
  }
  if(!u_new.isSameView(u)) {
    (u = u_new).run(exec);
  }
  if(!vt_new.isSameView(vt)) {
    (vt = vt_new).run(exec);
  }
#endif
}

} // end namespace matx

