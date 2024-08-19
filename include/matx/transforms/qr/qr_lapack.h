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
 * Parameters needed to execute a QR factorization. We distinguish unique
 * factorizations mostly by the data pointer in A
 */
struct DnQRHostParams_t {
  lapack_int_t m;
  lapack_int_t n;
  void *A;
  void *tau;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename OutTensor, typename TauTensor, typename ATensor>
class matxDnQRHostPlan_t : matxDnHostSolver_t<typename ATensor::value_type> {
  using OutTensor_t = remove_cvref_t<OutTensor>;
  using T1 = typename ATensor::value_type;
  using T2 = typename TauTensor::value_type;
  static constexpr int RANK = OutTensor_t::Rank();
  static_assert(RANK >= 2, "Input/Output tensor must be rank 2 or higher");

public:
  /**
   * Plan for factoring A such that \f$\textbf{A} = \textbf{Q} * \textbf{R}\f$
   *
   * Creates a handle for factoring matrix A into the format above. QR
   * decomposition in LAPACK does not return the Q matrix directly, and
   * it must be computed separately used the Householder reflections in the tau
   * output, along with the overwritten A matrix input. The input and output
   * parameters may be the same tensor. In that case, the input is destroyed and
   * the output is stored in-place.
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam T2
   *  Data type of Tau vector
   * @tparam RANK
   *  Rank of A matrix
   *
   * @param tau
   *   Scaling factors for reflections
   * @param a
   *   Input tensor view
   *
   */
  matxDnQRHostPlan_t(TauTensor &tau,
                       const ATensor &a)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(RANK-1 == TauTensor::Rank(), matxInvalidDim, "Tau tensor must be one rank less than output tensor");
    MATX_STATIC_ASSERT_STR(RANK == ATensor::Rank(), matxInvalidDim, "Output tensor must match A tensor rank in QR");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "QR solver does not support half precision");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T1, typename OutTensor_t::value_type>), matxInavlidType, "Input and Output types must match");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T1, T2>), matxInavlidType, "A and Tau types must match");

    params = GetQRParams(tau, a);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize() override
  {
    // perform workspace query with lwork = -1
    lapack_int_t info;
    T1 work_query;

    geqrf_dispatch(&params.m, &params.n, nullptr,
                    &params.m, nullptr,
                    &work_query, &this->lwork, &info);
      MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
        ("Parameter " + std::to_string(-info) + " had an illegal value in LAPACK geqrf workspace query").c_str());
    
    // the real part of the first elem of work holds the optimal lwork
    if constexpr (is_complex_v<T1>) {
      this->lwork = static_cast<lapack_int_t>(work_query.real());
    } else {
      this->lwork = static_cast<lapack_int_t>(work_query);
    }
  }

  static DnQRHostParams_t GetQRParams(TauTensor &tau,
                                  const ATensor &a)
  {
    DnQRHostParams_t params;

    params.batch_size = GetNumBatches(a);
    params.m = static_cast<lapack_int_t>(a.Size(RANK - 2));
    params.n = static_cast<lapack_int_t>(a.Size(RANK - 1));
    params.A = a.Data();
    params.tau = tau.Data();
    params.dtype = TypeToInt<T1>();

    return params;
  }

  template <ThreadsMode MODE>
  void Exec(OutTensor &out, TauTensor &tau,
            const ATensor &a, const HostExecutor<MODE> &exec)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Batch size checks
    for(int i = 0 ; i < RANK-2; i++) {
      MATX_ASSERT_STR(out.Size(i) == a.Size(i), matxInvalidDim, "Out and A must have the same batch sizes");
      MATX_ASSERT_STR(tau.Size(i) == a.Size(i), matxInvalidDim, "Tau and A must have the same batch sizes");
    }

    // Inner size checks
    MATX_ASSERT_STR((out.Size(RANK-2) == params.m) && (out.Size(RANK-1) == params.n), matxInvalidSize, "Out and A shapes do not match");
    MATX_ASSERT_STR(tau.Size(RANK-2) == cuda::std::min(params.m, params.n), matxInvalidSize, "Tau must be ... x min(m,n)");

    SetBatchPointers<BatchType::MATRIX>(out, this->batch_a_ptrs);
    SetBatchPointers<BatchType::VECTOR>(tau, this->batch_tau_ptrs);

    if (out.Data() != a.Data()) {
      (out = a).run(exec);
    }

    lapack_int_t info;
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      geqrf_dispatch(&params.m, &params.n, reinterpret_cast<T1*>(this->batch_a_ptrs[i]),
                     &params.m, reinterpret_cast<T1*>(this->batch_tau_ptrs[i]),
                     reinterpret_cast<T1*>(this->work), &this->lwork, &info);

      MATX_ASSERT_STR_EXP(info, 0, matxSolverError, "LAPACK geqrf error");
    }
  }

  /**
   * QR solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnQRHostPlan_t() {}

private:
  void geqrf_dispatch(const lapack_int_t* m, const lapack_int_t* n, T1* a,
                      const lapack_int_t* lda, T1* tau, T1* work_in,
                      const lapack_int_t* lwork_in, lapack_int_t* info)
  {
    if constexpr (std::is_same_v<T1, float>) {
      LAPACK_CALL(sgeqrf)(m, n, a, lda, tau, work_in, lwork_in, info);
    } else if constexpr (std::is_same_v<T1, double>) {
      LAPACK_CALL(dgeqrf)(m, n, a, lda, tau, work_in, lwork_in, info);
    } else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
      LAPACK_CALL(cgeqrf)(m, n, a, lda, tau, work_in, lwork_in, info);
    } else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
      LAPACK_CALL(zgeqrf)(m, n, a, lda, tau, work_in, lwork_in, info);
    }
  }
  
  std::vector<T2 *> batch_tau_ptrs;
  DnQRHostParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnQRHostParamsKeyHash {
  std::size_t operator()(const DnQRHostParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test QR parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnQRHostParamsKeyEq {
  bool operator()(const DnQRHostParams_t &l, const DnQRHostParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size &&
           l.dtype == t.dtype;
  }
};

using qr_Host_cache_t = std::unordered_map<DnQRHostParams_t, std::any, DnQRHostParamsKeyHash, DnQRHostParamsKeyEq>;
#endif

} // end namespace detail

/**
 * Perform a QR decomposition using a cached plan
 *
 * See documentation of matxDnQRHostPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to a LAPACK
 * library by deducing all parameters needed to perform a QR decomposition from
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
 * @param tau
 *   Output of reflection scalar values
 * @param a
 *   Input tensor A
 * @param exec
 *   Host executor
 */
template <typename OutTensor, typename TauTensor, typename ATensor, ThreadsMode MODE>
void qr_solver_impl([[maybe_unused]] OutTensor &&out,
                      [[maybe_unused]] TauTensor &&tau,
                      [[maybe_unused]] const ATensor &a,
                      [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(MATX_EN_CPU_SOLVER, matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured");
#if MATX_EN_CPU_SOLVER
  
  using T1 = typename remove_cvref_t<OutTensor>::value_type;

  auto tau_new = OpToTensor(tau, exec);
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

  // Get parameters required by these tensors
  auto params = detail::matxDnQRHostPlan_t<OutTensor, decltype(tau_new), decltype(a_new)>::GetQRParams(tau_new, tvt);

  // Get cache or new QR plan if it doesn't exist
  using cache_val_type = detail::matxDnQRHostPlan_t<OutTensor, decltype(tau_new), decltype(a_new)>;
  detail::GetCache().LookupAndExec<detail::qr_Host_cache_t>(
    detail::GetCacheIdFromType<detail::qr_Host_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(tau_new, tvt);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tvt, tau_new, tvt, exec);
    }
  );

  /* Temporary WAR
   * Copy and free async buffer for transpose */
  matx::copy(out, tv.PermuteMatrix(), exec);
  matxFree(tp);
#endif
}

} // end namespace matx
