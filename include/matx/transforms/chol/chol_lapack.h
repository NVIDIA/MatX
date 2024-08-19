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
 * Parameters needed to execute a cholesky factorization. We distinguish unique
 * factorizations mostly by the data pointer in A
 */
struct DnCholHostParams_t {
  lapack_int_t n;
  void *A;
  size_t batch_size;
  char uplo;
  MatXDataType_t dtype;
};

template <typename OutputTensor, typename ATensor>
class matxDnCholHostPlan_t : matxDnHostSolver_t<typename remove_cvref_t<ATensor>::value_type> {
  using OutTensor_t = remove_cvref_t<OutputTensor>;
  using T1 = typename remove_cvref_t<ATensor>::value_type;
  static constexpr int RANK = OutTensor_t::Rank();
  static_assert(RANK >= 2, "Input/Output tensor must be rank 2 or higher");

public:
  /**
   * Plan for solving
   * \f$\textbf{A} = \textbf{L} * \textbf{L^{H}}\f$ or \f$\textbf{A} =
   * \textbf{U} * \textbf{U^{H}}\f$ using the Cholesky method
   *
   * Creates a handle for solving the factorization of A = M * M^H of a dense
   * matrix using the Cholesky method, where M is either the upper or lower
   * triangular portion of A. Input matrix A must be a square Hermitian matrix
   * positive-definite where only the upper or lower triangle is used. This does
   * require a workspace.
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam RANK
   *  Rank of A matrix
   *
   * @param a
   *   Input tensor view
   * @param uplo
   *   Use upper or lower triangle for computation
   *
   */
  matxDnCholHostPlan_t(const ATensor &a,
                         const char uplo = 'U')
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(RANK == remove_cvref_t<ATensor>::Rank(), matxInvalidDim,  "Cholesky input/output tensor ranks must match");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "Cholesky solver does not support half precision");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T1, typename OutTensor_t::value_type>), matxInavlidType, "Input and Output types must match");

    params = GetCholParams(a, uplo);
  }

  static DnCholHostParams_t GetCholParams(const ATensor &a,
                                      const char uplo)
  {
    DnCholHostParams_t params;
    params.batch_size = GetNumBatches(a);
    params.n = static_cast<lapack_int_t>(a.Size(RANK - 1));
    params.A = a.Data();
    params.uplo = uplo;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  template <ThreadsMode MODE>
  void Exec(OutputTensor &out, const ATensor &a,
            const HostExecutor<MODE> &exec, const char uplo = 'U')
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(a.Size(RANK - 1) == a.Size(RANK - 2), matxInvalidSize, "Input to Cholesky must be a square matrix");

    // Ensure output size matches input
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT(out.Size(i) == a.Size(i), matxInvalidSize);
    }

    SetBatchPointers<BatchType::MATRIX>(out, this->batch_a_ptrs);

    if (out.Data() != a.Data()) {
      (out = a).run(exec);
    }

    lapack_int_t info;

    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      potrf_dispatch(&uplo, &params.n,
                     reinterpret_cast<T1*>(this->batch_a_ptrs[i]),
                     &params.n, &info);

      if (info < 0) {
        MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
          ("Parameter " + std::to_string(-info) + " had an illegal value in LAPACK potrf").c_str());
      } else {
        MATX_ASSERT_STR_EXP(info, 0, matxSolverError, 
          (std::to_string(info) + "-th leading minor is not positive definite in LAPACK potrf").c_str());
      }
    }
  }

  /**
   * Cholesky solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnCholHostPlan_t() {}

private:
  void potrf_dispatch(const char* uplo, const lapack_int_t* n, T1* a,
                      const lapack_int_t* lda, lapack_int_t* info)
  {
    if constexpr (std::is_same_v<T1, float>) {
      LAPACK_CALL(spotrf)(uplo, n, a, lda, info);
    } else if constexpr (std::is_same_v<T1, double>) {
      LAPACK_CALL(dpotrf)(uplo, n, a, lda, info);
    } else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
      LAPACK_CALL(cpotrf)(uplo, n, a, lda, info);
    } else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
      LAPACK_CALL(zpotrf)(uplo, n, a, lda, info);
    }
  }
  
  DnCholHostParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnCholHostParamsKeyHash {
  std::size_t operator()(const DnCholHostParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.n)) + (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test cholesky parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct DnCholHostParamsKeyEq {
  bool operator()(const DnCholHostParams_t &l, const DnCholHostParams_t &t) const
      noexcept
  {
    return l.n == t.n && l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

using chol_Host_cache_t = std::unordered_map<DnCholHostParams_t, std::any, DnCholHostParamsKeyHash, DnCholHostParamsKeyEq>;
#endif

} // end namespace detail


/**
 * Perform a Cholesky decomposition using a cached plan
 *
 * See documentation of matxDnCholHostPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the LAPACK
 * library by deducing all parameters needed to perform a Cholesky decomposition
 * from only the matrix A. The input and output parameters may be the same
 * tensor. In that case, the input is destroyed and the output is stored
 * in-place. Input must be a positive-definite Hermitian or real symmetric matrix.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor
 * @param a
 *   Input tensor
 * @param exec
 *   Host executor
 * @param uplo
 *   Part of matrix to fill
 */
template <typename OutputTensor, typename ATensor, ThreadsMode MODE>
void chol_impl([[maybe_unused]] OutputTensor &&out,
               [[maybe_unused]] const ATensor &a,
               [[maybe_unused]] const HostExecutor<MODE> &exec,
               [[maybe_unused]] SolverFillMode uplo = SolverFillMode::UPPER)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(MATX_EN_CPU_SOLVER, matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured");
#if MATX_EN_CPU_SOLVER
    
  using OutputTensor_t = remove_cvref_t<OutputTensor>;
  using T1 = typename OutputTensor_t::value_type;
  constexpr int RANK = ATensor::Rank();

  auto a_new = OpToTensor(a, exec);

  if(!a_new.isSameView(a)) {
    (a_new = a).run(exec);
  }

  // LAPACK assumes column-major matrices and MatX uses row-major matrices.
  // One way to address this is to create a transposed copy of the input to
  // use with the factorization, followed by transposing the output. However,
  // for matrices with no additional padding, we can also change the value of
  // uplo to effectively change the matrix to column-major. This allows us to
  // compute the factorization without additional transposes. If we do not
  // have contiguous input and output tensors, then we create a temporary
  // contiguous tensor for use with LAPACK.
  uplo = (uplo == SolverFillMode::UPPER) ? SolverFillMode::LOWER : SolverFillMode::UPPER;

  const bool allContiguous = a_new.IsContiguous() && out.IsContiguous();
  auto tv = [allContiguous, &a_new, &out, &exec]() -> auto {
    if (allContiguous) {
      (out = a_new).run(exec);
      return out;
    } else{
      auto t = make_tensor<T1>(a_new.Shape(), MATX_HOST_MALLOC_MEMORY);
      (t = a_new).run(exec);
      return t;
    }
  }();

  const char uplo_lapack = (uplo == SolverFillMode::UPPER)? 'U' : 'L';

  // Get parameters required by these tensors
  auto params = detail::matxDnCholHostPlan_t<OutputTensor, decltype(tv)>::GetCholParams(tv, uplo_lapack);

  using cache_val_type = detail::matxDnCholHostPlan_t<OutputTensor, decltype(a_new)>;
  detail::GetCache().LookupAndExec<detail::chol_Host_cache_t>(
    detail::GetCacheIdFromType<detail::chol_Host_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(tv, uplo_lapack);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tv, tv, exec, uplo_lapack);
    }
  );

  if (! allContiguous) {
    matx::copy(out, tv, exec);
  }
#endif
}

} // end namespace matx