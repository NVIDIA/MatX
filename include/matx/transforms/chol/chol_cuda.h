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
 * Parameters needed to execute a cholesky factorization. We distinguish unique
 * factorizations mostly by the data pointer in A
 */
struct DnCholCUDAParams_t {
  int64_t n;
  void *A;
  size_t batch_size;
  cublasFillMode_t uplo;
  MatXDataType_t dtype;
};

template <typename OutputTensor, typename ATensor>
class matxDnCholCUDAPlan_t : matxDnCUDASolver_t {
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
   * positive-definite where only the upper or lower triangle is used.
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
  matxDnCholCUDAPlan_t(const ATensor &a,
                         cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(RANK == remove_cvref_t<ATensor>::Rank(), matxInvalidDim,  "Cholesky input/output tensor ranks must match");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "Cholesky solver does not support half precision");
    MATX_STATIC_ASSERT_STR((std::is_same_v<T1, typename OutTensor_t::value_type>), matxInavlidType, "Input and Output types must match");

    params = GetCholParams(a, uplo);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize() override
  {
    cusolverStatus_t ret = cusolverDnXpotrf_bufferSize(this->handle, this->dn_params, params.uplo,
                                            params.n, MatXTypeToCudaType<T1>(),
                                            params.A, params.n,
                                            MatXTypeToCudaType<T1>(), &this->dspace,
                                            &this->hspace);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnCholCUDAParams_t GetCholParams(const ATensor &a,
                                      cublasFillMode_t uplo)
  {
    DnCholCUDAParams_t params;
    params.batch_size = GetNumBatches(a);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.uplo = uplo;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(OutputTensor &out, const ATensor &a,
            const cudaExecutor &exec, cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
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

    const auto stream = exec.getStream();
    cusolverDnSetStream(this->handle, stream);

    // At this time cuSolver does not have a batched 64-bit cholesky interface.
    // Change this to use the batched version once available.
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXpotrf(
          this->handle, this->dn_params, uplo, params.n, MatXTypeToCudaType<T1>(),
          this->batch_a_ptrs[i], params.n, MatXTypeToCudaType<T1>(),
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
          ("Parameter " + std::to_string(-info) + " had an illegal value in cuSolver Xpotrf").c_str());
      } else {
        MATX_ASSERT_STR_EXP(info, 0, matxSolverError, 
          (std::to_string(info) + "-th leading minor is not positive definite in cuSolver Xpotrf").c_str());
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
  ~matxDnCholCUDAPlan_t() {}

private:
  DnCholCUDAParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnCholCUDAParamsKeyHash {
  std::size_t operator()(const DnCholCUDAParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.n)) + (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test cholesky parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct DnCholCUDAParamsKeyEq {
  bool operator()(const DnCholCUDAParams_t &l, const DnCholCUDAParams_t &t) const
      noexcept
  {
    return l.n == t.n && l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

using chol_cuda_cache_t = std::unordered_map<DnCholCUDAParams_t, std::any, DnCholCUDAParamsKeyHash, DnCholCUDAParamsKeyEq>;

} // end namespace detail


/**
 * Perform a Cholesky decomposition using a cached plan
 *
 * See documentation of matxDnCholCUDAPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
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
 *   CUDA executor
 * @param uplo
 *   Part of matrix to fill
 */
template <typename OutputTensor, typename ATensor>
void chol_impl(OutputTensor &&out, const ATensor &a,
          const cudaExecutor &exec,
          SolverFillMode uplo = SolverFillMode::UPPER)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
    
  using OutputTensor_t = remove_cvref_t<OutputTensor>;
  using T1 = typename OutputTensor_t::value_type;
  constexpr int RANK = ATensor::Rank();

  auto a_new = OpToTensor(a, exec);

  if(!a_new.isSameView(a)) {
    (a_new = a).run(exec);
  }

  // cuSolver assumes column-major matrices and MatX uses row-major matrices.
  // One way to address this is to create a transposed copy of the input to
  // use with the factorization, followed by transposing the output. However,
  // for matrices with no additional padding, we can also change the value of
  // uplo to effectively change the matrix to column-major. This allows us to
  // compute the factorization without additional transposes. If we do not
  // have contiguous input and output tensors, then we create a temporary
  // contiguous tensor for use with cuSolver.
  uplo = (uplo == SolverFillMode::UPPER) ? SolverFillMode::LOWER : SolverFillMode::UPPER;

  const bool allContiguous = a_new.IsContiguous() && out.IsContiguous();
  auto tv = [allContiguous, &a_new, &out, &exec]() -> auto {
    if (allContiguous) {
      (out = a_new).run(exec);
      return out;
    } else{
      auto t = make_tensor<T1>(a_new.Shape(), MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
      (t = a_new).run(exec);
      return t;
    }
  }();

  cublasFillMode_t uplo_cusolver = (uplo == SolverFillMode::UPPER)? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  // Get parameters required by these tensors
  auto params = detail::matxDnCholCUDAPlan_t<OutputTensor, decltype(tv)>::GetCholParams(tv, uplo_cusolver);

  using cache_val_type = detail::matxDnCholCUDAPlan_t<OutputTensor, decltype(a_new)>;
  detail::GetCache().LookupAndExec<detail::chol_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::chol_cuda_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(tv, uplo_cusolver);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tv, tv, exec, uplo_cusolver);
    }
  );

  if (! allContiguous) {
    matx::copy(out, tv, exec);
  }
}

} // end namespace matx