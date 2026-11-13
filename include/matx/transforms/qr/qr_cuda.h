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
#include "matx/operators/slice.h"
#include "matx/operators/index.h"
#include "matx/transforms/solver_common.h"

#include <cstdio>
#include <numeric>
#include <cuda/std/__algorithm/min.h>

namespace matx {

namespace detail {

  template<typename AType>
    inline auto qr_internal_workspace(const AType &A, cudaStream_t stream) {
      using ATypeS = typename AType::value_type;
      const int RANK = AType::Rank();

      index_t m = A.Size(RANK-2);

      cuda::std::array<index_t, RANK-1> uShape;
      for(int i = 0; i < RANK-2; i++) {
        uShape[i] = A.Size(i);
      }
      uShape[RANK-2] = A.Size(RANK-1);

      auto QShape = A.Shape();
      QShape[RANK-1] = m;

      auto Qin = make_tensor<ATypeS>(QShape, MATX_ASYNC_DEVICE_MEMORY, stream);
      auto wwt = make_tensor<ATypeS>(QShape, MATX_ASYNC_DEVICE_MEMORY, stream);
      auto u = make_tensor<ATypeS>(uShape, MATX_ASYNC_DEVICE_MEMORY, stream);

      return cuda::std::make_tuple(Qin, wwt, u);
    }

  template<typename QType, typename RType, typename AType, typename WType>
    inline void qr_internal(QType &Q, RType &R, const AType &A, WType workspace, const cudaExecutor &exec) {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
      const auto stream = exec.getStream();

      static_assert(AType::Rank() >= 2);
      static_assert(QType::Rank() == AType::Rank());
      static_assert(RType::Rank() == AType::Rank());

      MATX_ASSERT_STR(AType::Rank() == QType::Rank(), matxInvalidDim, "qr: A and Q must have the same rank");
      MATX_ASSERT_STR(AType::Rank() == RType::Rank(), matxInvalidDim, "qr: A and R must have the same rank");

      using ATypeS = typename AType::value_type;
      const int RANK = AType::Rank();

      index_t m = A.Size(RANK-2);
      index_t n = A.Size(RANK-1);
      index_t k = cuda::std::min(m,n);
      if(m<=n) k--;  // these matrices have one less update since the diagonal ends on the bottom of the matrix

      auto Qin = cuda::std::get<0>(workspace);
      auto wwt  = cuda::std::get<1>(workspace);
      auto u = cuda::std::get<2>(workspace);

      static_assert(decltype(Qin)::Rank() == QType::Rank());
      static_assert(decltype(wwt)::Rank() == QType::Rank());
      static_assert(decltype(u)::Rank() == QType::Rank()-1);

      // Create Identity matrix
      auto E = eye<ATypeS>({m, m});

      // Clone over batch Dims
      auto ECShape = Q.Shape();
      ECShape[RANK-1] = matxKeepDim;
      ECShape[RANK-2] = matxKeepDim;

      auto I = clone<RANK>(E, ECShape);

      // Inititalize Q
      (Q = I).run(stream);
      (R = A).run(stream);

      // we will slice X directly from R.
      cuda::std::array<index_t, RANK> xSliceB, xSliceE;
      xSliceB.fill(0); xSliceE.fill(matxEnd);
      xSliceE[RANK-1] = matxDropDim; // drop last dim to make a vector


      // v is of size m x 1.  Instead of allocating additional memory we will just reuse a row of Qin
      cuda::std::array<index_t, RANK> vSliceB, vSliceE;
      vSliceB.fill(0); vSliceE.fill(matxEnd);
      // select a single row of Q to alias as v
      vSliceE[RANK-2] = matxDropDim;
      auto v = slice<RANK-1>(Qin, vSliceB, vSliceE);
      auto xz = v; // alias


      // N is of size 1.  Instead of allocating additional memory we will just reuse an entry of Qin
      cuda::std::array<index_t, RANK> nSliceB, nSliceE;
      nSliceB.fill(0); nSliceE.fill(matxEnd);
      // select a single row of Q to alias as v
      nSliceE[RANK-2] = matxDropDim;
      nSliceE[RANK-1] = matxDropDim;

      auto N = slice<RANK-2>(wwt, nSliceB, nSliceE);

      // N cloned with RANK-2 of size m.
      cuda::std::array<index_t, RANK-1> ncShape;
      ncShape.fill(matxKeepDim);
      ncShape[RANK-2] = m;
      auto nc = clone<RANK-1>(N,ncShape);

      // aliasing some memory here to share storage and provide clarity in the code below
      auto s = N; // alias
      auto sc = nc; // alias
      auto w = v; // alias

      for(int i = 0 ; i < k ; i++) {

        // slice off a column of R and alias as x
        xSliceB[RANK-1] = i;
        auto x = slice<RANK-1>(R, xSliceB, xSliceE);

        // operator which zeros out values above current index in matrix
        (xz = as_type<typename inner_op_type_t<ATypeS>::type >(index(x.Rank()-1) >= i) * x).run(stream);

        // compute L2 norm without sqrt.
        (N = sum(abs2(xz))).run(stream);
        //(N = sqrt(N)).run(stream);  // sqrt folded into next op

        (v = xz + as_type<typename inner_op_type_t<ATypeS>::type >(index(v.Rank()-1) == i) * sign(xz) * sqrt(nc)).run(stream);

        auto r = x;  // alias column of R happens to be the same as x

        (s = sum(abs2(v))).run(stream);
        //(s = sqrt(s)).run(stream);  // sqrt folded into next op

        // IFELSE to avoid nans when dividing by zero
        using v_type = typename decltype(v)::value_type;
        (IFELSE(sc != v_type(0),
                w = (v / sqrt(sc)),
                w = v_type(0))).run(stream);

        (u = matvec(conj(transpose_matrix(R)), w, 2 , 0)).run(stream);

        (R = outer(w, conj(u), -1, 1)).run(stream);

        // entries below diagonal should be numerical zero.  Zero them out to avoid additional FP error.
        (IF(index(x.Rank()-1) > i, r = ATypeS(0)) ).run(stream);

        (wwt = outer(w, conj(w))).run(stream);

        (Qin = Q).run(stream);  // save input
        matmul_impl(Q, Qin, wwt, exec, -2, 1);

      }
    }
} // end namespace detail


/**
 * Perform QR decomposition on a matrix using housholders reflections. If rank > 2, operations are batched.
 *
 * @tparam QType
 *   Tensor or operator type for output of Q matrix or tensor output.
 * @tparam RType
 *   Tensor or operator type for output of R matrix
 * @tparam AType
 *   Tensor or operator type for output of A input tensors.
 *
 * @param Q
 *   Q output tensor or operator.
 * @param R
 *   R output tensor or operator.
 * @param A
 *   Input tensor or operator for tensor A input.
 * @param exec
 *   CUDA executor
 */
template<typename QType, typename RType, typename AType>
inline void qr_impl(QType &Q, RType &R, const AType &A, const cudaExecutor &exec) {
  const auto stream = exec.getStream();
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  static_assert(AType::Rank() >= 2);
  static_assert(QType::Rank() == AType::Rank());
  static_assert(RType::Rank() == AType::Rank());

  MATX_ASSERT_STR(AType::Rank() == QType::Rank(), matxInvalidDim, "qr: A and Q must have the same rank");
  MATX_ASSERT_STR(AType::Rank() == RType::Rank(), matxInvalidDim, "qr: A and R must have the same rank");

  auto workspace = detail::qr_internal_workspace(A, stream);
  detail::qr_internal(Q,R,A,workspace,exec);
}


/********************************************** SOLVER QR
 * *********************************************/

namespace detail {

/**
 * Parameters needed to execute a QR factorization. We distinguish unique
 * factorizations mostly by the data pointer in A
 */
struct DnQRCUDAParams_t {
  index_t rank;
  index_t m;
  index_t n;
  void *A;
  void *tau;
  size_t batch_size;
  MatXDataType_t dtype;
  cudaExecutor exec;
};

template <typename OutTensor, typename TauTensor, typename ATensor>
class matxDnQRCUDAPlan_t : matxDnCUDASolver_t {
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
   * decomposition in cuBLAS/cuSolver does not return the Q matrix directly, and
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
  matxDnQRCUDAPlan_t(TauTensor &tau,
                       const ATensor &a,
                       const cudaExecutor &exec)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(RANK-1 == TauTensor::Rank(), matxInvalidDim, "Tau tensor must be one rank less than output tensor");
    MATX_STATIC_ASSERT_STR(RANK == ATensor::Rank(), matxInvalidDim, "Output tensor must match A tensor rank in QR");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "QR solver does not support half precision");
    MATX_STATIC_ASSERT_STR((cuda::std::is_same_v<T1, typename OutTensor_t::value_type>), matxInavlidType, "Input and Output types must match");
    MATX_STATIC_ASSERT_STR((cuda::std::is_same_v<T1, T2>), matxInavlidType, "A and Tau types must match");

    params = GetQRParams(tau, a, exec);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size, false, exec);
  }

  void GetWorkspaceSize() override
  {
    CheckSolverLibraryStatus(cusolverDnXgeqrf_bufferSize(
            this->handle, this->dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
            params.A, params.m, MatXTypeToCudaType<T2>(), params.tau,
            MatXTypeToCudaType<T1>(), &this->dspace, &this->hspace),
        "cuSolver Xgeqrf_bufferSize");
  }

  static DnQRCUDAParams_t GetQRParams(TauTensor &tau,
                                  const ATensor &a,
                                  const cudaExecutor &exec)
  {
    DnQRCUDAParams_t params;
    params.rank = RANK;
    params.batch_size = GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.tau = tau.Data();
    params.dtype = TypeToInt<T1>();
    params.exec = exec;
    return params;
  }

  /**
   * Run geqrf on every matrix.
   *
   * @param status_info If non-null, each matrix's info is written to
   * status_info[i], and the call does not synchronize. Otherwise info is copied
   * back after a stream synchronization and checked in debug builds only.
   * cuSolver return-status errors throw in every build type.
   */
  void Exec(OutTensor &out, TauTensor &tau,
            const ATensor &a, const cudaExecutor &exec, int *status_info = nullptr)
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

    const auto stream = exec.getStream();
    CheckSolverLibraryStatus(cusolverDnSetStream(this->handle, stream), "cusolverDnSetStream");

    // With a status, info goes into it instead of being checked here
    const bool has_status = status_info != nullptr;
    int *infos = has_status ? status_info : this->d_info;

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      const auto ret = cusolverDnXgeqrf(
          this->handle, this->dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
          this->batch_a_ptrs[i], params.m, MatXTypeToCudaType<T2>(),
          this->batch_tau_ptrs[i], MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(this->d_workspace) + i * this->dspace, this->dspace,
          reinterpret_cast<uint8_t *>(this->h_workspace) + i * this->hspace, this->hspace,
          infos + i);
      CheckSolverLibraryStatus(ret, "cuSolver Xgeqrf");
    }

    if (!has_status) {
      std::vector<int> h_info(this->batch_a_ptrs.size());
      CheckSolverInfosOnHost(this->d_info, h_info, stream, "cuSolver Xgeqrf");
    }
  }

  /**
   * QR solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnQRCUDAPlan_t() {}

private:
  std::vector<T2 *> batch_tau_ptrs;
  DnQRCUDAParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnQRCUDAParamsKeyHash {
  std::size_t operator()(const DnQRCUDAParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.rank)) + (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size)) + (std::hash<uint64_t>()((uint64_t)(k.exec.getStream())));
  }
};

/**
 * Test QR parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnQRCUDAParamsKeyEq {
  bool operator()(const DnQRCUDAParams_t &l, const DnQRCUDAParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size &&
           l.dtype == t.dtype && l.exec.getStream() == t.exec.getStream() &&
           l.rank == t.rank;
  }
};

using qr_cuda_cache_t = std::unordered_map<DnQRCUDAParams_t, std::any, DnQRCUDAParamsKeyHash, DnQRCUDAParamsKeyEq>;

// qr_solver and qr_econ use the same key type but store different plan types,
// so qr_econ needs its own cache. The distinct hash type gives this map, and so
// its cache ID, a type distinct from qr_cuda_cache_t.
struct DnEconQRCUDAParamsKeyHash : DnQRCUDAParamsKeyHash {};
using qr_econ_cuda_cache_t = std::unordered_map<DnQRCUDAParams_t, std::any, DnEconQRCUDAParamsKeyHash, DnQRCUDAParamsKeyEq>;

} // end namespace detail

/**
 * Perform a QR decomposition using a cached plan
 *
 * See documentation of matxDnQRCUDAPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
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
 *   CUDA executor
 * @param status
 *   Optional solver status. When given, geqrf info is written into it on the
 *   stream, and the call does not synchronize the stream.
 */
template <typename OutTensor, typename TauTensor, typename ATensor>
void qr_solver_impl(OutTensor &&out, TauTensor &&tau,
        const ATensor &a, const cudaExecutor &exec, detail::SolverStatusState *status = nullptr)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using T1 = typename remove_cvref_t<OutTensor>::value_type;

  // Validate the status before any work, so a rejected status allocates nothing.
  // One stage: geqrf.
  const auto status_run = detail::BeginStatusRun(detail::ExecutorKind::Cuda, status, 1, a);
  int *status_info = status_run.Info();

  auto tau_new = OpToTensor(tau, exec);
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
  auto params = detail::matxDnQRCUDAPlan_t<OutTensor, decltype(tau_new), decltype(a_new)>::GetQRParams(tau_new, tvt, exec);

  // Get cache or new QR plan if it doesn't exist
  using cache_val_type = detail::matxDnQRCUDAPlan_t<OutTensor, decltype(tau_new), decltype(a_new)>;
  auto cache_id = detail::GetCacheIdFromType<detail::qr_cuda_cache_t>();
  MATX_LOG_DEBUG("QR transform (full): cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::qr_cuda_cache_t>(
    cache_id,
    params,
    [&]() {
      return std::make_shared<cache_val_type>(tau_new, tvt, exec);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tvt, tau_new, tvt, exec, status_info);
    },
    exec
  );

  /* Temporary WAR
   * Copy out of the transpose buffer, which tp_tensor frees on return */
  matx::copy(out, tv.PermuteMatrix(), exec);
}

/********************************************** Economic QR
 * *********************************************/

namespace detail {

template<class>
inline constexpr bool always_false_v = false;

template <typename OutTensor, typename RTensor, typename TauTensor, typename ATensor>
class matxDnEconQRCUDAPlan_t : matxDnCUDASolver_t {
  using OutTensor_t = remove_cvref_t<OutTensor>;
  using RTensor_t = remove_cvref_t<RTensor>;
  using T1 = typename ATensor::value_type;
  using T2 = typename TauTensor::value_type;
  static constexpr int RANK = OutTensor_t::Rank();
  static_assert(RANK >= 2, "Input/Output tensor must be rank 2 or higher");

public:
  /**
   * Plan for factoring A such that \f$\textbf{A} = \textbf{Q} * \textbf{R}\f$
   *
   * Creates a handle for factoring matrix A into the format above. QR
   * decomposition in cuBLAS/cuSolver does not return the Q matrix directly, and
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
  matxDnEconQRCUDAPlan_t(TauTensor &tau,
                       const ATensor &a,
                       const cudaExecutor &exec)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Dim checks
    MATX_STATIC_ASSERT_STR(RANK-1 == TauTensor::Rank(), matxInvalidDim, "Tau tensor must be one rank less than output tensor");
    MATX_STATIC_ASSERT_STR(RANK == ATensor::Rank(), matxInvalidDim, "Output tensor must match A tensor rank in QR");
    MATX_STATIC_ASSERT_STR(RANK == RTensor_t::Rank(), matxInvalidDim, "Output tensor R must match A tensor rank in QR");

    // Type checks
    MATX_STATIC_ASSERT_STR(!is_half_v<T1>, matxInvalidType, "QR solver does not support half precision");
    MATX_STATIC_ASSERT_STR((cuda::std::is_same_v<T1, typename OutTensor_t::value_type>), matxInavlidType, "Input and Output types must match");
    MATX_STATIC_ASSERT_STR((cuda::std::is_same_v<T1, typename RTensor_t::value_type>), matxInavlidType, "Input and Output types must match");
    MATX_STATIC_ASSERT_STR((cuda::std::is_same_v<T1, T2>), matxInavlidType, "A and Tau types must match");

    params = GetQRParams(tau, a, exec);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size, false, exec, 2);
  }

  void GetWorkspaceSize() override
  {
    CheckSolverLibraryStatus(cusolverDnXgeqrf_bufferSize(
            this->handle, this->dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
            params.A, params.m, MatXTypeToCudaType<T2>(), params.tau,
            MatXTypeToCudaType<T1>(), &this->dspace, &this->hspace),
        "cuSolver Xgeqrf_bufferSize");

    // Query orgqr with the same sizes Exec passes it. orgqr's lwork counts
    // elements of T1, while dspace counts bytes.
    const int64_t k = std::min(params.m, params.n);
    CheckSolverLibraryStatus(orgqr_bufferSize_dispatch(
            params.m, k, k, params.A, params.m, params.tau, &orgqr_lwork),
        "cuSolver orgqr_bufferSize");

    this->dspace = std::max(this->dspace,
        static_cast<size_t>(orgqr_lwork) * sizeof(T1));
  }

  static DnQRCUDAParams_t GetQRParams(TauTensor &tau,
                                  const ATensor &a,
                                  const cudaExecutor &exec)
  {
    DnQRCUDAParams_t params;
    params.rank = RANK;
    params.batch_size = GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.tau = tau.Data();
    params.dtype = TypeToInt<T1>();
    params.exec = exec;
    return params;
  }

  /**
   * Run geqrf, extract R, then run orgqr on every matrix.
   *
   * @param status_info If non-null, geqrf info for matrix i is written to
   * status_info[i] and orgqr info to status_info[batches + i], and the call
   * does not synchronize. Otherwise info is copied back after each stage with a
   * stream synchronization and checked in debug builds only. cuSolver
   * return-status errors throw in every build type.
   */
  void Exec(OutTensor &out, RTensor &out_r, TauTensor &tau,
            const ATensor &a, const cudaExecutor &exec, int *status_info = nullptr)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Batch size checks
    for(int i = 0 ; i < RANK-2; i++) {
      MATX_ASSERT_STR(out.Size(i) == a.Size(i), matxInvalidDim, "Out and A must have the same batch sizes");
      MATX_ASSERT_STR(tau.Size(i) == a.Size(i), matxInvalidDim, "Tau and A must have the same batch sizes");
      MATX_ASSERT_STR(out.Size(i) == out_r.Size(i), matxInvalidDim, "Out and R must have the same batch sizes");
    }

    // Inner size checks
    MATX_ASSERT_STR((out.Size(RANK-2) == params.m) && (out.Size(RANK-1) == params.n), matxInvalidSize, "Out and A shapes do not match");
    MATX_ASSERT_STR(tau.Size(RANK-2) == cuda::std::min(params.m, params.n), matxInvalidSize, "Tau must be ... x min(m,n)");
    MATX_ASSERT_STR((out_r.Size(RANK-2) == cuda::std::min(params.m, params.n)) && (out_r.Size(RANK-1) == params.n), matxInvalidSize, "R and out shapes are not compatible");

    SetBatchPointers<BatchType::MATRIX>(out, this->batch_a_ptrs);
    SetBatchPointers<BatchType::VECTOR>(tau, this->batch_tau_ptrs);

    if (out.Data() != a.Data()) {
      (out = a).run(exec);
    }

    const auto stream = exec.getStream();
    CheckSolverLibraryStatus(cusolverDnSetStream(this->handle, stream), "cusolverDnSetStream");

    // With a status, info goes into it instead of being checked here.
    // geqrf info is the first `batches` entries and orgqr info the next, in
    // either d_info or the status's two stages.
    const bool has_status = status_info != nullptr;
    const size_t batches = this->batch_a_ptrs.size();
    int *geqrf_info = has_status ? status_info : this->d_info;
    int *orgqr_info = geqrf_info + batches;

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < batches; i++) {
      const auto ret = cusolverDnXgeqrf(
          this->handle, this->dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
          this->batch_a_ptrs[i], params.m, MatXTypeToCudaType<T2>(),
          this->batch_tau_ptrs[i], MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(this->d_workspace) + i * this->dspace, this->dspace,
          reinterpret_cast<uint8_t *>(this->h_workspace) + i * this->hspace, this->hspace,
          geqrf_info + i);
      CheckSolverLibraryStatus(ret, "cuSolver Xgeqrf");
    }

    // Without a status, each stage's info is checked on the host, reusing one buffer
    std::vector<int> h_info(has_status ? 0 : batches);
    if (!has_status) {
      CheckSolverInfosOnHost(geqrf_info, h_info, stream, "cuSolver Xgeqrf");
    }

    // at this point, R is stored in the upper triangular part of out
    // so copy it out into memory assigned by user.
    // Per the cuSolver documentation (quoted):
    // The matrix R is overwritten in upper triangular part of A,
    // including diagonal elements
    cuda::std::array<index_t, RANK> rSliceB, rSliceE;
    rSliceB.fill(0); rSliceE.fill(matxEnd);
    rSliceE[RANK-2] = matxDropDim;
    rSliceE[RANK-1] = params.n; // taking upper triangular portion of out

    for (int i = 0; i < std::min(params.m, params.n); i++){
      rSliceB[RANK-2] = i;
      (slice<RANK-1>(out_r, rSliceB, rSliceE) = as_type<typename OutTensor_t::value_type>(index(RANK-2) >= i)*slice<RANK-1>(out, rSliceB, rSliceE)).run(exec);
    }

    for (size_t i = 0; i < batches; i++) {
      const auto ret = orgqr_dispatch(
            params.m, std::min(params.m, params.n), std::min(params.m, params.n),
            this->batch_a_ptrs[i], params.m, this->batch_tau_ptrs[i],
            this->d_workspace, orgqr_lwork, orgqr_info + i);
      CheckSolverLibraryStatus(ret, "cuSolver orgqr");
    }

    if (!has_status) {
      CheckSolverInfosOnHost(orgqr_info, h_info, stream, "cuSolver orgqr");
    }
  }

  /**
   * QR solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnEconQRCUDAPlan_t() {}

private:

  cusolverStatus_t orgqr_bufferSize_dispatch(
    int64_t m64, int64_t n64, int64_t k64,
    void* A, int64_t lda64,
    void* tau,
    int* lwork){

    int m = static_cast<int>(m64);
    int n = static_cast<int>(n64);
    int k = static_cast<int>(k64);
    int lda = static_cast<int>(lda64);

    if constexpr (cuda::std::is_same_v<T1, float>) {
      return cusolverDnSorgqr_bufferSize(handle, m, n, k,
              reinterpret_cast<T1*>(A), lda,
              reinterpret_cast<T1*>(tau),
              lwork);
    }
    else if constexpr (cuda::std::is_same_v<T1, double>) {
      return cusolverDnDorgqr_bufferSize(handle, m, n, k,
        reinterpret_cast<T1*>(A), lda,
        reinterpret_cast<T1*>(tau),
              lwork);
    }
    else if constexpr (cuda::std::is_same_v<T1, cuda::std::complex<float>>) {
      return cusolverDnCungqr_bufferSize(handle, m, n, k,
              reinterpret_cast<cuComplex*>(A), lda,
              reinterpret_cast<cuComplex*>(tau),
              lwork);
    }
    else if constexpr (cuda::std::is_same_v<T1, cuda::std::complex<double>>) {
      return cusolverDnZungqr_bufferSize(handle, m, n, k,
        reinterpret_cast<cuDoubleComplex*>(A), lda,
        reinterpret_cast<cuDoubleComplex*>(tau),
              lwork);
    }
    else{
      // should never reach here
      static_assert(always_false_v<T1>, "Unsupported type for economic QR");
    }

    return CUSOLVER_STATUS_NOT_SUPPORTED;
  }

  cusolverStatus_t orgqr_dispatch(
    int64_t m64, int64_t n64, int64_t k64,
    void* A, int64_t lda64,
    void* tau,
    void* work, int64_t lwork64,
    int* info)
  {
    int m = static_cast<int>(m64);
    int n = static_cast<int>(n64);
    int k = static_cast<int>(k64);
    int lda = static_cast<int>(lda64);
    int lwork = static_cast<int>(lwork64);

    if constexpr (cuda::std::is_same_v<T1, float>) {
      return cusolverDnSorgqr(handle, m, n, k,
              reinterpret_cast<T1*>(A), lda,
              reinterpret_cast<T1*>(tau),
              reinterpret_cast<T1*>(work), lwork, info);
    }
    else if constexpr (cuda::std::is_same_v<T1, double>) {
      return cusolverDnDorgqr(handle, m, n, k,
              reinterpret_cast<T1*>(A), lda,
              reinterpret_cast<T1*>(tau),
              reinterpret_cast<T1*>(work), lwork, info);
    }
    else if constexpr (cuda::std::is_same_v<T1, cuda::std::complex<float>>) {
      return cusolverDnCungqr(handle, m, n, k,
              reinterpret_cast<cuComplex*>(A), lda,
              reinterpret_cast<cuComplex*>(tau),
              reinterpret_cast<cuComplex*>(work), lwork, info);
    }
    else if constexpr (cuda::std::is_same_v<T1, cuda::std::complex<double>>) {
      return cusolverDnZungqr(handle, m, n, k,
              reinterpret_cast<cuDoubleComplex*>(A), lda,
              reinterpret_cast<cuDoubleComplex*>(tau),
              reinterpret_cast<cuDoubleComplex*>(work), lwork, info);
    }
    else{
      // should never reach here
      static_assert(always_false_v<T1>, "Unsupported type for economic QR");
    }

    return CUSOLVER_STATUS_NOT_SUPPORTED;
  }

  std::vector<T2 *> batch_tau_ptrs;
  DnQRCUDAParams_t params;
  int orgqr_lwork = 0; // orgqr workspace size in elements of T1
};


} // end namespace detail

/**
 * Perform a QR decomposition using a cached plan
 *
 * See documentation of matxDnQRCUDAPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
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
 *   Orthogonal matrix Q
 * @param out_r
 *   Upper triangular output matrix R
 * @param a
 *   Input tensor A
 * @param exec
 *   CUDA executor
 * @param status
 *   Optional solver status. When given, geqrf info is written to stage 0 and
 *   orgqr info to stage 1, and the call does not
 *   synchronize the stream.
 */
template <typename OutTensor, typename RTensor, typename ATensor>
void qr_econ_impl(OutTensor &&out, RTensor &&out_r,
        const ATensor &a, const cudaExecutor &exec, detail::SolverStatusState *status = nullptr)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using T1 = typename remove_cvref_t<OutTensor>::value_type;
  const int RANK = ATensor::Rank();

  // Validate the status before any work, so a rejected status allocates nothing.
  // Two stages: geqrf, then orgqr.
  const auto status_run = detail::BeginStatusRun(detail::ExecutorKind::Cuda, status, 2, a);
  int *status_info = status_run.Info();

  cuda::std::array<index_t, RANK-1> tau_shape;
  for(int i = 0; i < RANK-2; i++) {
    tau_shape[i] = a.Size(i);
  }
  tau_shape[RANK-2] = cuda::std::min(a.Size(RANK-1), a.Size(RANK-2));

  // Stream-ordered allocation, so freeing tau does not synchronize the device
  auto tau_new = make_tensor<T1>(tau_shape, MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
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
  auto params = detail::matxDnEconQRCUDAPlan_t<OutTensor, RTensor, decltype(tau_new), decltype(a_new)>::GetQRParams(tau_new, tvt, exec);

  // Get cache or new QR plan if it doesn't exist
  using cache_val_type = detail::matxDnEconQRCUDAPlan_t<OutTensor, RTensor, decltype(tau_new), decltype(a_new)>;
  auto cache_id = detail::GetCacheIdFromType<detail::qr_econ_cuda_cache_t>();
  MATX_LOG_DEBUG("QR transform (economic): cache_id={}", cache_id);
  detail::GetCache().LookupAndExec<detail::qr_econ_cuda_cache_t>(
    cache_id,
    params,
    [&]() {
      return std::make_shared<cache_val_type>(tau_new, tvt, exec);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(tvt, out_r, tau_new, tvt, exec, status_info);
    },
    exec
  );

  /* Temporary WAR
   * Copy out of the transpose buffer, which tp_tensor frees on return */
  cuda::std::array<index_t, RANK> rSliceB, rSliceE;
  rSliceB.fill(0); rSliceE.fill(matxEnd);
  rSliceE[RANK-2] = params.m;
  rSliceE[RANK-1] = std::min(params.m, params.n);
  matx::copy(out, slice(tv.PermuteMatrix(), rSliceB, rSliceE), exec);
}

} // end namespace matx
