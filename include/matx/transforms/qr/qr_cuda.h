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
#include "matx/operators/slice.h"
#include "matx/transforms/solver_common.h"

#include <cstdio>
#include <numeric>

namespace matx {

namespace detail {

  template<typename AType>
    inline auto qr_internal_workspace(const AType &A, cudaStream_t stream) {
      using ATypeS = typename AType::value_type;
      const int RANK = AType::Rank();

      index_t m = A.Size(RANK-2);
      index_t n = A.Size(RANK-1);

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
      using NTypeS = typename inner_op_type_t<ATypeS>::type;
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
        (xz = (index(x.Rank()-1) >= i) * x).run(stream);

        // compute L2 norm without sqrt. 
        (N = sum(abs2(xz))).run(stream);
        //(N = sqrt(N)).run(stream);  // sqrt folded into next op

        (v = xz + (index(v.Rank()-1) == i) * sign(xz) * sqrt(nc)).run(stream); 

        auto r = x;  // alias column of R happens to be the same as x

        (s = sum(abs2(v))).run(stream);
        //(s = sqrt(s)).run(stream);  // sqrt folded into next op

        // IFELSE to avoid nans when dividing by zero
        (IFELSE(sc != NTypeS(0), 
                w = (v / sqrt(sc)),
                w = NTypeS(0))).run(stream);

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
  int64_t m;
  int64_t n;
  void *A;
  void *tau;
  size_t batch_size;
  MatXDataType_t dtype;
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
    cusolverStatus_t ret = cusolverDnXgeqrf_bufferSize(
            this->handle, this->dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
            params.A, params.m, MatXTypeToCudaType<T2>(), params.tau,
            MatXTypeToCudaType<T1>(), &this->dspace, &this->hspace);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnQRCUDAParams_t GetQRParams(TauTensor &tau,
                                  const ATensor &a)
  {
    DnQRCUDAParams_t params;

    params.batch_size = GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.tau = tau.Data();
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(OutTensor &out, TauTensor &tau,
            const ATensor &a, const cudaExecutor &exec)
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
    cusolverDnSetStream(this->handle, stream);

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXgeqrf(
          this->handle, this->dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
          this->batch_a_ptrs[i], params.m, MatXTypeToCudaType<T2>(),
          this->batch_tau_ptrs[i], MatXTypeToCudaType<T1>(),
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
      MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
        ("Parameter " + std::to_string(-info) + " had an illegal value in cuSolver Xgeqrf").c_str());
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
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test QR parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnQRCUDAParamsKeyEq {
  bool operator()(const DnQRCUDAParams_t &l, const DnQRCUDAParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size &&
           l.dtype == t.dtype;
  }
};

using qr_cuda_cache_t = std::unordered_map<DnQRCUDAParams_t, std::any, DnQRCUDAParamsKeyHash, DnQRCUDAParamsKeyEq>;

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
 */
template <typename OutTensor, typename TauTensor, typename ATensor>
void qr_solver_impl(OutTensor &&out, TauTensor &&tau,
        const ATensor &a, const cudaExecutor &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using T1 = typename remove_cvref_t<OutTensor>::value_type;

  auto tau_new = OpToTensor(tau, exec);
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
  auto params = detail::matxDnQRCUDAPlan_t<OutTensor, decltype(tau_new), decltype(a_new)>::GetQRParams(tau_new, tvt);

  // Get cache or new QR plan if it doesn't exist
  using cache_val_type = detail::matxDnQRCUDAPlan_t<OutTensor, decltype(tau_new), decltype(a_new)>;
  detail::GetCache().LookupAndExec<detail::qr_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::qr_cuda_cache_t>(),
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
}

} // end namespace matx
