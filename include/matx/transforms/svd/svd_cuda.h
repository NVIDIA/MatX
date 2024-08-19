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
inline auto svdbpi_impl_workspace(const AType &A, cudaStream_t stream) {
  using ATypeS = typename AType::value_type;
  const int RANK = AType::Rank();

  auto m = A.Size(RANK-2);  // rows
  auto n = A.Size(RANK-1);  // cols
  auto d = cuda::std::min(n,m); // dim for AAT or ATA

  auto ATShape = A.Shape();
  ATShape[RANK-2] = d;
  ATShape[RANK-1] = d;

  auto QShape = A.Shape();
  QShape[RANK-1] = d;
  QShape[RANK-2] = d;

  auto RShape = A.Shape();
  RShape[RANK-1] = d;
  RShape[RANK-2] = d;  

  cuda::std::array<index_t,RANK-2> l2NormShape;
  for(int i=0;i<RANK-2;i++) {
    l2NormShape[i] = A.Size(i);
  }

  // temp memory for block power iteration
  auto AT = make_tensor<ATypeS>(ATShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto Q = make_tensor<ATypeS>(QShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto Qold = make_tensor<ATypeS>(QShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto R = make_tensor<ATypeS>(RShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto Z = make_tensor<ATypeS>(QShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto l2Norm = make_tensor<float>(l2NormShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto converged = make_tensor<int>({}, MATX_ASYNC_DEVICE_MEMORY, stream); 
  return cuda::std::tuple(AT, Q, Qold, R, Z, l2Norm, converged);
}

} // end namespace detail


/**
 * Perform a SVD decomposition using the power iteration.  This version of
 * SVD works well on small n/m with large batch.
 *
 *
 * @tparam UType
 *   Tensor or operator type for output of U singular vectors.
 * @tparam SType
 *   Tensor or operator type for output of S singular values. SType must have Rank one less than AType.
 * @tparam VType
 *   Tensor or operator type for output of VT singular vectors.
 * @tparam AType
 *   Tensor or operator type for output of A input tensors.
 * @tparam X0Type
 *   Tensor or operator type for X0 initial guess in power iteration.
 *
 * @param U
 *   U tensor or operator for left singular vectors output with size "batches by m by k"
 * @param S
 *   S tensor or operator for singular values output with size "batches by k"
 * @param VT
 *   VT tensor or operator for right singular vectors output as VH with size "batches by k by n"
 * @param A
 *   Input tensor or operator for tensor A input with size "batches by m by n"
 * @param x0
 *   Input tensor or operator signaling the initial guess for x0 at each power iteration.  A
 *   Random tensor of size batches x min(n,m) is suggested.
 * @param iterations
 *   The number of power iterations to perform for each singular value.  
 * @param exec
 *   CUDA executor
 * @param k
 *    The number of singular values to find.  Default is all singular values: min(m,n).
 */
template<typename UType, typename SType, typename VTType, typename AType, typename X0Type>
void svdpi_impl(UType &U, SType &S, VTType &VT, AType &A, X0Type &x0, int iterations,  const cudaExecutor &exec, index_t k=-1) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  const auto stream = exec.getStream();

  static_assert(UType::Rank() == AType::Rank());
  static_assert(VTType::Rank() == AType::Rank());
  static_assert(SType::Rank() == AType::Rank()-1);

  using ATypeS = typename AType::value_type;
  using STypeS = typename SType::value_type;
  const int RANK = AType::Rank();

  auto m = A.Size(RANK-2);  // rows
  auto n = A.Size(RANK-1);  // cols
  auto d = cuda::std::min(n,m); // dim for AAT or ATA
  
  // if sentinal found get all singularvalues
  if( k == -1 ) k = (int) d;

  // assert batch sizes are the same
  for(int i = 0 ; i < RANK-2; i++) {
    MATX_ASSERT_STR(U.Size(i) == A.Size(i), matxInvalidDim, "svdpi:  U and A must have the same batch sizes");
    MATX_ASSERT_STR(VT.Size(i) == A.Size(i), matxInvalidDim, "svdpi:  VT and A must have the same batch sizes");
    MATX_ASSERT_STR(S.Size(i) == A.Size(i), matxInvalidDim, "svdpi:  S and A must have the same batch sizes");
  }
  
  MATX_ASSERT_STR(U.Size(RANK-2) == m, matxInvalidDim, "svdpi: U must have Size(RANK-2) == m");
  MATX_ASSERT_STR(U.Size(RANK-1) == k, matxInvalidDim, "svdpi: U must have Size(RANK-1) == k");
  MATX_ASSERT_STR(VT.Size(RANK-2) == k, matxInvalidDim, "svdpi: VT must have Size(RANK-2) == k");
  MATX_ASSERT_STR(VT.Size(RANK-1) == n, matxInvalidDim, "svdpi: VT must have Size(RANK-1) == n");
  MATX_ASSERT_STR(S.Size(RANK-2) == k, matxInvalidDim, "svdpi:  S must have Size(RANK-2) == k");

  MATX_ASSERT_STR(x0.Size(x0.Rank()-1) == d, matxInvalidSize, "svdpi: Initial guess x0 must have the last dimension equal to min(m,n)");

  // compute u or v first.  Depending on n and m one is more efficient than the other
  bool ufirst = (n >= m);

  // Create shapes for tensors below
  auto AShape = A.Shape();

  // AT will be dxd
  auto ATShape = A.Shape();
  ATShape[RANK-1] = d;
  ATShape[RANK-2] = d;

  //XM will be dx1
  auto xmShape = ATShape;
  xmShape[RANK-1] = 1;

  // create x slice parameters
  // begin is zeros, last dim is dropped
  auto xSliceB = A.Shape();
  auto xSliceE = A.Shape();

  xSliceB.fill(0);
  xSliceE.fill(matxEnd);
  xSliceE[RANK-1] = matxDropDim;

  //one per batch dim
  cuda::std::array<index_t, RANK-2> sumsShape;
  for(int i=0;i<RANK-2;i++) {
    sumsShape[i] = AShape[i];
  }

  auto Ap = make_tensor<ATypeS>(AShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto uv = make_tensor<ATypeS>(AShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto AT = make_tensor<ATypeS>(ATShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto xm = make_tensor<ATypeS>(xmShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  
  // we shouldn't need sums but cub doesn't support strided tensors so we cannot write directly at this time.
  auto sums = make_tensor<STypeS>(sumsShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto x = slice<RANK-1>(xm, xSliceB, xSliceE);

  (Ap = A).run(stream);

  // for each singular value
  for(int i = 0; i < k; i++) {

    cuda::std::array<index_t, SType::Rank()> sShapeB, sShapeE;

    sShapeB.fill(0);
    sShapeE.fill(matxEnd);

    sShapeB[RANK-2] = i;
    sShapeE[RANK-2] = matxDropDim;

    // storing singular values/nrms in current value of S
    auto s = slice<RANK-2>(S,sShapeB, sShapeE);

    // power iteration to extract dominant singular vector
    {
      // initialize x randomly
      (x = x0).run(stream);

      if ( ufirst ) {
        // compute A*AT
        // use conj transpose for complex
        matmul_impl(AT, Ap, conj(transpose_matrix(Ap)), exec);
      } else { // !ufirst
        // use conj transpose for complex
        matmul_impl(AT, conj(transpose_matrix(Ap)), Ap, exec);
      } // end ufirst

      // for each fixed point iteration
      for(int it = 0; it < iterations; it++) {

        matmul_impl(xm, AT, xm, exec);

        // normalize x at each iteration to avoid instability
        // first compute sum of squares, norm will work for complex and real
#if 0
        sum(s, abs2(x), stream);
#else
        //WAR cub not supporting strided output
        (sums = sum(abs2(x))).run(stream);
        (s = sums).run(stream);
#endif

        const int CRANK = s.Rank()+1;  // adding one more dim to s
        cuda::std::array<index_t, CRANK> sCloneShape;
        sCloneShape.fill(matxKeepDim);
        sCloneShape[CRANK-1] = d;  // last dim is cloned d ways

        (x = x / sqrt(clone(s, sCloneShape))).run(stream);
      }
    }

    // slice out current singular vectors and singular value we are working on
    cuda::std::array<index_t, RANK> umShapeB, umShapeE;
    cuda::std::array<index_t, RANK> vmShapeB, vmShapeE;

    umShapeB.fill(0);
    vmShapeB.fill(0);

    umShapeE.fill(matxEnd);
    vmShapeE.fill(matxEnd);

    // making this dim only 1 element in size
    umShapeB[RANK-1] = i;
    umShapeE[RANK-1] = i+1;

    // making this dim only 1 element in size
    vmShapeB[RANK-2] = i;
    vmShapeE[RANK-2] = i+1;

    auto um = slice<RANK>(U, umShapeB, umShapeE);         // as matrix for matmul_impl
    auto vm = slice<RANK>(VT, vmShapeB, vmShapeE);         // as matrix for matmul_impl

    // u/v will drop the dim that is one element a vector
    umShapeE[RANK-1] = matxDropDim;
    vmShapeE[RANK-2] = matxDropDim;

    auto u = slice<RANK-1>(U, umShapeB, umShapeE);  // as vector
    auto v = slice<RANK-1>(VT,vmShapeB, vmShapeE);  // as vector

    if( ufirst ) {
      // copy singular vector to u
      (u = x).run(stream);

      // compute v
      // for complex we need the conj transpose of Ap
      // example-begin transpose_matrix-test-1
      matmul_impl(transpose_matrix(vm), conj(transpose_matrix(Ap)), um, exec);    // (n x 1) = (n x m) ( (m x 1)
      // example-end transpose_matrix-test-1

      // compute singular value as L2 norm of v
      // first compute sum of squares, norm will work for complex and real
#if 0
      sum(s, abs2(v), stream);
#else
      //WAR cub not supporting strided output
      (sums = sum(abs2(v))).run(stream);
      (s = sums).run(stream);;
#endif
      (s = sqrt(s)).run(stream);

      // normalize v

      const int CRANK = s.Rank()+1;  // adding one more dim to s
      cuda::std::array<index_t, CRANK> sCloneShape;
      sCloneShape.fill(matxKeepDim);
      sCloneShape[CRANK-1] = n;  // last dim is cloned n ways

      // since v is stored as the transpose we should also store it as the conj
      (v = conj(v / clone(s, sCloneShape))).run(stream);
    } else {  // !ufirst
      // copy singular vector to v
      // store V as the conj transpose
      (v = conj(x)).run(stream);

      // compute u, undo conj
      matmul_impl(um, Ap, conj(transpose_matrix(vm)), exec);    // (m x 1) = (m x n) ( (n x 1)
      // compute singular value as L2 norm of v
      // first compute sum of squares, norm will work for complex and real
#if 0
      sum(s, abs2(u), stream);
#else
      //WAR cub not supporting strided output
      (sums = sum(abs2(u))).run(stream);
      (s = sums).run(stream);;
#endif
      (s = sqrt(s)).run(stream);
      // normalize u
      const int CRANK = s.Rank()+1;  // adding one more dim to s
      cuda::std::array<index_t, CRANK> sCloneShape;
      sCloneShape.fill(matxKeepDim);
      sCloneShape[CRANK-1] = m;  // last dim is cloned m ways

      (u = u / clone(s, sCloneShape)).run(stream);
    } // end ufirst
    
    // Remove current singular vectors from matrix
    if(i < k - 1) {

      // vm is already conj for complex so no need to conj here
      matmul_impl(uv, um, vm, exec);

      const int CRANK = s.Rank()+ 2;  // adding one more dim to s
      cuda::std::array<index_t, CRANK> sCloneShape;
      sCloneShape.fill(matxKeepDim);
      sCloneShape[CRANK-2] = m;  // second to last dim is cloned m ways
      sCloneShape[CRANK-1] = n;  // last dim is cloned n ways

      (Ap = Ap - clone(s, sCloneShape)  * uv).run(stream);
    }
  }
}

/**
 * Perform a SVD decomposition using the block power iteration.  This version of
 * SVD works well on small n/m with large batch.
 *
 *
 * @tparam UType
 *   Tensor or operator type for output of U singular vectors.
 * @tparam SType
 *   Tensor or operator type for output of S singular values. SType must have Rank one less than AType.
 * @tparam VType
 *   Tensor or operator type for output of VT singular vectors.
 * @tparam AType
 *   Tensor or operator type for output of A input tensors.
 * @tparam Q0Type
 *   Tensor or operator type for X0 initial guess in power iteration.
 *
 * @param U
 *   U tensor or operator for left singular vectors output with size "batches by m by n"
 * @param S
 *   S tensor or operator for singular values output with size "batches by min(m,n)"
 * @param VT
 *   VT tensor or operator for right singular vectors output as VH with size "batches by min(m,n) by n"
 * @param A
 *   Input tensor or operator for tensor A input with size "batches by m by n"
 * @param max_iters
 *   The approximate maximum number of QR iterations to perform. 
 * @param tol
 *   The termination tolerance for the QR iteration. Setting this to 0 will skip the tolerance check.
 * @param exec
 *   CUDA executor
 */
template<typename UType, typename SType, typename VTType, typename AType>
inline void svdbpi_impl(UType &U, SType &S, VTType &VT, const AType &A, int max_iters, float tol,  const cudaExecutor &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL);
  const auto stream = exec.getStream();

  static_assert(UType::Rank() == AType::Rank());
  static_assert(VTType::Rank() == AType::Rank());
  static_assert(SType::Rank() == AType::Rank()-1);

  using STypeS = typename SType::value_type;
  const int RANK = AType::Rank();

  auto m = A.Size(RANK-2);  // rows
  auto n = A.Size(RANK-1);  // cols
  auto d = cuda::std::min(n,m); // dim for AAT or ATA

  // assert batch sizes are the same
  for(int i = 0 ; i < RANK-2; i++) {
    MATX_ASSERT_STR(U.Size(i) == A.Size(i), matxInvalidDim, "svdbpi:  U and A must have the same batch sizes");
    MATX_ASSERT_STR(VT.Size(i) == A.Size(i), matxInvalidDim, "svdbpi:  VT and A must have the same batch sizes");
    MATX_ASSERT_STR(S.Size(i) == A.Size(i), matxInvalidDim, "svdbpi:  S and A must have the same batch sizes");
  }

  MATX_ASSERT_STR(U.Size(RANK-2) == m, matxInvalidDim, "svdbpi: U must have Size(RANK-2) == m");
  MATX_ASSERT_STR(U.Size(RANK-1) == d, matxInvalidDim, "svdbpi: U must have Size(RANK-1) == d");
  MATX_ASSERT_STR(VT.Size(RANK-2) == d, matxInvalidDim, "svdbpi: VT must have Size(RANK-2) == d");
  MATX_ASSERT_STR(VT.Size(RANK-1) == n, matxInvalidDim, "svdbpi: VT must have Size(RANK-1) == n");
  MATX_ASSERT_STR(S.Size(RANK-2) == d, matxInvalidDim, "svdbpi:  S must have Size(RANK-2) == d");

  int converged_host = false;
  cudaStream_t d2h;
  cudaEvent_t event;
  if(tol>0.0f) {
    cudaStreamCreateWithFlags(&d2h,cudaStreamNonBlocking);
    cudaEventCreate(&event);
  }

  auto [AT, Q, Qold, R, Z, l2Norm, converged] = detail::svdbpi_impl_workspace(A, stream);
  auto qr_workspace = detail::qr_internal_workspace(Z, stream);

  // create spd matrix
  if ( m >= n ) {
    matmul_impl(AT, conj(transpose_matrix(A)), A, exec);
  } else {
    matmul_impl(AT, A, conj(transpose_matrix(A)), exec);
  }

  auto e2 = eye({d,d});
  auto cShape = A.Shape();
  cShape[RANK-1] = matxKeepDim;
  cShape[RANK-2] = matxKeepDim;

  (Qold = Q = clone<RANK>(e2, cShape)).run(stream);

  // TODO multistream?
  for(int i = 0; i < max_iters; i+=2)
  {

    // double pump this iteration so we get Qold and Q for tolerance checking.
    // We might take an extra iteration but it will overheads associated with checking concergence.
    matmul_impl(Z, AT, Q, exec);
    detail::qr_internal(Qold, R, Z, qr_workspace, exec);

    matmul_impl(Z, AT, Qold, exec);
    detail::qr_internal(Q, R, Z, qr_workspace, exec);

    if(tol!=0.0f) {

      cudaStreamSynchronize(d2h);  // wait for d2h transfer to finish
      if(converged_host == true) {
        // if converged exit loop
        break;
      }

      //compute L2(Q-Qold)
      // sqrt folded into next operation
      (l2Norm = sum(abs2(Q-Qold))).run(stream);  

      // compute if all batches have converged
      if constexpr (RANK > 2) {
        (converged = all(as_int(sqrt(l2Norm) < tol))).run(stream);
      } else {
        (converged = as_int(sqrt(l2Norm) < tol)).run(stream);
      }
      
      // event to record when converged is ready in stream
      cudaEventRecord(event, stream);
      // wait for d2h transfer until converged is ready
      cudaStreamWaitEvent(d2h, event);

      // copy convergence criteria to host.  
      // This is in unpinned memory and cannot on most systems run asynchronously.  
      // We do this here to hide the copy/sync behind prior launch latency/execution of next iteration.
      cudaMemcpyAsync(&converged_host, converged.Data(), sizeof(int), cudaMemcpyDeviceToHost, d2h);
    }
  }

  (S = real(sqrt(diag(R)))).run(stream);

  if( m >= n ) {
    (VT = conj(transpose_matrix(Q))).run(stream);
    matmul_impl(U, A, Q, exec);

    auto DShape = U.Shape();
    DShape.fill(matxKeepDim);
    DShape[RANK-2] = m;
    auto D = clone<RANK>(S, DShape);

    // normalize U by singular values 
    // IF required to avoid nans when singular value is 0
    (IF(D != STypeS(0), U = U / D)).run(stream);

  } else {
    (U = Q).run(stream);
    matmul_impl(VT, conj(transpose_matrix(Q)), A, exec);

    auto DShape = VT.Shape();
    DShape.fill(matxKeepDim);
    DShape[RANK-1] = n;
    auto D = clone<RANK>(S, DShape);

    // normalize VT by singular values
    // IF required to avoid nans when singular value is 0
    (IF(D != STypeS(0), VT = VT / D)).run(stream);
  }

  if(tol>0.0f) {
    cudaEventDestroy(event);
    cudaStreamDestroy(d2h);
  }
}


/********************************************** SOLVER SVD
 * *********************************************/

namespace detail {

/**
 * Parameters needed to execute singular value decomposition. We distinguish
 * unique factorizations mostly by the data pointer in A.
 */
struct DnSVDCUDAParams_t {
  int64_t m;
  int64_t n;
  char jobz;
  void *A;
  void *U;
  void *VT;
  void *S;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename UTensor, typename STensor, typename VtTensor, typename ATensor>
class matxDnSVDCUDAPlan_t : matxDnCUDASolver_t {
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
   * Creates a handle for decomposing matrix A into the format above. cuSolver destroys
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
   *
   */
  matxDnSVDCUDAPlan_t(UTensor &u,
                        STensor &s,
                        VtTensor &vt,
                        const ATensor &a,
                        const char jobz = 'A')
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

    params = GetSVDParams(u, s, vt, a, jobz);
    this->GetWorkspaceSize();
    this->AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize() override
  {
    // Use all mode for a larger workspace size that works for all modes
    cusolverStatus_t ret =
        cusolverDnXgesvd_bufferSize(
            this->handle, this->dn_params, 'A', 'A', params.m, params.n,
            MatXTypeToCudaType<T1>(), params.A, params.m,
            MatXTypeToCudaType<T3>(), params.S, MatXTypeToCudaType<T2>(),
            params.U, params.m, MatXTypeToCudaType<T4>(), params.VT, params.n,
            MatXTypeToCudaType<T1>(), &this->dspace, &this->hspace);

    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnSVDCUDAParams_t
  GetSVDParams(UTensor &u, STensor &s,
               VtTensor &vt, const ATensor &a,
               const char jobz = 'A')
  {
    DnSVDCUDAParams_t params;
    params.batch_size = GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.U = u.Data();
    params.VT = vt.Data();
    params.S = s.Data();
    params.jobz = jobz;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(UTensor &u, STensor &s, VtTensor &vt,
            const ATensor &a, const cudaExecutor &exec,
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

    const auto stream = exec.getStream();
    cusolverDnSetStream(this->handle, stream);

    // At this time cuSolver does not have a batched 64-bit SVD interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < this->batch_a_ptrs.size(); i++) {

      auto ret = cusolverDnXgesvd(
          this->handle, this->dn_params, jobz, jobz, params.m, params.n,
          MatXTypeToCudaType<T1>(), this->batch_a_ptrs[i], params.m,
          MatXTypeToCudaType<T3>(), this->batch_s_ptrs[i], MatXTypeToCudaType<T2>(),
          this->batch_u_ptrs[i], params.m, MatXTypeToCudaType<T4>(), this->batch_vt_ptrs[i],
          params.n, MatXTypeToCudaType<T1>(),
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
          ("Parameter " + std::to_string(-info) + " had an illegal value in cuSolver Xgesvd").c_str());
      } else {
        MATX_ASSERT_STR_EXP(info, 0, matxSolverError, 
          (std::to_string(info) + " superdiagonals of an intermediate bidiagonal form did not converge to zero in cuSolver Xgesvd").c_str());
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
  ~matxDnSVDCUDAPlan_t() {}

private:
  std::vector<T2 *> batch_u_ptrs;
  std::vector<T3 *> batch_s_ptrs;
  std::vector<T4 *> batch_vt_ptrs;
  DnSVDCUDAParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnSVDCUDAParamsKeyHash {
  std::size_t operator()(const DnSVDCUDAParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test SVD parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnSVDCUDAParamsKeyEq {
  bool operator()(const DnSVDCUDAParams_t &l, const DnSVDCUDAParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

using svd_cuda_cache_t = std::unordered_map<DnSVDCUDAParams_t, std::any, DnSVDCUDAParamsKeyHash, DnSVDCUDAParamsKeyEq>;

}

/**
 * Perform a SVD decomposition using a cached plan
 *
 * See documentation of matxDnSVDCUDAPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
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
 *   CUDA Executor
 * @param jobz
 *   Specifies options for computing all, part, or none of the matrices U and VT. See
 * SVDMode documentation for more info
 *
 */
template <typename UTensor, typename STensor, typename VtTensor, typename ATensor>
void svd_impl(UTensor &&u, STensor &&s,
         VtTensor &&vt, const ATensor &a,
         const cudaExecutor &exec, const SVDMode jobz = SVDMode::ALL)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using T1 = typename ATensor::value_type;
  constexpr int RANK = ATensor::Rank();
  const auto stream = exec.getStream();

  auto s_new = getSolverSupportedTensor(s, exec);

  /* Temporary WAR
     cuSolver assumes column-major matrices and MatX uses row-major matrices.
     One way to address this is to create a transposed copy of the input to
     use with the factorization, followed by transposing the outputs. For SVD,
     we can skip this by passing in a permuted view of row-major A, which is
     equivalent to col-major AT, and swapping the inputs U and VT.

     However, cuSolver only supports m>=n. Thus, this optimization can be used
     for m<=n, but for m>n, we must do temp allocations for U/VT and tranpose the
     col-major cuSolver outputs.

     Eventually these limitations may be fixed in cuSolver.
  */
 
  // cuSolver destroys the input, so we need to make a copy of A regardless
  T1 *tp;
  auto a_shape = a.Shape();
  auto a_total_size = std::accumulate(a_shape.begin(), a_shape.begin() + ATensor::Rank(), 1, std::multiplies< typename ATensor::desc_type::shape_type>());
  matxAlloc(reinterpret_cast<void **>(&tp), sizeof(T1) * a_total_size, MATX_ASYNC_DEVICE_MEMORY, stream);

  const char job_cusolver = detail::SVDModeToChar(jobz);
  const bool m_leq_n = a.Size(RANK-2) <= a.Size(RANK-1);
  
  if (m_leq_n) {
    // get col-major AT
    auto a_new = make_tensor(tp, a_shape);
    (a_new = a).run(exec);
    auto at_col_maj = transpose_matrix(a_new);

    auto u_new = getSolverSupportedTensor(u, exec);
    auto vt_new = getSolverSupportedTensor(vt, exec);

    // swap U and VT
    auto u_in = vt_new;
    auto vt_in = u_new;

    // Get parameters required by these tensors
    auto params = detail::matxDnSVDCUDAPlan_t<decltype(u_in), decltype(s_new), decltype(vt_in), decltype(at_col_maj)>::
      GetSVDParams(u_in, s_new, vt_in, at_col_maj, job_cusolver);

    // Get cache or new SVD plan if it doesn't exist
    using cache_val_type = detail::matxDnSVDCUDAPlan_t<decltype(u_in), decltype(s_new), decltype(vt_in), decltype(at_col_maj)>;
    detail::GetCache().LookupAndExec<detail::svd_cuda_cache_t>(
      detail::GetCacheIdFromType<detail::svd_cuda_cache_t>(),
      params,
      [&]() {
        return std::make_shared<cache_val_type>(u_in, s_new, vt_in, at_col_maj, job_cusolver);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->Exec(u_in, s_new, vt_in, at_col_maj, exec, job_cusolver);
      }
    );

    if(!u_new.isSameView(u)) {
      (u = u_new).run(exec);
    }
    if(!vt_new.isSameView(vt)) {
      (vt = vt_new).run(exec);
    }
  } else {
    // get col-major A
    auto tv = TransposeCopy(tp, a, exec);
    auto tvt = tv.PermuteMatrix();

    auto u_col_maj = make_tensor<T1>(u.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
    auto vt_col_maj = make_tensor<T1>(vt.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);

    // Get parameters required by these tensors
    auto params = detail::matxDnSVDCUDAPlan_t<decltype(u_col_maj), decltype(s_new), decltype(vt_col_maj), decltype(tvt)>::
        GetSVDParams(u_col_maj, s_new, vt_col_maj, tvt, job_cusolver);

    // Get cache or new SVD plan if it doesn't exist
    using cache_val_type = detail::matxDnSVDCUDAPlan_t<decltype(u_col_maj), decltype(s_new), decltype(vt_col_maj), decltype(tvt)>;
    detail::GetCache().LookupAndExec<detail::svd_cuda_cache_t>(
      detail::GetCacheIdFromType<detail::svd_cuda_cache_t>(),
      params,
      [&]() {
        return std::make_shared<cache_val_type>(u_col_maj, s_new, vt_col_maj, tvt, job_cusolver);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->Exec(u_col_maj, s_new, vt_col_maj, tvt, exec, job_cusolver);
      }
    );

    // cuSolver writes u and vt in col-major format, so we need to transpose them back.
    (u = transpose_matrix(u_col_maj)).run(exec);
    (vt = transpose_matrix(vt_col_maj)).run(exec);
  }

  if(!s_new.isSameView(s)) {
    (s = s_new).run(exec);
  }

  matxFree(tp);
}

} // end namespace matx