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
#include <cstdio>
#include <numeric>

namespace matx {

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
 * @param stream
 *   CUDA stream
 * @param k
 *    The number of singular values to find.  Default is all singular values: min(m,n).
 */
template<typename UType, typename SType, typename VTType, typename AType, typename X0Type>
void svdpi_impl(UType &U, SType &S, VTType &VT, AType &A, X0Type &x0, int iterations,  cudaStream_t stream, index_t k=-1) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  static_assert(UType::Rank() == AType::Rank());
  static_assert(VTType::Rank() == AType::Rank());
  static_assert(SType::Rank() == AType::Rank()-1);

  using ATypeS = typename AType::scalar_type;
  using STypeS = typename SType::scalar_type;
  const int RANK = AType::Rank();

  auto m = A.Size(RANK-2);  // rows
  auto n = A.Size(RANK-1);  // cols
  auto d = std::min(n,m); // dim for AAT or ATA
  
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
  std::array<index_t, RANK-2> sumsShape;
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

    std::array<index_t, SType::Rank()> sShapeB, sShapeE;

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
        matmul_impl(AT, Ap, conj(transpose_matrix(Ap)), stream);
      } else { // !ufirst
        // use conj transpose for complex
        matmul_impl(AT, conj(transpose_matrix(Ap)), Ap, stream);
      } // end ufirst

      // for each fixed point iteration
      for(int it = 0; it < iterations; it++) {

        matmul_impl(xm, AT, xm, stream);

        // normalize x at each iteration to avoid instability
        // first compute sum of squares, norm will work for complex and real
#if 0
        sum(s, norm(x), stream);
#else
        //WAR cub not supporting strided output
        (sums = sum(norm(x))).run(stream);
        (s = sums).run(stream);
#endif

        const int CRANK = s.Rank()+1;  // adding one more dim to s
        std::array<index_t, CRANK> sCloneShape;
        sCloneShape.fill(matxKeepDim);
        sCloneShape[CRANK-1] = d;  // last dim is cloned d ways

        (x = x / sqrt(clone(s, sCloneShape))).run(stream);
      }
    }

    // slice out current singular vectors and singular value we are working on
    std::array<index_t, RANK> umShapeB, umShapeE;
    std::array<index_t, RANK> vmShapeB, vmShapeE;

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
      matmul_impl(transpose_matrix(vm), conj(transpose_matrix(Ap)), um, stream);    // (n x 1) = (n x m) ( (m x 1)
      // example-end transpose_matrix-test-1

      // compute singular value as L2 norm of v
      // first compute sum of squares, norm will work for complex and real
#if 0
      sum(s, norm(v), stream);
#else
      //WAR cub not supporting strided output
      (sums = sum(norm(v))).run(stream);
      (s = sums).run(stream);;
#endif
      (s = sqrt(s)).run(stream);

      // normalize v

      const int CRANK = s.Rank()+1;  // adding one more dim to s
      std::array<index_t, CRANK> sCloneShape;
      sCloneShape.fill(matxKeepDim);
      sCloneShape[CRANK-1] = n;  // last dim is cloned n ways

      // since v is stored as the transpose we should also store it as the conj
      (v = conj(v / clone(s, sCloneShape))).run(stream);
    } else {  // !ufirst
      // copy singular vector to v
      // store V as the conj transpose
      (v = conj(x)).run(stream);

      // compute u, undo conj
      matmul_impl(um, Ap, conj(transpose_matrix(vm)), stream);    // (m x 1) = (m x n) ( (n x 1)
      // compute singular value as L2 norm of v
      // first compute sum of squares, norm will work for complex and real
#if 0
      sum(s, norm(u), stream);
#else
      //WAR cub not supporting strided output
      (sums = sum(norm(u))).run(stream);
      (s = sums).run(stream);;
#endif
      (s = sqrt(s)).run(stream);
      // normalize u
      const int CRANK = s.Rank()+1;  // adding one more dim to s
      std::array<index_t, CRANK> sCloneShape;
      sCloneShape.fill(matxKeepDim);
      sCloneShape[CRANK-1] = m;  // last dim is cloned m ways

      (u = u / clone(s, sCloneShape)).run(stream);
    } // end ufirst
    
    // Remove current singular vectors from matrix
    if(i < k - 1) {

      // vm is already conj for complex so no need to conj here
      matmul_impl(uv, um, vm, stream);

      const int CRANK = s.Rank()+ 2;  // adding one more dim to s
      std::array<index_t, CRANK> sCloneShape;
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
 * @param iterations
 *   The number of power iterations to perform for each singular value.  
 * @param stream
 *   CUDA stream
 */
template<typename UType, typename SType, typename VTType, typename AType>
void svdbpi_impl(UType &U, SType &S, VTType &VT, AType &A, int iterations,  cudaStream_t stream) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  static_assert(UType::Rank() == AType::Rank());
  static_assert(VTType::Rank() == AType::Rank());
  static_assert(SType::Rank() == AType::Rank()-1);

  using ATypeS = typename AType::scalar_type;
  using STypeS = typename SType::scalar_type;
  const int RANK = AType::Rank();

  auto m = A.Size(RANK-2);  // rows
  auto n = A.Size(RANK-1);  // cols
  auto d = std::min(n,m); // dim for AAT or ATA
  
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
  

  auto ATShape = A.Shape();
  ATShape[RANK-2] = d;
  ATShape[RANK-1] = d;

  auto QShape = A.Shape();
  QShape[RANK-1] = d;
  QShape[RANK-2] = d;
  
  auto RShape = A.Shape();
  RShape[RANK-1] = d;
  RShape[RANK-2] = d;  

  // temp memory for block power iteration
  auto AT = make_tensor<ATypeS>(ATShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto Q = make_tensor<ATypeS>(QShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto R = make_tensor<ATypeS>(RShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto Z = make_tensor<ATypeS>(QShape, MATX_ASYNC_DEVICE_MEMORY, stream);

  std::array<index_t, RANK-2> NShape;
  for(int i = 0; i < RANK-2; i++) {
    NShape[i] = Z.Size(i);
  }

  std::array<index_t, RANK> VMShape;
  for(int i = 0; i < RANK-1; i++) {
    VMShape[i] = Z.Size(i);
  }
  VMShape[RANK-1] = 1;

  std::array<index_t, RANK> HShape;
  for(int i = 0; i < RANK-2; i++) {
    HShape[i] = Z.Size(i);
  }
  HShape[RANK-2] = Z.Size(RANK-2);
  HShape[RANK-1] = Z.Size(RANK-2);

  // temp memory for qr
  auto N = make_tensor<STypeS>(NShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto VM = make_tensor<ATypeS>(VMShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto H = make_tensor<ATypeS>(HShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto QN = make_tensor<ATypeS>(QShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto RN = make_tensor<ATypeS>(RShape, MATX_ASYNC_DEVICE_MEMORY, stream);

  // create spd matrix
  if ( m >= n ) {
    matmul_impl(AT, conj(transpose_matrix(A)), A, stream);
  } else {
    matmul_impl(AT, A, conj(transpose_matrix(A)), stream);
  }
   
  auto e2 = eye({d,d});
  auto cShape = A.Shape();
  cShape[RANK-1] = matxKeepDim;
  cShape[RANK-2] = matxKeepDim;
  
  (Q = clone<RANK>(e2, cShape)).run(stream);

  for(int i = 0; i < iterations; i++) {
    matmul_impl(Z, AT, Q, stream);
    qr_internal(Q,R,Z,N,VM,H,QN,RN,stream);
  }

  (S = real(sqrt(diag(R)))).run(stream);
  
  if( m >= n ) {
    (VT = conj(transpose_matrix(Q))).run(stream);
    matmul_impl(U, A, Q, stream);
    
    auto DShape = U.Shape();
    DShape.fill(matxKeepDim);
    DShape[RANK-2] = m;
    auto D = clone<RANK>(S, DShape);
    
    // normalize U by singular values
    (U = U * STypeS(1) / D).run(stream);
  } else {
    (U = Q).run(stream);
    matmul_impl(VT, conj(transpose_matrix(Q)), A, stream);
    
    auto DShape = VT.Shape();
    DShape.fill(matxKeepDim);
    DShape[RANK-1] = n;
    auto D = clone<RANK>(S, DShape);
    
    // normalize VT by singular values
    (VT = VT * STypeS(1) / D).run(stream);
  }
}

} // end namespace matx

