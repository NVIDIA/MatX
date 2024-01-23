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
#include "matx/operators/slice.h"
#include <cstdio>
#include <numeric>

namespace matx {

namespace detail {

  template<typename AType>
    inline auto qr_internal_workspace(const AType &A, cudaStream_t stream) {
      using ATypeS = typename AType::scalar_type;
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
    inline void qr_internal(QType &Q, RType &R, const AType &A, WType workspace, cudaStream_t stream) {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

      static_assert(AType::Rank() >= 2);
      static_assert(QType::Rank() == AType::Rank());
      static_assert(RType::Rank() == AType::Rank());

      MATX_ASSERT_STR(AType::Rank() == QType::Rank(), matxInvalidDim, "qr: A and Q must have the same rank");
      MATX_ASSERT_STR(AType::Rank() == RType::Rank(), matxInvalidDim, "qr: A and R must have the same rank");

      using ATypeS = typename AType::scalar_type;
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
        matmul_impl(Q, Qin, wwt, stream, -2, 1);

      }
    }
} // end namespace detail

/**
 * Perform QR decomposition on a matrix using housholders reflections. If rank > 2 operations are batched.
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
 * @param stream
 *   CUDA stream
 */
template<typename QType, typename RType, typename AType>
inline void qr_impl(QType &Q, RType &R, const AType &A, cudaStream_t stream) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  static_assert(AType::Rank() >= 2);
  static_assert(QType::Rank() == AType::Rank());
  static_assert(RType::Rank() == AType::Rank());

  MATX_ASSERT_STR(AType::Rank() == QType::Rank(), matxInvalidDim, "qr: A and Q must have the same rank");
  MATX_ASSERT_STR(AType::Rank() == RType::Rank(), matxInvalidDim, "qr: A and R must have the same rank");

  auto workspace = qr_internal_workspace(A, stream);
  qr_internal(Q,R,A,workspace,stream);
}

} // end namespace matx
