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

  // internal qr implementation which takes memory in api call
  template<typename QType, typename RType, typename AType, 
    typename NType, typename VMType, typename HType, typename QNType, typename RNType>
      void qr_internal(QType Q, RType R, AType A,
          NType N, VMType VM, HType H, QNType QN, RNType RN, cudaStream_t stream) {
        MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

        static_assert(A.Rank() >= 2);
        static_assert(Q.Rank() == A.Rank());
        static_assert(R.Rank() == A.Rank());

        MATX_ASSERT_STR(A.Rank() == Q.Rank(), matxInvalidDim, "qr: A and Q must have the same rank");
        MATX_ASSERT_STR(A.Rank() == R.Rank(), matxInvalidDim, "qr: A and R must have the same rank");

        using ATypeS = typename AType::scalar_type;
        using NTypeS = typename inner_op_type_t<ATypeS>::type;
        const int RANK = AType::Rank();

        index_t m = A.Size(RANK-2);
        index_t n = A.Size(RANK-1);
        index_t k = std::min(m,n);
        if(m<=n) k--;  // these matrices have one less update since the diagonal ends on the bottom of the matrix

        // Create Identity matrix
        auto E = eye<ATypeS>({m, m});

        // Clone over batch Dims
        auto ECShape = Q.Shape();
        ECShape[RANK-1] = matxKeepDim;
        ECShape[RANK-2] = matxKeepDim;

        auto I = clone<RANK>(E, ECShape);

        auto ci = N;  // alias

        // Inititalize Q
        (Q = I).run(stream);
        (R = A).run(stream);

        // setup slices
        std::array<index_t, RANK> mSliceB, mSliceE;  // matrix slice starting at i,i
        mSliceB.fill(0); mSliceE.fill(matxEnd);

        std::array<index_t, RANK> qSliceB, qSliceE;  // matrix slice starting at 0,i
        qSliceB.fill(0); qSliceE.fill(matxEnd);

        std::array<index_t, RANK> xSliceB, xSliceE;  // vector slice starting at i,i
        xSliceB.fill(0); xSliceE.fill(matxEnd);
        xSliceE[RANK-1] = matxDropDim; // drop last dim to make a vector

        std::array<index_t, RANK> vSliceB, vSliceE;  // vector slice starting at i,0?
        vSliceB.fill(0); vSliceE.fill(matxEnd);
        vSliceE[RANK-1] = matxDropDim; // drop last dim to make a vector

        std::array<index_t, RANK> vmSliceB, vmSliceE;  // vector slice as a matrix starting at i
        vmSliceB.fill(0); vmSliceE.fill(matxEnd);

        // N cloned across x.
        std::array<index_t, RANK-1> ncShape;
        ncShape.fill(matxKeepDim);

        // clone  ci across hi
        std::array<index_t, RANK> cicShape;
        cicShape.fill(matxKeepDim);

        for(int i = 0 ; i < k ; i++) {

          // update top left corner of slice
          mSliceB[RANK-2] = i;
          mSliceB[RANK-1] = i;

          // update q slice
          qSliceB[RANK-1] = i;

          // update v/vm slices to start at i
          xSliceB[RANK-2] = i;
          xSliceB[RANK-1] = i;
          vmSliceB[RANK-2] = i;
          vSliceB[RANK-2] = i;

          // matrix slices
          auto qi = slice<RANK>(Q, qSliceB, qSliceE);
          auto qn = slice<RANK>(QN, qSliceB, qSliceE);
          auto ri = slice<RANK>(R, mSliceB, mSliceE);
          auto rn = slice<RANK>(RN, mSliceB, mSliceE);
          auto hi = slice<RANK>(H, mSliceB, mSliceE);
          auto vm = slice<RANK>(VM, vmSliceB, vmSliceE);

          // vector slices
          auto v = slice<RANK-1>(VM, vSliceB, vSliceE);
          auto x = slice<RANK-1>(R, xSliceB, xSliceE);

          //update clone shape
          ncShape[RANK-2] = m-i;
          auto nc = clone<RANK-1>(N,ncShape);

          // update cloned ci shape
          cicShape[RANK-2] = m - i;
          cicShape[RANK-1] = m - i;
          auto cic = clone<RANK>(ci, cicShape);

          // create identity matrix and clone across full rank
          auto Ei = eye<ATypeS>({m-i,m-i});
          auto Ii = clone<RANK>(Ei, ECShape);

          (N = sum(norm(x))).run(stream);
          (N = sqrt(N)).run(stream);

          // copy x into v and apply signed addition of nc
          (IFELSE( (index(v.Rank()-1) == 0),
                   v = x + sign(x, ATypeS(1) ) * nc,
                   v = x)).run(stream);

          (ci = sum(norm(v))).run(stream);

          (ci = NTypeS(2) / ci).run(stream);

          matmul_impl(hi, vm, conj(transpose_matrix(vm)), stream);

          (hi = Ii - cic * hi).run(stream);

          // update panel of r
          matmul_impl(rn, hi, ri, stream);

          // update panel of q
          matmul_impl(qn, qi, hi, stream); 

          // deep copy required (can't swap)
          // copy current panels into output matrix
          (ri = rn).run(stream);
          (qi = qn).run(stream);

          // R & Q now contain latest
        }

        // zero out lower triangular part of R
        (IF(index(RANK-1) < index(RANK-2), R = 0)).run(stream);

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
void qr_impl(QType Q, RType R, AType A, cudaStream_t stream) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  static_assert(A.Rank() >= 2);
  static_assert(Q.Rank() == A.Rank());
  static_assert(R.Rank() == A.Rank());

  MATX_ASSERT_STR(A.Rank() == Q.Rank(), matxInvalidDim, "qr: A and Q must have the same rank");
  MATX_ASSERT_STR(A.Rank() == R.Rank(), matxInvalidDim, "qr: A and R must have the same rank");

  using ATypeS = typename AType::scalar_type;
  using NTypeS = typename inner_op_type_t<ATypeS>::type;
  const int RANK = AType::Rank();

  index_t m = A.Size(RANK-2);
  index_t n = A.Size(RANK-1);
  index_t k = std::min(m,n);
  if(m<=n) k--;  // these matrices have one less update since the diagonal ends on the bottom of the matrix

  std::array<index_t, RANK-2> NShape;
  for(int i = 0; i < RANK-2; i++) {
    NShape[i] = A.Size(i);
  }

  std::array<index_t, RANK> VMShape;
  for(int i = 0; i < RANK-1; i++) {
    VMShape[i] = A.Size(i);
  }
  VMShape[RANK-1] = 1;

  std::array<index_t, RANK> HShape;
  for(int i = 0; i < RANK-2; i++) {
    HShape[i] = A.Size(i);
  }
  HShape[RANK-2] = m;
  HShape[RANK-1] = m;

  auto N = make_tensor<NTypeS>(NShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto VM = make_tensor<ATypeS>(VMShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto H = make_tensor<ATypeS>(HShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto QN = make_tensor<ATypeS>(Q.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
  auto RN = make_tensor<ATypeS>(R.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);

  qr_internal(Q,R,A,N,VM,H,QN,RN,stream);
}

} // end namespace matx
