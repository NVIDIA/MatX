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

#include "assert.h"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;

template <typename TensorType>
class QR2SolverTestNonHalfTypes : public ::testing::Test{
};

TYPED_TEST_SUITE(QR2SolverTestNonHalfTypes,
  MatXFloatNonHalfTypesCUDAExec);

template <typename TestType, int RANK>
void qr_test( const index_t (&AshapeA)[RANK]) { 
  using AType = TestType;
  using SType = typename inner_op_type_t<AType>::type;
  
  cudaStream_t stream = 0;
  cudaExecutor exec{stream};

  cuda::std::array<index_t, RANK> Ashape = detail::to_array(AshapeA);
  cuda::std::array<index_t, RANK> Qshape = Ashape;
  cuda::std::array<index_t, RANK> Rshape = Ashape;

  index_t m = Ashape[RANK-2];
  index_t n = Ashape[RANK-1];

  Qshape[RANK-2] = m;
  Qshape[RANK-1] = m;

  Rshape[RANK-2] = m;
  Rshape[RANK-1] = n;

  auto A = make_tensor<AType>(Ashape);
  auto Q = make_tensor<AType>(Qshape);
  auto R = make_tensor<AType>(Rshape);
  
  (A = random<float>(Ashape, NORMAL)).run(exec);
  
  // example-begin qr-test-1
  (mtie(Q, R) = qr(A)).run(exec);
  // example-end qr-test-1

  auto mdiffQTQ = make_tensor<SType>({});
  auto mdiffQR = make_tensor<SType>({});

  {
    // QTQ == Identity
    auto QTQ = make_tensor<AType>(Qshape);
    (QTQ = matmul(conj(transpose_matrix(Q)), Q)).run(exec);
    auto e = eye<SType>({m, m});

    auto eShape = Qshape;
    eShape[RANK-1] = matxKeepDim;
    eShape[RANK-2] = matxKeepDim;
    auto I = clone<RANK>(e, eShape);
  
    (mdiffQTQ = max(abs(QTQ-I))).run(exec);

  }

  {
    // Q*R == A
    auto QR = make_tensor<AType>(Ashape);
    (QR = matmul(Q, R)).run(exec);
    
    (mdiffQR = max(abs(A-QR))).run(exec);
  }

  exec.sync();

  ASSERT_NEAR( mdiffQTQ(), SType(0), .00001);
  ASSERT_NEAR( mdiffQR(), SType(0), .00001);
}

TYPED_TEST(QR2SolverTestNonHalfTypes, QR2)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;  
  
  qr_test<TestType>({4,4});
  qr_test<TestType>({4,16});
  qr_test<TestType>({16,4});

  qr_test<TestType>({25,4,4});
  qr_test<TestType>({25,4,16});
  qr_test<TestType>({25,16,4});

  qr_test<TestType>({5,5,4,4});
  qr_test<TestType>({5,5,4,16});
  qr_test<TestType>({5,5,16,4});
  
  MATX_EXIT_HANDLER();
}
