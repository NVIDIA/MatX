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
#include <cuda/std/__algorithm/min.h>

using namespace matx;

template <typename TensorType>
class QREconSolverTestNonHalfTypes : public ::testing::Test{
};

TYPED_TEST_SUITE(QREconSolverTestNonHalfTypes,
  MatXFloatNonHalfTypesCUDAExec);

template <typename T>
T MakeQREconJITTestValue(double value)
{
  using SType = typename inner_op_type_t<T>::type;
  if constexpr (is_complex_v<T>) {
    return T{static_cast<SType>(value), static_cast<SType>(0)};
  }
  else {
    return static_cast<T>(value);
  }
}

#if defined(MATX_EN_MATHDX) && defined(MATX_EN_JIT)
template <typename TensorType>
class QREconSolverJITTestNonHalfTypes : public ::testing::Test {
};

TYPED_TEST_SUITE(QREconSolverJITTestNonHalfTypes,
  MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(QREconSolverJITTestNonHalfTypes, CuSolverDxSingleMatrixProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t rows = 4;
  constexpr index_t cols = 4;
  auto A = make_tensor<TestType>({rows, cols});
  auto Combined = make_tensor<TestType>({rows, cols});
  auto mdiff = make_tensor<value_type>({});

  for (index_t i = 0; i < rows; i++) {
    for (index_t j = 0; j < cols; j++) {
      A(i, j) = i == j ? MakeQREconJITTestValue<TestType>(static_cast<double>(i + 2)) : TestType{};
    }
  }

  auto op = qr_econ(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Q));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.R));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  (Combined = op.Q * op.R).run(exec);
  (mdiff = max(abs(Combined - A))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}

TEST(QREconSolverJITRegression, CuSolverDxMultipleSizesInOneJITExpression)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto A2 = make_tensor<TestType>({2, 2});
  auto A3 = make_tensor<TestType>({3, 3});
  auto Out = make_tensor<TestType>({2, 2});
  auto Ref = make_tensor<TestType>({2, 2});
  auto mdiff = make_tensor<TestType>({});

  A2(0, 0) = TestType{4};
  A2(0, 1) = TestType{1};
  A2(1, 0) = TestType{2};
  A2(1, 1) = TestType{3};

  A3(0, 0) = TestType{7};
  A3(0, 1) = TestType{1};
  A3(0, 2) = TestType{2};
  A3(1, 0) = TestType{3};
  A3(1, 1) = TestType{8};
  A3(1, 2) = TestType{1};
  A3(2, 0) = TestType{2};
  A3(2, 1) = TestType{4};
  A3(2, 2) = TestType{9};

  auto op2 = qr_econ(A2);
  auto op3 = qr_econ(A3);
  auto ref2 = qr_econ(A2);
  auto ref3 = qr_econ(A3);
  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (Out = op2.R + slice<2>(op3.R, {0, 0}, {2, 2})).run(jit_exec);
  (Ref = ref2.R + slice<2>(ref3.R, {0, 0}, {2, 2})).run(cuda_exec);
  (mdiff = max(abs(Out - Ref))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), TestType(0), TestType(0.001));

  MATX_EXIT_HANDLER();
}

TEST(QREconSolverJITRegression, CuSolverDxWideQProjectionRejectsJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto A = make_tensor<TestType>({3, 4});
  auto op = qr_econ(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Q));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QREconSolverJITTestNonHalfTypes, CuSolverDxBatchedMatrixProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t batches = 3;
  constexpr index_t rows = 4;
  constexpr index_t cols = 4;
  auto A = make_tensor<TestType>({batches, rows, cols});
  auto Combined = make_tensor<TestType>({batches, rows, cols});
  auto mdiff = make_tensor<value_type>({});

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < rows; i++) {
      for (index_t j = 0; j < cols; j++) {
        A(b, i, j) = i == j ? MakeQREconJITTestValue<TestType>(static_cast<double>(b + i + 2)) : TestType{};
      }
    }
  }

  auto op = qr_econ(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Q));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.R));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  (Combined = op.Q * op.R).run(exec);
  (mdiff = max(abs(Combined - A))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QREconSolverJITTestNonHalfTypes, CuSolverDxRank4BatchedMatrixProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t batch0 = 2;
  constexpr index_t batch1 = 2;
  constexpr index_t rows = 4;
  constexpr index_t cols = 4;
  auto A = make_tensor<TestType>({batch0, batch1, rows, cols});
  auto Combined = make_tensor<TestType>({batch0, batch1, rows, cols});
  auto mdiff = make_tensor<value_type>({});

  for (index_t b0 = 0; b0 < batch0; b0++) {
    for (index_t b1 = 0; b1 < batch1; b1++) {
      for (index_t i = 0; i < rows; i++) {
        for (index_t j = 0; j < cols; j++) {
          A(b0, b1, i, j) = i == j ?
            MakeQREconJITTestValue<TestType>(static_cast<double>(b0 + b1 + i + 2)) :
            TestType{};
        }
      }
    }
  }

  auto op = qr_econ(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Q));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.R));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  (Combined = op.Q * op.R).run(exec);
  (mdiff = max(abs(Combined - A))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}
#endif

template <typename TestType, int RANK>
void qr_econ_test( const index_t (&AshapeA)[RANK]) { 
  using AType = TestType;
  using SType = typename inner_op_type_t<AType>::type;
  
  cudaStream_t stream = 0;
  cudaExecutor exec{stream};

  cuda::std::array<index_t, RANK> Ashape = detail::to_array(AshapeA);
  cuda::std::array<index_t, RANK> Qshape = Ashape;
  cuda::std::array<index_t, RANK> Rshape = Ashape;

  index_t m = Ashape[RANK-2];
  index_t n = Ashape[RANK-1];
  index_t k = cuda::std::min(m, n);
  Qshape[RANK-2] = m;
  Qshape[RANK-1] = k;

  Rshape[RANK-2] = k;
  Rshape[RANK-1] = n;

  auto A = make_tensor<AType>(Ashape);
  auto Q = make_tensor<AType>(Qshape);
  auto R = make_tensor<AType>(Rshape);
  
  (A = random<AType>(Ashape, NORMAL)).run(exec);
  
  // example-begin qr-econ-test-1
  (mtie(Q, R) = qr_econ(A)).run(exec);
  // example-end qr-econ-test-1

  auto mdiffQTQ = make_tensor<SType>({});
  auto mdiffQR = make_tensor<SType>({});

  {
    // QTQ == Identity
    cuda::std::array<index_t, RANK> QTQshape = Qshape;
    QTQshape[RANK-1] = k;
    QTQshape[RANK-2] = k;

    auto QTQ = make_tensor<AType>(QTQshape);
    (QTQ = matmul(conj(transpose_matrix(Q)), Q)).run(exec);
    auto e = eye<AType>({k, k});

    auto eShape = QTQshape;
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
  ClearCachesAndAllocations();
}

TYPED_TEST(QREconSolverTestNonHalfTypes, QREcon)
{
  MATX_ENTER_HANDLER();
  ClearCachesAndAllocations();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;  
  
  qr_econ_test<TestType>({4,4});
  qr_econ_test<TestType>({4,16});
  
  qr_econ_test<TestType>({16,4});

  qr_econ_test<TestType>({25,4,4});
  qr_econ_test<TestType>({25,4,16});
  qr_econ_test<TestType>({25,16,4});

  qr_econ_test<TestType>({5,5,4,4});
  qr_econ_test<TestType>({5,5,4,16});
  qr_econ_test<TestType>({5,5,16,4});
  
  MATX_EXIT_HANDLER();
}

TYPED_TEST(QREconSolverTestNonHalfTypes, ProjectionAPI)
{
  MATX_ENTER_HANDLER();
  ClearCachesAndAllocations();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using SType = typename inner_op_type_t<TestType>::type;

  cudaExecutor exec{};
  auto A = make_tensor<TestType>({4, 3});
  auto QR = make_tensor<TestType>({4, 3});
  auto mdiff = make_tensor<SType>({});

  (A = random<TestType>(A.Shape(), NORMAL)).run(exec);

  auto op = qr_econ(A);
  (QR = matmul(op.Q, op.R)).run(exec);
  (mdiff = max(abs(A - QR))).run(exec);
  exec.sync();

  ASSERT_NEAR(mdiff(), SType(0), SType(0.0001));
  MATX_EXIT_HANDLER();
}
