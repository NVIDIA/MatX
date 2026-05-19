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
constexpr int m = 100;
constexpr int n = 50;

template <typename T> class QRSolverTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;   
   
  void SetUp() override
  {
    if constexpr (!detail::CheckSolverSupport<GExecType>()) {
      GTEST_SKIP();
    }

    // Use an arbitrary number of threads for the select threads host exec.
    if constexpr (is_select_threads_host_executor_v<GExecType>) {
      HostExecParams params{4};
      exec = SelectThreadsHostExecutor{params};
    }

    pb = std::make_unique<detail::MatXPybind>();
  }

  void TearDown() override { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  GExecType exec{};
  float thresh = 0.001f;
};

template <typename TensorType>
class QRSolverTestFloatTypes : public QRSolverTest<TensorType> {
};

TYPED_TEST_SUITE(QRSolverTestFloatTypes,
                 MatXFloatNonHalfTypesAllExecs);

template <typename T>
T MakeQRJITTestValue(double value)
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
class QRSolverJITTestFloatTypes : public ::testing::Test {
};

TYPED_TEST_SUITE(QRSolverJITTestFloatTypes,
                 MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxSingleMatrixQRProjectionJIT)
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
      A(i, j) = i == j ? MakeQRJITTestValue<TestType>(static_cast<double>(i + 2)) : TestType{};
    }
  }

  auto op = qr(A);

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

TEST(QRSolverJITRegression, CuSolverDxQRNonDiagonalHouseholderJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  constexpr index_t rows = 4;
  constexpr index_t cols = 4;
  auto A = make_tensor<TestType>({rows, cols});
  auto Q = make_tensor<TestType>({rows, cols});
  auto R = make_tensor<TestType>({rows, cols});
  auto Combined = make_tensor<TestType>({rows, cols});
  auto mdiff = make_tensor<TestType>({});

  A(0, 0) = TestType{4};
  A(0, 1) = TestType{1};
  A(0, 2) = TestType{2};
  A(0, 3) = TestType{0.5f};
  A(1, 0) = TestType{2};
  A(1, 1) = TestType{5};
  A(1, 2) = TestType{1};
  A(1, 3) = TestType{1};
  A(2, 0) = TestType{1};
  A(2, 1) = TestType{3};
  A(2, 2) = TestType{6};
  A(2, 3) = TestType{2};
  A(3, 0) = TestType{0.5f};
  A(3, 1) = TestType{1};
  A(3, 2) = TestType{2};
  A(3, 3) = TestType{7};

  auto op = qr(A);
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Q));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.R));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor jit_exec{};
  (Q = op.Q).run(jit_exec);
  (R = op.R).run(jit_exec);
  (Combined = matmul(Q, R)).run(cuda_exec);
  (mdiff = max(abs(Combined - A))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), TestType(0), TestType(0.001));

  MATX_EXIT_HANDLER();
}

TEST(QRSolverJITRegression, CuSolverDxQRMultipleSizesInOneJITExpression)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto A2 = make_tensor<TestType>({2, 2});
  auto A3 = make_tensor<TestType>({3, 3});
  auto Out = make_tensor<TestType>({2, 2});
  auto Ref = make_tensor<TestType>({2, 2});
  auto mdiff = make_tensor<TestType>({});

  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 2; j++) {
      A2(i, j) = i == j ? TestType(i + 2) : TestType{};
    }
  }
  for (index_t i = 0; i < 3; i++) {
    for (index_t j = 0; j < 3; j++) {
      A3(i, j) = i == j ? TestType(i + 5) : TestType{};
    }
  }

  auto op2 = qr(A2);
  auto op3 = qr(A3);
  auto ref2 = qr(A2);
  auto ref3 = qr(A3);
  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (Out = op2.R + slice<2>(op3.R, {0, 0}, {2, 2})).run(jit_exec);
  (Ref = ref2.R + slice<2>(ref3.R, {0, 0}, {2, 2})).run(cuda_exec);
  (mdiff = max(abs(Out - Ref))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), TestType(0), TestType(0.001));

  MATX_EXIT_HANDLER();
}

TEST(QRSolverJITRegression, CuSolverDxRectangularRProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  constexpr index_t rows = 4;
  constexpr index_t cols = 3;
  auto A = make_tensor<TestType>({rows, cols});
  auto R = make_tensor<TestType>({rows, cols});
  auto RefR = make_tensor<TestType>({rows, cols});
  auto mdiff = make_tensor<TestType>({});

  A(0, 0) = TestType{4};
  A(0, 1) = TestType{1};
  A(0, 2) = TestType{2};
  A(1, 0) = TestType{2};
  A(1, 1) = TestType{5};
  A(1, 2) = TestType{1};
  A(2, 0) = TestType{1};
  A(2, 1) = TestType{3};
  A(2, 2) = TestType{6};
  A(3, 0) = TestType{0.5f};
  A(3, 1) = TestType{1};
  A(3, 2) = TestType{2};

  auto op = qr(A);
  auto ref = qr(A);
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Q));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.R));

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (R = op.R).run(jit_exec);
  (RefR = ref.R).run(cuda_exec);
  (mdiff = max(abs(R - RefR))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), TestType(0), TestType(0.001));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxBatchedMatrixQRProjectionJIT)
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
        A(b, i, j) = i == j ? MakeQRJITTestValue<TestType>(static_cast<double>(b + i + 2)) : TestType{};
      }
    }
  }

  auto op = qr(A);

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

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxRank4BatchedMatrixQRProjectionJIT)
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
            MakeQRJITTestValue<TestType>(static_cast<double>(b0 + b1 + i + 2)) :
            TestType{};
        }
      }
    }
  }

  auto op = qr(A);

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

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxSingleMatrixQRSolverProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t rows = 4;
  constexpr index_t cols = 3;
  auto A = make_tensor<TestType>({rows, cols});
  auto RefOut = make_tensor<TestType>({rows, cols});
  auto RefTau = make_tensor<TestType>({std::min(rows, cols)});
  auto RefCombined = make_tensor<TestType>({rows, cols});
  auto Combined = make_tensor<TestType>({rows, cols});
  auto mdiff = make_tensor<value_type>({});

  detail::MatXPybind pb;
  pb.template InitAndRunTVGenerator<TestType>("00_solver", "qr", "run", {rows, cols});
  pb.NumpyToTensorView(A, "A");

  auto op = qr_solver(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Out));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Tau));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  (mtie(RefOut, RefTau) = qr_solver(A)).run(cuda_exec);
  (RefCombined = RefOut + clone<2>(RefTau, {rows, matxKeepDim})).run(cuda_exec);
  (Combined = op.Out + clone<2>(op.Tau, {rows, matxKeepDim})).run(exec);
  (mdiff = max(abs(Combined - RefCombined))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}

TEST(QRSolverJITRegression, CuSolverDxQRSolverMultipleSizesInOneJITExpression)
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

  auto op2 = qr_solver(A2);
  auto op3 = qr_solver(A3);
  auto ref2 = qr_solver(A2);
  auto ref3 = qr_solver(A3);
  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (Out = op2.Out + slice<2>(op3.Out, {0, 0}, {2, 2})).run(jit_exec);
  (Ref = ref2.Out + slice<2>(ref3.Out, {0, 0}, {2, 2})).run(cuda_exec);
  (mdiff = max(abs(Out - Ref))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), TestType(0), TestType(0.001));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxBatchedMatrixQRSolverProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t batches = 3;
  constexpr index_t rows = 4;
  constexpr index_t cols = 3;
  auto A = make_tensor<TestType>({batches, rows, cols});
  auto RefOut = make_tensor<TestType>({batches, rows, cols});
  auto RefTau = make_tensor<TestType>({batches, std::min(rows, cols)});
  auto RefCombined = make_tensor<TestType>({batches, rows, cols});
  auto Combined = make_tensor<TestType>({batches, rows, cols});
  auto mdiff = make_tensor<value_type>({});

  detail::MatXPybind pb;
  pb.template InitAndRunTVGenerator<TestType>("00_solver", "qr", "run", {batches, rows, cols});
  pb.NumpyToTensorView(A, "A");

  auto op = qr_solver(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Out));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Tau));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  (mtie(RefOut, RefTau) = qr_solver(A)).run(cuda_exec);
  (RefCombined = RefOut + clone<3>(RefTau, {matxKeepDim, rows, matxKeepDim})).run(cuda_exec);
  (Combined = op.Out + clone<3>(op.Tau, {matxKeepDim, rows, matxKeepDim})).run(exec);
  (mdiff = max(abs(Combined - RefCombined))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxRank4BatchedMatrixQRSolverProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t batch0 = 2;
  constexpr index_t batch1 = 2;
  constexpr index_t rows = 4;
  constexpr index_t cols = 3;
  auto A = make_tensor<TestType>({batch0, batch1, rows, cols});
  auto RefOut = make_tensor<TestType>({batch0, batch1, rows, cols});
  auto RefTau = make_tensor<TestType>({batch0, batch1, std::min(rows, cols)});
  auto RefCombined = make_tensor<TestType>({batch0, batch1, rows, cols});
  auto Combined = make_tensor<TestType>({batch0, batch1, rows, cols});
  auto mdiff = make_tensor<value_type>({});

  for (index_t b0 = 0; b0 < batch0; b0++) {
    for (index_t b1 = 0; b1 < batch1; b1++) {
      for (index_t i = 0; i < rows; i++) {
        for (index_t j = 0; j < cols; j++) {
          const double diag = i == j ? static_cast<double>(b0 + b1 + i + 4) : 0.0;
          const double off_diag = 0.125 * static_cast<double>((i + 1) * (j + 1));
          A(b0, b1, i, j) = MakeQRJITTestValue<TestType>(diag + off_diag);
        }
      }
    }
  }

  auto op = qr_solver(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Out));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Tau));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  (mtie(RefOut, RefTau) = qr_solver(A)).run(cuda_exec);
  (RefCombined = RefOut + clone<4>(RefTau, {matxKeepDim, matxKeepDim, rows, matxKeepDim})).run(cuda_exec);
  (Combined = op.Out + clone<4>(op.Tau, {matxKeepDim, matxKeepDim, rows, matxKeepDim})).run(exec);
  (mdiff = max(abs(Combined - RefCombined))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}
#endif

TYPED_TEST(QRSolverTestFloatTypes, QRBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto Av = make_tensor<TestType>({m, n});
  auto TauV = make_tensor<TestType>({std::min(m,n)});
  auto Qv = make_tensor<TestType>({m, std::min(m, n)});
  auto Rv = make_tensor<TestType>({std::min(m, n), n});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "qr", "run", {m, n});
  this->pb->NumpyToTensorView(Av, "A");
  this->pb->NumpyToTensorView(Qv, "Q");
  this->pb->NumpyToTensorView(Rv, "R");

  // example-begin qr_solver-test-1
  (mtie(Av, TauV) = qr_solver(Av)).run(this->exec);
  // example-end qr_solver-test-1
  this->exec.sync();

  // For now we're only verifying R. Q is a bit more complex to compute since
  // cuSolver/BLAS don't return Q, and instead return Householder reflections
  // that are used to compute Q. Eventually compute Q here and verify
  for (index_t i = 0; i < Av.Size(0); i++) {
    for (index_t j = 0; j < Av.Size(1); j++) {
      // R is stored only in the top triangle of A
      if (i <= j) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Av(i, j).real(), Rv(i, j).real(), this->thresh);
          ASSERT_NEAR(Av(i, j).imag(), Rv(i, j).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Av(i, j), Rv(i, j), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QRSolverTestFloatTypes, QRBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr int batches = 10;
  auto Av = make_tensor<TestType>({batches, m, n});
  auto TauV = make_tensor<TestType>({batches, std::min(m,n)});
  auto Qv = make_tensor<TestType>({batches, m, std::min(m, n)});
  auto Rv = make_tensor<TestType>({batches, std::min(m, n), n});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "qr", "run", {batches, m, n});
  this->pb->NumpyToTensorView(Av, "A");
  this->pb->NumpyToTensorView(Qv, "Q");
  this->pb->NumpyToTensorView(Rv, "R");

  (mtie(Av, TauV) = qr_solver(Av)).run(this->exec);
  this->exec.sync();

  // For now we're only verifying R. Q is a bit more complex to compute since
  // cuSolver/BLAS don't return Q, and instead return Householder reflections
  // that are used to compute Q. Eventually compute Q here and verify
  for (index_t b = 0; b < Av.Size(0); b++) {
    for (index_t i = 0; i < Av.Size(1); i++) {
      for (index_t j = 0; j < Av.Size(2); j++) {
        // R is stored only in the top triangle of A
        if (i <= j) {
          if constexpr (is_complex_v<TestType>) {
            ASSERT_NEAR(Av(b, i, j).real(), Rv(b, i, j).real(), this->thresh);
            ASSERT_NEAR(Av(b, i, j).imag(), Rv(b, i, j).imag(), this->thresh);
          }
          else {
            ASSERT_NEAR(Av(b, i, j), Rv(b, i, j), this->thresh);
          }
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QRSolverTestFloatTypes, ProjectionAPI)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t rows = 8;
  constexpr index_t cols = 5;
  using value_type = typename inner_op_type_t<TestType>::type;
  auto A = make_tensor<TestType>({rows, cols});
  auto RefOut = make_tensor<TestType>({rows, cols});
  auto RefTau = make_tensor<TestType>({std::min(rows, cols)});
  auto mdiff = make_tensor<value_type>({});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "qr", "run", {rows, cols});
  this->pb->NumpyToTensorView(A, "A");

  (mtie(RefOut, RefTau) = qr_solver(A)).run(this->exec);
  auto op = qr_solver(A);
  (mdiff = max(abs(op.Out - RefOut)) + max(abs(op.Tau - RefTau))).run(this->exec);
  this->exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(this->thresh));

  MATX_EXIT_HANDLER();
}
