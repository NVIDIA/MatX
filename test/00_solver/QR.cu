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

#if defined(MATX_EN_MATHDX) && defined(MATX_EN_JIT)
template <typename TensorType>
class QRSolverJITTestFloatTypes : public ::testing::Test {
};

TYPED_TEST_SUITE(QRSolverJITTestFloatTypes,
                 MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxSingleMatrixRejectsQRProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t rows = 4;
  constexpr index_t cols = 3;
  auto A = make_tensor<TestType>({rows, cols});
  auto Q = make_tensor<TestType>({rows, rows});
  auto R = make_tensor<TestType>({rows, cols});
  auto op = qr(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Q));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.R));

  CUDAJITExecutor exec{};
  EXPECT_THROW({ (Q = op.Q).run(exec); }, matx::detail::matxException);
  EXPECT_THROW({ (R = op.R).run(exec); }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxBatchedMatrixRejectsQRProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t batches = 3;
  constexpr index_t rows = 4;
  constexpr index_t cols = 3;
  auto A = make_tensor<TestType>({batches, rows, cols});
  auto Q = make_tensor<TestType>({batches, rows, rows});
  auto R = make_tensor<TestType>({batches, rows, cols});
  auto op = qr(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Q));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.R));

  CUDAJITExecutor exec{};
  EXPECT_THROW({ (Q = op.Q).run(exec); }, matx::detail::matxException);
  EXPECT_THROW({ (R = op.R).run(exec); }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxSingleMatrixRejectsQRSolverProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t rows = 4;
  constexpr index_t cols = 3;
  auto A = make_tensor<TestType>({rows, cols});
  auto Out = make_tensor<TestType>({rows, cols});
  auto Tau = make_tensor<TestType>({std::min(rows, cols)});
  auto op = qr_solver(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Out));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Tau));

  CUDAJITExecutor exec{};
  EXPECT_THROW({ (Out = op.Out).run(exec); }, matx::detail::matxException);
  EXPECT_THROW({ (Tau = op.Tau).run(exec); }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(QRSolverJITTestFloatTypes, CuSolverDxBatchedMatrixRejectsQRSolverProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t batches = 3;
  constexpr index_t rows = 4;
  constexpr index_t cols = 3;
  auto A = make_tensor<TestType>({batches, rows, cols});
  auto Out = make_tensor<TestType>({batches, rows, cols});
  auto Tau = make_tensor<TestType>({batches, std::min(rows, cols)});
  auto op = qr_solver(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Out));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Tau));

  CUDAJITExecutor exec{};
  EXPECT_THROW({ (Out = op.Out).run(exec); }, matx::detail::matxException);
  EXPECT_THROW({ (Tau = op.Tau).run(exec); }, matx::detail::matxException);

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
