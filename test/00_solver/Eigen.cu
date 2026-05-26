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
constexpr int dim_size = 100;

template <typename T> class EigenSolverTest : public ::testing::Test {
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
class EigenSolverTestFloatTypes : public EigenSolverTest<TensorType> {
};

template <typename TensorType>
class EigenProjectionSolverTestFloatTypes : public ::testing::Test {
};

TYPED_TEST_SUITE(EigenSolverTestFloatTypes,
                 MatXFloatNonHalfTypesAllExecs);
TYPED_TEST_SUITE(EigenProjectionSolverTestFloatTypes,
                 MatXFloatNonHalfTypesCUDAExec);

template <typename T>
T MakeEigenTestValue(double value);

#if defined(MATX_EN_MATHDX) && defined(MATX_EN_JIT)
template <typename TensorType>
class EigenSolverJITTestFloatTypes : public ::testing::Test {
};

TYPED_TEST_SUITE(EigenSolverJITTestFloatTypes,
                 MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(EigenSolverJITTestFloatTypes, CuSolverDxSingleMatrixProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t n = 4;
  auto A = make_tensor<TestType>({n, n});
  auto DiagA = make_tensor<value_type>({n});
  auto Expected = make_tensor<TestType>({n, n});
  auto Combined = make_tensor<TestType>({n, n});
  auto mdiff = make_tensor<value_type>({});

  for (index_t i = 0; i < n; i++) {
    DiagA(i) = value_type(i + 1);
    for (index_t j = 0; j < n; j++) {
      A(i, j) = i == j ? MakeEigenTestValue<TestType>(static_cast<double>(i + 1)) : TestType{};
      Expected(i, j) = i == j ? MakeEigenTestValue<TestType>(1.0) : TestType{};
    }
  }

  auto op = eig(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Vectors));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Values));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  if constexpr (is_complex_v<TestType>) {
    (Combined = op.Vectors + TestType{} * clone<2>(as_type<TestType>(op.Values), {n, matxKeepDim})).run(exec);
    (mdiff = max(abs(Combined - Expected))).run(cuda_exec);
  }
  else {
    (Combined = (clone<2>(as_type<TestType>(op.Values), {n, matxKeepDim}) -
                 clone<2>(as_type<TestType>(DiagA), {matxKeepDim, n})) * op.Vectors).run(exec);
    (mdiff = max(abs(Combined))).run(cuda_exec);
  }
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}

TEST(EigenSolverJITRegression, CuSolverDxMultipleSizesInOneJITExpression)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto A2 = make_tensor<TestType>({2, 2});
  auto A3 = make_tensor<TestType>({3, 3});
  auto Out = make_tensor<TestType>({2});
  auto Ref = make_tensor<TestType>({2});
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

  auto op2 = eig(A2);
  auto op3 = eig(A3);
  auto ref2 = eig(A2);
  auto ref3 = eig(A3);
  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (Out = op2.Values + slice<1>(op3.Values, {0}, {2})).run(jit_exec);
  (Ref = ref2.Values + slice<1>(ref3.Values, {0}, {2})).run(cuda_exec);
  (mdiff = max(abs(Out - Ref))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), TestType(0), TestType(0.001));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EigenSolverJITTestFloatTypes, CuSolverDxBatchedMatrixProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t batches = 3;
  constexpr index_t n = 4;
  auto A = make_tensor<TestType>({batches, n, n});
  auto DiagA = make_tensor<value_type>({batches, n});
  auto Expected = make_tensor<TestType>({batches, n, n});
  auto Combined = make_tensor<TestType>({batches, n, n});
  auto mdiff = make_tensor<value_type>({});

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < n; i++) {
      DiagA(b, i) = value_type(b + i + 1);
      for (index_t j = 0; j < n; j++) {
        A(b, i, j) = i == j ? MakeEigenTestValue<TestType>(static_cast<double>(b + i + 1)) : TestType{};
        Expected(b, i, j) = i == j ? MakeEigenTestValue<TestType>(1.0) : TestType{};
      }
    }
  }

  auto op = eig(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Vectors));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Values));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  if constexpr (is_complex_v<TestType>) {
    (Combined = op.Vectors + TestType{} * clone<3>(as_type<TestType>(op.Values), {matxKeepDim, n, matxKeepDim})).run(exec);
    (mdiff = max(abs(Combined - Expected))).run(cuda_exec);
  }
  else {
    (Combined = (clone<3>(as_type<TestType>(op.Values), {matxKeepDim, n, matxKeepDim}) -
                 clone<3>(as_type<TestType>(DiagA), {matxKeepDim, matxKeepDim, n})) * op.Vectors).run(exec);
    (mdiff = max(abs(Combined))).run(cuda_exec);
  }
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EigenSolverJITTestFloatTypes, CuSolverDxRank4BatchedMatrixProjectionJIT)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t batch0 = 2;
  constexpr index_t batch1 = 2;
  constexpr index_t n = 4;
  auto A = make_tensor<TestType>({batch0, batch1, n, n});
  auto DiagA = make_tensor<value_type>({batch0, batch1, n});
  auto Expected = make_tensor<TestType>({batch0, batch1, n, n});
  auto Combined = make_tensor<TestType>({batch0, batch1, n, n});
  auto mdiff = make_tensor<value_type>({});

  for (index_t b0 = 0; b0 < batch0; b0++) {
    for (index_t b1 = 0; b1 < batch1; b1++) {
      for (index_t i = 0; i < n; i++) {
        DiagA(b0, b1, i) = value_type(b0 + b1 + i + 1);
        for (index_t j = 0; j < n; j++) {
          A(b0, b1, i, j) = i == j ?
            MakeEigenTestValue<TestType>(static_cast<double>(b0 + b1 + i + 1)) :
            TestType{};
          Expected(b0, b1, i, j) = i == j ? MakeEigenTestValue<TestType>(1.0) : TestType{};
        }
      }
    }
  }

  auto op = eig(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Vectors));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Values));

  cudaExecutor cuda_exec{};
  CUDAJITExecutor exec{};
  if constexpr (is_complex_v<TestType>) {
    (Combined = op.Vectors + TestType{} * clone<4>(as_type<TestType>(op.Values), {matxKeepDim, matxKeepDim, n, matxKeepDim})).run(exec);
    (mdiff = max(abs(Combined - Expected))).run(cuda_exec);
  }
  else {
    (Combined = (clone<4>(as_type<TestType>(op.Values), {matxKeepDim, matxKeepDim, n, matxKeepDim}) -
                 clone<4>(as_type<TestType>(DiagA), {matxKeepDim, matxKeepDim, matxKeepDim, n})) * op.Vectors).run(exec);
    (mdiff = max(abs(Combined))).run(cuda_exec);
  }
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}
#endif

template <typename T>
T MakeEigenTestValue(double value)
{
  using SType = typename inner_op_type_t<T>::type;
  if constexpr (is_complex_v<T>) {
    return T{static_cast<SType>(value), static_cast<SType>(0)};
  }
  else {
    return static_cast<T>(value);
  }
}

TYPED_TEST(EigenProjectionSolverTestFloatTypes, ProjectionAPI)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t n = 4;
  cudaExecutor exec{};
  auto Bv = make_tensor<TestType>({n, n});
  auto residual = make_tensor<TestType>({n, n});
  auto mdiff = make_tensor<value_type>({});

  for (index_t i = 0; i < n; i++) {
    for (index_t j = 0; j < n; j++) {
      Bv(i, j) = i == j ? MakeEigenTestValue<TestType>(static_cast<double>(i + 1)) : TestType{};
    }
  }

  // example-begin eig-projection-test-1
  auto op = eig(Bv);
  (residual = matmul(Bv, op.Vectors) - matmul(op.Vectors, diag(as_type<TestType>(op.Values)))).run(exec);
  // example-end eig-projection-test-1
  (mdiff = max(abs(residual))).run(exec);
  exec.sync();

  ASSERT_NEAR(mdiff(), value_type(0), value_type(0.001));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EigenProjectionSolverTestFloatTypes, ProjectionNoVectorModeRejectsVectors)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t n = 4;
  cudaExecutor exec{};
  auto Bv = make_tensor<TestType>({n, n});
  auto Values = make_tensor<value_type>({n});
  auto Vectors = make_tensor<TestType>({n, n});
  auto VectorsPlus = make_tensor<TestType>({n, n});

  for (index_t i = 0; i < n; i++) {
    for (index_t j = 0; j < n; j++) {
      Bv(i, j) = i == j ? MakeEigenTestValue<TestType>(static_cast<double>(i + 1)) : TestType{};
    }
  }

  auto op = eig(Bv, EigenMode::NO_VECTOR);
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op.Vectors));
  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op.Vectors), 0);
  (Values = op.Values).run(exec);
  exec.sync();

  EXPECT_THROW({ (Vectors = op.Vectors).run(exec); }, matx::detail::matxException);
  EXPECT_THROW({ (VectorsPlus = op.Vectors + TestType{}).run(exec); }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EigenSolverTestFloatTypes, EigenBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  auto Bv = make_tensor<TestType>({dim_size, dim_size});
  auto Evv = make_tensor<TestType>({dim_size, dim_size});
  auto Wov = make_tensor<value_type>({dim_size});

  auto Gv = make_tensor<TestType>({dim_size, 1});
  auto Lvv = make_tensor<TestType>({dim_size, 1});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "eig", "run", {dim_size});
  this->pb->NumpyToTensorView(Bv, "B");

  // example-begin eig-test-1
  // Note that eigenvalue/vector solutions are not necessarily ordered in the same way other libraries
  // may order them. When comparing against other libraries it's best to check A*v = lambda * v
  (mtie(Evv, Wov) = eig(Bv)).run(this->exec);
  // example-end eig-test-1

  // Now we need to go through all the eigenvectors and eigenvalues and make
  // sure the results match the equation A*v = lambda*v, where v are the
  // eigenvectors corresponding to the eigenvalue lambda.
  for (index_t i = 0; i < dim_size; i++) {
    auto v = slice<2>(Evv, {0, i}, {matxEnd, i + 1});

    // Compute lambda*v
    (Lvv = Wov(i) * v).run(this->exec);

    // Compute A*v
    (Gv = matmul(Bv, v)).run(this->exec);
    this->exec.sync();

    // Compare
    for (index_t j = 0; j < dim_size; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Gv(j, 0).real(), Lvv(j, 0).real(), this->thresh);
        ASSERT_NEAR(Gv(j, 0).imag(), Lvv(j, 0).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Gv(j, 0), Lvv(j, 0), this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EigenSolverTestFloatTypes, EigenBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr int batches = 10;
  auto Bv = make_tensor<TestType>({batches, dim_size, dim_size});
  auto Evv = make_tensor<TestType>({batches, dim_size, dim_size});
  auto Wov = make_tensor<value_type>({batches, dim_size});

  auto Gv = make_tensor<TestType>({dim_size, 1});
  auto Lvv = make_tensor<TestType>({dim_size, 1});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "eig", "run", {batches, dim_size});
  this->pb->NumpyToTensorView(Bv, "B");

  // Note that eigenvalue/vector solutions are not necessarily ordered in the same way other libraries
  // may order them. When comparing against other libraries it's best to check A*v = lambda * v
  (mtie(Evv, Wov) = eig(Bv)).run(this->exec);

  // Now we need to go through all the eigenvectors and eigenvalues and make
  // sure the results match the equation A*v = lambda*v, where v are the
  // eigenvectors corresponding to the eigenvalue lambda.
  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < dim_size; i++) {
      // ith col vector from bth batch
      auto v = slice<2>(Evv, {b, 0, i}, {matxDropDim, matxEnd, i + 1});

      // Compute lambda*v
      (Lvv = Wov(b, i) * v).run(this->exec);

      // Compute A*v
      auto Bv_b = slice<2>(Bv, {b, 0, 0}, {matxDropDim, matxEnd, matxEnd});
      (Gv = matmul(Bv_b, v)).run(this->exec);
      this->exec.sync();

      // Compare
      for (index_t j = 0; j < dim_size; j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Gv(j, 0).real(), Lvv(j, 0).real(), this->thresh);
          ASSERT_NEAR(Gv(j, 0).imag(), Lvv(j, 0).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Gv(j, 0), Lvv(j, 0), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}
