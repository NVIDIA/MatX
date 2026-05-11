////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include "assert.h"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;

template <typename T> class DenseSolveTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;

  void SetUp() override
  {
    if constexpr (!detail::CheckSolverSupport<GExecType>()) {
      GTEST_SKIP();
    }

    if constexpr (is_select_threads_host_executor_v<GExecType>) {
      HostExecParams params{4};
      exec = SelectThreadsHostExecutor{params};
    }

    pb = std::make_unique<detail::MatXPybind>();
  }

  void TearDown() override { pb.reset(); }

  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.001f;
};

template <typename T>
class DenseSolveTestFloatTypes : public DenseSolveTest<T> {
};

TYPED_TEST_SUITE(DenseSolveTestFloatTypes, MatXFloatNonHalfTypesAllExecs);

TYPED_TEST(DenseSolveTestFloatTypes, SolveVectorRHS)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t n = 8;
  auto A = make_tensor<TestType>({n, n});
  auto B = make_tensor<TestType>({n});
  auto X = make_tensor<TestType>({n});

  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_solver", "solve", "run_vector", {n});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(B, "B");

  // example-begin solve-test-1
  (X = solve(A, B)).run(this->exec);
  // example-end solve-test-1
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, X, "X", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DenseSolveTestFloatTypes, SolveMatrixRHS)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t n = 8;
  constexpr index_t nrhs = 3;
  auto A = make_tensor<TestType>({n, n});
  auto B = make_tensor<TestType>({n, nrhs});
  auto X = make_tensor<TestType>({n, nrhs});

  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_solver", "solve", "run_matrix", {n, nrhs});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(B, "B");

  (X = solve(A, B)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, X, "X", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DenseSolveTestFloatTypes, SolveInExpression)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t n = 8;
  auto A = make_tensor<TestType>({n, n});
  auto B = make_tensor<TestType>({n});
  auto C = make_tensor<TestType>({n});
  auto X = make_tensor<TestType>({n});

  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_solver", "solve", "run_vector_expression", {n});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(B, "B");
  this->pb->NumpyToTensorView(C, "C");

  (X = solve(A, B) * C).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, X, "X", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DenseSolveTestFloatTypes, SolveBatchedVectorRHS)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t batches = 5;
  constexpr index_t n = 8;
  auto A = make_tensor<TestType>({batches, n, n});
  auto B = make_tensor<TestType>({batches, n});
  auto X = make_tensor<TestType>({batches, n});

  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_solver", "solve", "run_batched_vector", {batches, n});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(B, "B");

  (X = solve(A, B)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, X, "X", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DenseSolveTestFloatTypes, SolveBatchedMatrixRHS)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t batches = 5;
  constexpr index_t n = 8;
  constexpr index_t nrhs = 3;
  auto A = make_tensor<TestType>({batches, n, n});
  auto B = make_tensor<TestType>({batches, n, nrhs});
  auto X = make_tensor<TestType>({batches, n, nrhs});

  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_solver", "solve", "run_batched_matrix", {batches, n, nrhs});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(B, "B");

  (X = solve(A, B)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, X, "X", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DenseSolveTestFloatTypes, SolveInPlaceRHS)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t n = 8;
  constexpr index_t nrhs = 3;
  auto A = make_tensor<TestType>({n, n});
  auto B = make_tensor<TestType>({n, nrhs});

  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_solver", "solve", "run_matrix", {n, nrhs});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(B, "B");

  (B = solve(A, B)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, B, "X", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DenseSolveTestFloatTypes, SolveOperatorInputs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t batches = 5;
  constexpr index_t n = 8;
  constexpr index_t nrhs = 3;
  auto Afull = make_tensor<TestType>({batches, n, n + 1});
  auto Bfull = make_tensor<TestType>({batches, n, nrhs + 1});
  auto A = slice(Afull, {0, 0, 0}, {matxEnd, matxEnd, n});
  auto B = slice(Bfull, {0, 0, 0}, {matxEnd, matxEnd, nrhs});
  auto X = make_tensor<TestType>({batches, n, nrhs});

  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_solver", "solve", "run_batched_matrix", {batches, n, nrhs});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(B, "B");

  (X = solve(A, B)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, X, "X", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DenseSolveTestFloatTypes, SolveInvalidShape)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t n = 8;
  auto A = make_tensor<TestType>({n, n});
  auto B = make_tensor<TestType>({n + 1});
  auto X = make_tensor<TestType>({n + 1});

  ASSERT_THROW({
    (X = solve(A, B)).run(this->exec);
    this->exec.sync();
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DenseSolveTestFloatTypes, SolveSingularMatrix)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t n = 4;
  auto A = make_tensor<TestType>({n, n});
  auto B = make_tensor<TestType>({n});
  auto X = make_tensor<TestType>({n});

  for (index_t i = 0; i < n; i++) {
    B(i) = static_cast<TestType>(1);
    for (index_t j = 0; j < n; j++) {
      A(i, j) = static_cast<TestType>(0);
    }
  }

  ASSERT_THROW({
    (X = solve(A, B)).run(this->exec);
    this->exec.sync();
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}
