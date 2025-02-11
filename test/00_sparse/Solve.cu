////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

template <typename T> class SolveSparseTest : public ::testing::Test {
protected:
  float thresh = 0.001f;
};

template <typename T> class SolveSparseTestsAll : public SolveSparseTest<T> { };

TYPED_TEST_SUITE(SolveSparseTestsAll, MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(SolveSparseTestsAll, SolveCSR) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  //
  // Setup a system of equations AX=Y, where X is the unknown.
  // Solve for sparse A in CSR format.
  //
  // | 1 2 0 0 |   | 1 5 |   |  5 17 |
  // | 0 3 0 0 | x | 2 6 | = |  6 18 |
  // | 0 0 4 0 |   | 3 7 |   | 12 28 |
  // | 0 0 0 5 |   | 4 8 |   | 20 40 |
  //
  auto A = make_tensor<TestType>({4, 4});
  auto X = make_tensor<TestType>({4, 2});
  auto E = make_tensor<TestType>({4, 2});
  auto Y = make_tensor<TestType>({4, 2});
  // Coeffs.
  A(0, 0) = static_cast<TestType>(1); A(0, 1) = static_cast<TestType>(2);
  A(0, 2) = static_cast<TestType>(0); A(0, 3) = static_cast<TestType>(0);
  A(1, 0) = static_cast<TestType>(0); A(1, 1) = static_cast<TestType>(3);
  A(1, 2) = static_cast<TestType>(0); A(1, 3) = static_cast<TestType>(0);
  A(2, 0) = static_cast<TestType>(0); A(2, 1) = static_cast<TestType>(0);
  A(2, 2) = static_cast<TestType>(4); A(2, 3) = static_cast<TestType>(0);
  A(3, 0) = static_cast<TestType>(0); A(3, 1) = static_cast<TestType>(0);
  A(3, 2) = static_cast<TestType>(0); A(3, 3) = static_cast<TestType>(5);
  // Expected.
  E(0, 0) = static_cast<TestType>(1); E(0, 1) = static_cast<TestType>(5);
  E(1, 0) = static_cast<TestType>(2); E(1, 1) = static_cast<TestType>(6);
  E(2, 0) = static_cast<TestType>(3); E(2, 1) = static_cast<TestType>(7);
  E(3, 0) = static_cast<TestType>(4); E(3, 1) = static_cast<TestType>(8);
  // RHS.
  Y(0, 0) = static_cast<TestType>(5); Y(0, 1) = static_cast<TestType>(17);
  Y(1, 0) = static_cast<TestType>(6); Y(1, 1) = static_cast<TestType>(18);
  Y(2, 0) = static_cast<TestType>(12); Y(2, 1) = static_cast<TestType>(28);
  Y(3, 0) = static_cast<TestType>(20); Y(3, 1) = static_cast<TestType>(40);

  // Convert dense A to sparse S in CSR format with int-32 indices.
  auto S = experimental::make_zero_tensor_csr<TestType, int32_t, int32_t>({4, 4});
  (S = dense2sparse(A)).run(exec);
  ASSERT_EQ(S.Nse(), 5);

  // Solve the system.
  (X = solve(S, Y)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 2; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(X(i, j).real(), E(i, j).real(), this->thresh);
        ASSERT_NEAR(X(i, j).imag(), E(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(X(i, j), E(i, j), this->thresh);
      }
    }
  }

  // Allow dense computations (pre-/post-solve).
  TestType C3 = static_cast<TestType>(3);
  TestType C5 = static_cast<TestType>(5);
  (Y = (Y - C3)).run(exec);
  (X = solve(S, Y + C3) + C5).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 2; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR((X(i, j) - C5).real(), E(i, j).real(), this->thresh);
        ASSERT_NEAR((X(i, j) - C5).imag(), E(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(X(i, j) - C5, E(i, j), this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}
