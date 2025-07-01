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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
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

//
// Helper method to construct:
//
// |  1  1  0  0 |
// | -1  2  1  0 |
// |  0 -1  3  1 |
// |  0  0 -1  4 |
//
template <typename T, typename IDX> static auto makeDIA() {
  const index_t n = 4;
  const index_t d = 3;
  auto D = make_tensor<T>({d * n});
  // DL //////////////////////
  D(0) = static_cast<T>(-1);
  D(1) = static_cast<T>(-1);
  D(2) = static_cast<T>(-1);
  D(3) = static_cast<T>(-1);
  // DM //////////////////////
  D(4) = static_cast<T>(1);
  D(5) = static_cast<T>(2);
  D(6) = static_cast<T>(3);
  D(7) = static_cast<T>(4);
  // DU //////////////////////
  D(8) = static_cast<T>(1);
  D(9) = static_cast<T>(1);
  D(10) = static_cast<T>(1);
  D(11) = static_cast<T>(1);
  // FIX DL/DU ///////////////
  if constexpr (std::is_same_v<IDX, experimental::DIA_INDEX_I>) {
    D(0) = static_cast<T>(0);
    D(11) = static_cast<T>(0);
  } else {
    D(3) = static_cast<T>(0);
    D(8) = static_cast<T>(0);
  }
  ////////////////////////////
  auto O = make_tensor<index_t>({d});
  O(0) = -1;
  O(1) = 0;
  O(2) = 1;
  return experimental::make_tensor_dia<IDX>(D, O, {n, n});
}

template <typename T> static auto makeB() {
  const index_t n = 4;
  tensor_t<T, 1> B = make_tensor<T>({n});
  B(0) = static_cast<T>(10);
  B(1) = static_cast<T>(20);
  B(2) = static_cast<T>(30);
  B(3) = static_cast<T>(40);
  return B;
}

template <typename T> static auto makeC() {
  const index_t n = 4;
  tensor_t<T, 1> C = make_tensor<T>({n});
  C(0) = static_cast<T>(30);
  C(1) = static_cast<T>(60);
  C(2) = static_cast<T>(110);
  C(3) = static_cast<T>(130);
  return C;
}

template <typename T> class DiaSparseTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;
  void SetUp() override { CheckTestTypeSupport<GTestType>(); }
  float thresh = 0.001f;
};

template <typename T> class DiaSparseTestsAll : public DiaSparseTest<T> {};

template <typename T> class DiaSolveSparseTestsAll : public DiaSparseTest<T> {};

TYPED_TEST_SUITE(DiaSparseTestsAll, MatXFloatNonComplexHalfTypesCUDAExec);
TYPED_TEST_SUITE(DiaSolveSparseTestsAll, MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(DiaSparseTestsAll, MatvecDIAI) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeDIA<TestType, experimental::DIA_INDEX_I>();
  auto B = makeB<TestType>();
  auto C = makeC<TestType>();

  const auto m = A.Size(0);
  const auto n = A.Size(1);

  // Matvec.
  auto O = make_tensor<TestType>({m});
  (O = matvec(A, B)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    if constexpr (is_complex_v<TestType>) {
      ASSERT_NEAR(O(i).real(), C(i).real(), this->thresh);
      ASSERT_NEAR(O(i).imag(), C(i).imag(), this->thresh);
    } else {
      ASSERT_NEAR(O(i), C(i), this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DiaSparseTestsAll, MatvecDIAJ) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeDIA<TestType, experimental::DIA_INDEX_J>();
  auto B = makeB<TestType>();
  auto C = makeC<TestType>();

  const auto m = A.Size(0);
  const auto n = A.Size(1);

  // Matvec.
  auto O = make_tensor<TestType>({m});
  (O = matvec(A, B)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    if constexpr (is_complex_v<TestType>) {
      ASSERT_NEAR(O(i).real(), C(i).real(), this->thresh);
      ASSERT_NEAR(O(i).imag(), C(i).imag(), this->thresh);
    } else {
      ASSERT_NEAR(O(i), C(i), this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DiaSolveSparseTestsAll, SolveDIAI) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeDIA<TestType, experimental::DIA_INDEX_I>();

  const auto m = A.Size(0);

  auto X = make_tensor<TestType, 2>({2, m});

  X(0, 0) = static_cast<TestType>(3);
  X(0, 1) = static_cast<TestType>(6);
  X(0, 2) = static_cast<TestType>(11);
  X(0, 3) = static_cast<TestType>(13);
  X(1, 0) = static_cast<TestType>(30);
  X(1, 1) = static_cast<TestType>(60);
  X(1, 2) = static_cast<TestType>(110);
  X(1, 3) = static_cast<TestType>(130);

  // Solve.
  (X = solve(A, X)).run(exec);

  // Verify result.
  exec.sync();
  auto E = make_tensor<TestType>({2, 4});
  E(0, 0) = static_cast<TestType>(1);
  E(0, 1) = static_cast<TestType>(2);
  E(0, 2) = static_cast<TestType>(3);
  E(0, 3) = static_cast<TestType>(4);
  E(1, 0) = static_cast<TestType>(10);
  E(1, 1) = static_cast<TestType>(20);
  E(1, 2) = static_cast<TestType>(30);
  E(1, 3) = static_cast<TestType>(40);
  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 4; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(X(i, j).real(), E(i, j).real(), this->thresh);
        ASSERT_NEAR(X(i, j).imag(), E(i, j).imag(), this->thresh);
      } else {
        ASSERT_NEAR(X(i, j), E(i, j), this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DiaSolveSparseTestsAll, SolveBatchedUniformDIAI) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  // batch0          batch1             batch0       batch1
  // | 10 -1  0  0 | | 20 -2  0  0 | x  | 1  2  3  4 |  5  6   7   8 |
  // | -1 10 -1  0 | | -2 20 -2  0 |
  // |  0 -1 10 -1 | |  0 -2 20 -2 | =  | 8 16 24 37 | 88 96 112 146 |
  // |  0  0 -1 10 | |  0  0 -2 20 |
  auto D = make_tensor<TestType>({2 * 3 * 4});
  D(0) = static_cast<TestType>(0);
  D(1) = static_cast<TestType>(-1);
  D(2) = static_cast<TestType>(-1);
  D(3) = static_cast<TestType>(-1);
  D(4) = static_cast<TestType>(0);
  D(5) = static_cast<TestType>(-2);
  D(6) = static_cast<TestType>(-2);
  D(7) = static_cast<TestType>(-2);
  D(8) = static_cast<TestType>(10);
  D(9) = static_cast<TestType>(10);
  D(10) = static_cast<TestType>(10);
  D(11) = static_cast<TestType>(10);
  D(12) = static_cast<TestType>(20);
  D(13) = static_cast<TestType>(20);
  D(14) = static_cast<TestType>(20);
  D(15) = static_cast<TestType>(20);
  D(16) = static_cast<TestType>(-1);
  D(17) = static_cast<TestType>(-1);
  D(18) = static_cast<TestType>(-1);
  D(19) = static_cast<TestType>(0);
  D(20) = static_cast<TestType>(-2);
  D(21) = static_cast<TestType>(-2);
  D(22) = static_cast<TestType>(-2);
  D(23) = static_cast<TestType>(0);
  auto O = make_tensor<index_t>({3});
  O(0) = -1;
  O(1) = 0;
  O(2) = 1;
  auto A =
      experimental::make_tensor_uniform_batched_dia<experimental::DIA_INDEX_I>(
          D, O, {2, 4, 4});

  // RHS.
  auto X = make_tensor<TestType>({2 * 4});
  X(0) = static_cast<TestType>(8);
  X(1) = static_cast<TestType>(16);
  X(2) = static_cast<TestType>(24);
  X(3) = static_cast<TestType>(37);
  X(4) = static_cast<TestType>(88);
  X(5) = static_cast<TestType>(96);
  X(6) = static_cast<TestType>(112);
  X(7) = static_cast<TestType>(146);

  // Expected.
  auto E = make_tensor<TestType>({2 * 4});
  E(0) = static_cast<TestType>(1);
  E(1) = static_cast<TestType>(2);
  E(2) = static_cast<TestType>(3);
  E(3) = static_cast<TestType>(4);
  E(4) = static_cast<TestType>(5);
  E(5) = static_cast<TestType>(6);
  E(6) = static_cast<TestType>(7);
  E(7) = static_cast<TestType>(8);

  // Solve.
  (X = solve(A, X)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < 2 * 4; i++) {
    if constexpr (is_complex_v<TestType>) {
      ASSERT_NEAR(X(i).real(), E(i).real(), this->thresh);
      ASSERT_NEAR(X(i).imag(), E(i).imag(), this->thresh);
    } else {
      ASSERT_NEAR(X(i), E(i), this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}
