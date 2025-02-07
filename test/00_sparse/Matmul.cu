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

//
// Helper method to construct:
//
//   | 1, 2, 0, 0, 0, 0, 0, 0 |
//   | 0, 0, 0, 0, 0, 0, 0, 3 |
//   | 0, 0, 0, 0, 0, 0, 4, 0 |
//   | 0, 0, 5, 6, 0, 7, 0, 0 |
//
template <typename T> static auto makeA() {
  const index_t m = 4;
  const index_t n = 8;
  tensor_t<T, 2> A = make_tensor<T>({m, n});
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      A(i, j) = static_cast<T>(0);
    }
  }
  A(0, 0) = static_cast<T>(1);
  A(0, 1) = static_cast<T>(2);
  A(1, 7) = static_cast<T>(3);
  A(2, 6) = static_cast<T>(4);
  A(3, 2) = static_cast<T>(5);
  A(3, 3) = static_cast<T>(6);
  A(3, 5) = static_cast<T>(7);
  return A;
}

template <typename T> static auto makeB() {
  const index_t m = 8;
  const index_t n = 2;
  tensor_t<T, 2> B = make_tensor<T>({m, n});
  B(0, 0) = static_cast<T>(1);  B(0, 1) = static_cast<T>(2);
  B(1, 0) = static_cast<T>(3);  B(1, 1) = static_cast<T>(4);
  B(2, 0) = static_cast<T>(5);  B(2, 1) = static_cast<T>(6);
  B(3, 0) = static_cast<T>(7);  B(3, 1) = static_cast<T>(8);
  B(4, 0) = static_cast<T>(9);  B(4, 1) = static_cast<T>(10);
  B(5, 0) = static_cast<T>(11); B(5, 1) = static_cast<T>(12);
  B(6, 0) = static_cast<T>(13); B(6, 1) = static_cast<T>(14);
  B(7, 0) = static_cast<T>(15); B(7, 1) = static_cast<T>(16);
  return B;
}

template <typename T> static auto makeE() {
  const index_t m = 4;
  const index_t n = 2;
  tensor_t<T, 2> E = make_tensor<T>({m, n});
  E(0, 0) = static_cast<T>(7);   E(0, 1) = static_cast<T>(10);
  E(1, 0) = static_cast<T>(45);  E(1, 1) = static_cast<T>(48);
  E(2, 0) = static_cast<T>(52);  E(2, 1) = static_cast<T>(56);
  E(3, 0) = static_cast<T>(144); E(3, 1) = static_cast<T>(162);
  return E;
}

template <typename T> class MatmulSparseTest : public ::testing::Test {
protected:
  float thresh = 0.001f;
};

template <typename T> class MatmulSparseTestsAll : public MatmulSparseTest<T> { };

TYPED_TEST_SUITE(MatmulSparseTestsAll, MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(MatmulSparseTestsAll, MatmulCOO) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeA<TestType>();
  auto B = makeB<TestType>();
  auto E = makeE<TestType>();
  const auto m = A.Size(0);
  const auto k = A.Size(1);
  const auto n = B.Size(1);

  // Convert dense A to sparse S.
  auto S = experimental::make_zero_tensor_coo<TestType, index_t>({m, k});
  (S = dense2sparse(A)).run(exec);
  ASSERT_EQ(S.Nse(), 7);
  ASSERT_EQ(S.posSize(1), 0);

  // Matmul.
  auto O = make_tensor<TestType>({m, n});
  (O = matmul(S, B)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(O(i, j).real(), E(i, j).real(), this->thresh);
        ASSERT_NEAR(O(i, j).imag(), E(i,j ).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(O(i, j), E(i, j), this->thresh);
      }

    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatmulSparseTestsAll, MatmulCSR) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeA<TestType>();
  auto B = makeB<TestType>();
  auto E = makeE<TestType>();
  const auto m = A.Size(0);
  const auto k = A.Size(1);
  const auto n = B.Size(1);

  // Convert dense A to sparse S.
  auto S = experimental::make_zero_tensor_csr<TestType, index_t, index_t>({m, k});
  (S = dense2sparse(A)).run(exec);
  ASSERT_EQ(S.Nse(), 7);
  ASSERT_EQ(S.posSize(1), m + 1);

  // Matmul.
  auto O = make_tensor<TestType>({m, n});
  (O = matmul(S, B)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(O(i, j).real(), E(i, j).real(), this->thresh);
        ASSERT_NEAR(O(i, j).imag(), E(i,j ).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(O(i, j), E(i, j), this->thresh);
      }

    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatmulSparseTestsAll, MatmulCSC) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeA<TestType>();
  auto B = makeB<TestType>();
  auto E = makeE<TestType>();
  const auto m = A.Size(0);
  const auto k = A.Size(1);
  const auto n = B.Size(1);

  // Convert dense A to sparse S.
  auto S = experimental::make_zero_tensor_csc<TestType, index_t, index_t>({m, k});
  (S = dense2sparse(A)).run(exec);
  ASSERT_EQ(S.Nse(), 7);
  ASSERT_EQ(S.posSize(1), k + 1);

  // Matmul.
  auto O = make_tensor<TestType>({m, n});
  (O = matmul(S, B)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(O(i, j).real(), E(i, j).real(), this->thresh);
        ASSERT_NEAR(O(i, j).imag(), E(i,j ).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(O(i, j), E(i, j), this->thresh);
      }

    }
  }

  MATX_EXIT_HANDLER();
}
