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
// | 1 2 0 0 |
// | 0 3 0 0 |
// | 0 0 4 0 |
// | 0 0 0 5 |
//
template <typename T> static auto makeA() {
  const index_t m = 4;
  const index_t n = 4;
  tensor_t<T, 2> A = make_tensor<T>({m, n});
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      A(i, j) = static_cast<T>(0);
    }
  }
  A(0, 0) = static_cast<T>(1);
  A(0, 1) = static_cast<T>(2);
  A(1, 1) = static_cast<T>(3);
  A(2, 2) = static_cast<T>(4);
  A(3, 3) = static_cast<T>(5);
  return A;
}

template <typename T> static auto makeB() {
  const index_t n = 4;
  tensor_t<T, 1> B = make_tensor<T>({n});
  B(0) = static_cast<T>(1);
  B(1) = static_cast<T>(2);
  B(2) = static_cast<T>(3);
  B(3) = static_cast<T>(4);
  return B;
}

template <typename T> static auto makeC() {
  const index_t m = 4;
  tensor_t<T, 1> E = make_tensor<T>({m});
  E(0) = static_cast<T>(5);
  E(1) = static_cast<T>(6); 
  E(2) = static_cast<T>(12);
  E(3) = static_cast<T>(20);
  return E;
}

template <typename T> class MatvecSparseTest : public ::testing::Test {
protected:
  float thresh = 0.001f;
};

template <typename T> class MatvecSparseTestsAll : public MatvecSparseTest<T> { };

TYPED_TEST_SUITE(MatvecSparseTestsAll, MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(MatvecSparseTestsAll, MatvecCOO) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeA<TestType>();
  auto B = makeB<TestType>();
  auto C = makeC<TestType>();
  const auto m = A.Size(0);
  const auto n = A.Size(1);

  // Convert dense A to sparse S.
  auto S = experimental::make_zero_tensor_coo<TestType, index_t>({m, n});
  (S = dense2sparse(A)).run(exec);
  ASSERT_EQ(S.Nse(), 5);

  // Matvec.
  auto O = make_tensor<TestType>({m});
  (O = matvec(S, B)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    if constexpr (is_complex_v<TestType>) {
      ASSERT_NEAR(O(i).real(), C(i).real(), this->thresh);
      ASSERT_NEAR(O(i).imag(), C(i).imag(), this->thresh);
    }
    else {
      ASSERT_NEAR(O(i), C(i), this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatvecSparseTestsAll, MatvecCSR) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeA<TestType>();
  auto B = makeB<TestType>();
  auto C = makeC<TestType>();
  const auto m = A.Size(0);
  const auto n = A.Size(1);

  // Convert dense A to sparse S.
  auto S = experimental::make_zero_tensor_csr<TestType, index_t, index_t>({m, n});
  (S = dense2sparse(A)).run(exec);
  ASSERT_EQ(S.Nse(), 5);

  // Matvec.
  auto O = make_tensor<TestType>({m});
  (O = matvec(S, B)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    if constexpr (is_complex_v<TestType>) {
      ASSERT_NEAR(O(i).real(), C(i).real(), this->thresh);
      ASSERT_NEAR(O(i).imag(), C(i).imag(), this->thresh);
    }
    else {
      ASSERT_NEAR(O(i), C(i), this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatvecSparseTestsAll, MatvecCSC) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto A = makeA<TestType>();
  auto B = makeB<TestType>();
  auto C = makeC<TestType>();
  const auto m = A.Size(0);
  const auto n = A.Size(1);

  // Convert dense A to sparse S.
  auto S = experimental::make_zero_tensor_csc<TestType, index_t, index_t>({m, n});
  (S = dense2sparse(A)).run(exec);
  ASSERT_EQ(S.Nse(), 5);

  // Matvec.
  auto O = make_tensor<TestType>({m});
  (O = matvec(S, B)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    if constexpr (is_complex_v<TestType>) {
      ASSERT_NEAR(O(i).real(), C(i).real(), this->thresh);
      ASSERT_NEAR(O(i).imag(), C(i).imag(), this->thresh);
    }
    else {
      ASSERT_NEAR(O(i), C(i), this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}
