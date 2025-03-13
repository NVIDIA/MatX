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

// Helper method
template <typename T> static auto makeD() {
  const index_t m = 10;
  const index_t n = 10;
  tensor_t<T, 2> D = make_tensor<T>({m, n});
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      D(i, j) = static_cast<T>(0);
    }
  }
  D(0, 1) = static_cast<T>(1);
  D(4, 4) = static_cast<T>(2);
  D(9, 1) = static_cast<T>(3);
  D(9, 9) = static_cast<T>(4);
  return D;
}

template <typename T> class ConvertSparseTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;
  void SetUp() override {
    CheckTestTypeSupport<GTestType>();
  }
};

template <typename T> class ConvertSparseTestsAll : public ConvertSparseTest<T> { };

TYPED_TEST_SUITE(ConvertSparseTestsAll, MatXFloatNonComplexTypesCUDAExec);

TYPED_TEST(ConvertSparseTestsAll, ConvertCOO) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto D = makeD<TestType>();
  const auto m = D.Size(0);
  const auto n = D.Size(1);

  // Convert dense D to sparse S.
  auto S = experimental::make_zero_tensor_coo<TestType, index_t>({m, n});
  (S = dense2sparse(D)).run(exec);
  ASSERT_EQ(S.Rank(), 2);
  ASSERT_EQ(S.Size(0), m);
  ASSERT_EQ(S.Size(1), n);
  ASSERT_EQ(S.Nse(), 4);
  ASSERT_EQ(S.posSize(0), 2);
  ASSERT_EQ(S.posSize(1), 0);
  ASSERT_EQ(S.crdSize(0), 4);
  ASSERT_EQ(S.crdSize(1), 4);

  // Getters are expensive, but fully functional!
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      ASSERT_EQ(S(i, j), D(i, j));
    }
  }

  // Convert sparse S back to dense D.
  auto O = make_tensor<TestType>({m, n});
  (O = sparse2dense(S)).run(exec);
  
  // Back to cheap random-access getters only.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      ASSERT_EQ(O(i, j), D(i, j));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ConvertSparseTestsAll, ConvertCSR) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto D = makeD<TestType>();
  const auto m = D.Size(0);
  const auto n = D.Size(1);

  // Convert dense D to sparse S.
  auto S = experimental::make_zero_tensor_csr<TestType, index_t, index_t>({m, n});
  (S = dense2sparse(D)).run(exec);
  ASSERT_EQ(S.Rank(), 2);
  ASSERT_EQ(S.Size(0), m);
  ASSERT_EQ(S.Size(1), n);
  ASSERT_EQ(S.Nse(), 4);
  ASSERT_EQ(S.posSize(0), 0);
  ASSERT_EQ(S.posSize(1), m + 1);
  ASSERT_EQ(S.crdSize(0), 0);
  ASSERT_EQ(S.crdSize(1), 4);

  // Getters are expensive, but fully functional!
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      ASSERT_EQ(S(i, j), D(i, j));
    }
  }

  // Convert sparse S back to dense D.
  auto O = make_tensor<TestType>({m, n});
  (O = sparse2dense(S)).run(exec);
  
  // Back to cheap random-access getters only.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      ASSERT_EQ(O(i, j), D(i, j));
    }
  }

  // Allow transforming output.
  (transpose(O) = sparse2dense(S)).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      ASSERT_EQ(O(j, i), D(i, j));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ConvertSparseTestsAll, ConvertCSC) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto D = makeD<TestType>();
  const auto m = D.Size(0);
  const auto n = D.Size(1);

  // Convert dense D to sparse S.
  auto S = experimental::make_zero_tensor_csc<TestType, index_t, index_t>({m, n});
  (S = dense2sparse(D)).run(exec);
  ASSERT_EQ(S.Rank(), 2);
  ASSERT_EQ(S.Size(0), m);
  ASSERT_EQ(S.Size(1), n);
  ASSERT_EQ(S.Nse(), 4);
  ASSERT_EQ(S.posSize(0), 0);
  ASSERT_EQ(S.posSize(1), n + 1);
  ASSERT_EQ(S.crdSize(0), 0);
  ASSERT_EQ(S.crdSize(1), 4);

  // Getters are expensive, but fully functional!
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      ASSERT_EQ(S(i, j), D(i, j));
    }
  }

  // Convert sparse S back to dense D.
  auto O = make_tensor<TestType>({m, n});
  (O = sparse2dense(S)).run(exec);
  
  // Back to cheap random-access getters only.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      ASSERT_EQ(O(i, j), D(i, j));
    }
  }

  // Allow dense computations (pre-convert).
  TestType C3 = static_cast<TestType>(3);
  (S = dense2sparse(D + C3)).run(exec);

  ASSERT_EQ(S.Nse(), 100); // fully dense now

  // Allow dense computations (post-convert).
  TestType C5 = static_cast<TestType>(5);
  (O = sparse2dense(S) + C5).run(exec);

  // Verify result.
  exec.sync();
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      ASSERT_EQ(O(i, j) - C5, D(i, j) + C3);
    }
  }

  MATX_EXIT_HANDLER();
}
