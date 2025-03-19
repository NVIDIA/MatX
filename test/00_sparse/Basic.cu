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

template <typename T> class BasicSparseTest : public ::testing::Test { };

template <typename T> class BasicSparseTestsAll : public BasicSparseTest<T> { };

TYPED_TEST_SUITE(BasicSparseTestsAll, MatXAllTypesAllExecs);

TYPED_TEST(BasicSparseTestsAll, MakeZeroCOO) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto A = experimental::make_zero_tensor_coo<TestType, index_t>({17, 33});
  ASSERT_EQ(A.Rank(), 2);
  ASSERT_EQ(A.Size(0), 17);
  ASSERT_EQ(A.Size(1), 33);
  ASSERT_EQ(A.Nse(), 0);
  ASSERT_EQ(A.posSize(0), 2);
  ASSERT_EQ(A.posSize(1), 0);
  ASSERT_EQ(A.crdSize(0), 0);
  ASSERT_EQ(A.crdSize(1), 0);
  // Element getter.
  ASSERT_EQ(A(0, 0), static_cast<TestType>(0)); // not found
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicSparseTestsAll, MakeCOO) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto vals = make_tensor<TestType>({3});
  auto idxi = make_tensor<index_t>({3});
  auto idxj = make_tensor<index_t>({3});
  vals(0) = static_cast<TestType>(10);
  vals(1) = static_cast<TestType>(20);
  vals(2) = static_cast<TestType>(30);
  idxi(0) = 0; idxi(1) = 0; idxi(2) = 3;
  idxj(0) = 3; idxj(1) = 6; idxj(2) = 7;
  auto A = experimental::make_tensor_coo(vals, idxi, idxj, {4, 8});
  ASSERT_EQ(A.Rank(), 2);
  ASSERT_EQ(A.Size(0), 4);
  ASSERT_EQ(A.Size(1), 8);
  ASSERT_EQ(A.Nse(), 3);
  ASSERT_EQ(A.posSize(0), 2);
  ASSERT_EQ(A.posSize(1), 0);
  ASSERT_EQ(A.crdSize(0), 3);
  ASSERT_EQ(A.crdSize(1), 3);
  // Element getters.
  ASSERT_EQ(A(0, 0), static_cast<TestType>(0)); // not found
  ASSERT_EQ(A(0, 3), vals(0));
  ASSERT_EQ(A(0, 6), vals(1));
  ASSERT_EQ(A(3, 7), vals(2));
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicSparseTestsAll, MakeZeroCSR) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto A = experimental::make_zero_tensor_csr<TestType, index_t, index_t>({17, 33});
  ASSERT_EQ(A.Rank(), 2);
  ASSERT_EQ(A.Size(0), 17);
  ASSERT_EQ(A.Size(1), 33);
  ASSERT_EQ(A.Nse(), 0);
  ASSERT_EQ(A.posSize(0), 0);
  ASSERT_EQ(A.posSize(1), 18);
  ASSERT_EQ(A.crdSize(0), 0);
  ASSERT_EQ(A.crdSize(1), 0);
  // Element getter.
  ASSERT_EQ(A(0, 0), static_cast<TestType>(0)); // not found
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicSparseTestsAll, MakeCSR) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto vals = make_tensor<TestType>({3});
  auto rowp = make_tensor<index_t>({5});
  auto col = make_tensor<index_t>({3});
  vals(0) = static_cast<TestType>(10);
  vals(1) = static_cast<TestType>(20);
  vals(2) = static_cast<TestType>(30);
  rowp(0) = 0; rowp(1) = 2; rowp(2) = 2; rowp(3) = 2; rowp(4) = 3;
  col(0) = 3; col(1) = 6; col(2) = 7;
  auto A = experimental::make_tensor_csr(vals, rowp, col, {4, 8});
  ASSERT_EQ(A.Rank(), 2);
  ASSERT_EQ(A.Size(0), 4);
  ASSERT_EQ(A.Size(1), 8);
  ASSERT_EQ(A.Nse(), 3);
  ASSERT_EQ(A.posSize(0), 0);
  ASSERT_EQ(A.posSize(1), 5);
  ASSERT_EQ(A.crdSize(0), 0);
  ASSERT_EQ(A.crdSize(1), 3);
  // Element getters.
  ASSERT_EQ(A(0, 0), static_cast<TestType>(0)); // not found
  ASSERT_EQ(A(0, 3), vals(0));
  ASSERT_EQ(A(0, 6), vals(1));
  ASSERT_EQ(A(3, 7), vals(2));
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicSparseTestsAll, MakeZeroCSC) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto A = experimental::make_zero_tensor_csc<TestType, index_t, index_t>({17, 33});
  ASSERT_EQ(A.Rank(), 2);
  ASSERT_EQ(A.Size(0), 17);
  ASSERT_EQ(A.Size(1), 33);
  ASSERT_EQ(A.Nse(), 0);
  ASSERT_EQ(A.posSize(0), 0);
  ASSERT_EQ(A.posSize(1), 34);
  ASSERT_EQ(A.crdSize(0), 0);
  ASSERT_EQ(A.crdSize(1), 0);
  // Element getter.
  ASSERT_EQ(A(0, 0), static_cast<TestType>(0)); // not found
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicSparseTestsAll, MakeCSC) {
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto vals = make_tensor<TestType>({3});
  auto row = make_tensor<index_t>({3});
  auto colp = make_tensor<index_t>({9});
  vals(0) = static_cast<TestType>(10);
  vals(1) = static_cast<TestType>(20);
  vals(2) = static_cast<TestType>(30);
  colp(0) = 0; colp(1) = 0; colp(2) = 0;
  colp(3) = 0; colp(4) = 1; colp(5) = 1;
  colp(6) = 1; colp(7) = 2; colp(8) = 3;
  row(0) = 0; row(1) = 0; row(2) = 3;
  auto A = experimental::make_tensor_csc(vals, colp, row, {4, 8});
  ASSERT_EQ(A.Rank(), 2);
  ASSERT_EQ(A.Size(0), 4);
  ASSERT_EQ(A.Size(1), 8);
  ASSERT_EQ(A.Nse(), 3);
  ASSERT_EQ(A.posSize(0), 0);
  ASSERT_EQ(A.posSize(1), 9);
  ASSERT_EQ(A.crdSize(0), 0);
  ASSERT_EQ(A.crdSize(1), 3);
  // Element getters.
  ASSERT_EQ(A(0, 0), static_cast<TestType>(0)); // not found
  ASSERT_EQ(A(0, 3), vals(0));
  ASSERT_EQ(A(0, 6), vals(1));
  ASSERT_EQ(A(3, 7), vals(2));
  MATX_EXIT_HANDLER();
}
