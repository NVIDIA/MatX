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

template <typename TensorType>
class CopyTestsAll : public ::testing::Test
{
};

TYPED_TEST_SUITE(CopyTestsAll, MatXTypesAllExecs);

TYPED_TEST(CopyTestsAll, CopyOutParam)
{
  MATX_ENTER_HANDLER();

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto sync = [&exec]() constexpr {
    if constexpr (std::is_same_v<ExecType,cudaExecutor>) {
      cudaDeviceSynchronize();
    }
  };

  const int SZ = 5;
  TestType DEFAULT, TEST_VAL;
  if constexpr (std::is_same_v<TestType, bool>) {
    DEFAULT = true;
    TEST_VAL = false;
  } else {
    DEFAULT = {2};
    TEST_VAL = {7};
  }

  // The following tests create an N dimensional tensor, N > 0, and populates
  // the full tensor with value DEFAULT, except for the index given by
  // {SZ/2, SZ/2, ..., SZ/2}, which is TEST_VAL. This tensor is then copied
  // and we verify that the copy has the same pattern.
  #define TEST_NUM_DIMS(N) \
    do { \
      std::array<index_t, N> dims; \
      dims.fill(SZ); \
      auto in = make_tensor<TestType>(dims); \
      auto out = make_tensor<TestType>(dims); \
      (in = DEFAULT).run(exec); \
      sync(); \
      std::array<index_t, N> inds; \
      inds.fill(SZ/2); \
      in(inds) = TEST_VAL; \
      sync(); \
      matx::copy(out, in, exec); \
      sync(); \
      ASSERT_EQ(in(inds), out(inds)); \
      ASSERT_EQ(out(inds), TEST_VAL); \
      inds.fill(0); \
      ASSERT_EQ(in(inds), out(inds)); \
      ASSERT_EQ(out(inds), DEFAULT); \
    } while (0);

  TEST_NUM_DIMS(1);
  TEST_NUM_DIMS(2);
  TEST_NUM_DIMS(3);
  TEST_NUM_DIMS(4);
  TEST_NUM_DIMS(5);
  #undef TEST_NUM_DIMS

  // 0D tensors are an exception to the above test because as scalars they can
  // only hold a single value. Thus, we create a 0D tensor with value TEST_VAL
  // and verify that the copy receives the same value.
  {
    auto in = make_tensor<TestType>();
    auto out = make_tensor<TestType>();
    in() = TEST_VAL;
    sync();
    matx::copy(out, in, exec);
    sync();
    ASSERT_EQ(in(), out());
    ASSERT_EQ(out(), TEST_VAL);
  }

  // Also test that deep copying from a slice works as expected
  {
    auto in = make_tensor<TestType>({SZ, SZ, SZ});
    auto out = make_tensor<TestType>({SZ});
    (in = DEFAULT).run(exec);
    sync();
    in(0, SZ/2, 0) = TEST_VAL;
    sync();
    matx::copy(out, slice<1>(in, {0,0,0}, {matxDropDim,matxEnd,matxDropDim}), exec);
    sync();
    ASSERT_EQ(out.Rank(), 1);
    ASSERT_EQ(out.Size(0), SZ);
    ASSERT_EQ(out(SZ/2), TEST_VAL);
    ASSERT_EQ(out(0), DEFAULT);
  }

  if constexpr (std::is_same_v<ExecType,cudaExecutor>) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CopyTestsAll, CopyReturn)
{
  MATX_ENTER_HANDLER();

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto sync = [&exec]() constexpr {
    if constexpr (std::is_same_v<ExecType,cudaExecutor>) {
      cudaDeviceSynchronize();
    }
  };

  const int SZ = 5;
  TestType DEFAULT, TEST_VAL;
  if constexpr (std::is_same_v<TestType, bool>) {
    DEFAULT = true;
    TEST_VAL = false;
  } else {
    DEFAULT = {2};
    TEST_VAL = {7};
  }

  // The following tests create an N dimensional tensor, N > 0, and populates
  // the full tensor with value DEFAULT, except for the index given by
  // {SZ/2, SZ/2, ..., SZ/2}, which is TEST_VAL. This tensor is then copied
  // and we verify that the copy has the same pattern.
  #define TEST_NUM_DIMS(N) \
    do { \
      std::array<index_t, N> dims; \
      dims.fill(SZ); \
      auto in = make_tensor<TestType>(dims); \
      (in = DEFAULT).run(exec); \
      sync(); \
      std::array<index_t, N> inds; \
      inds.fill(SZ/2); \
      in(inds) = TEST_VAL; \
      sync(); \
      auto out = matx::copy(in, exec); \
      sync(); \
      ASSERT_EQ(in(inds), out(inds)); \
      ASSERT_EQ(out(inds), TEST_VAL); \
      inds.fill(0); \
      ASSERT_EQ(in(inds), out(inds)); \
      ASSERT_EQ(out(inds), DEFAULT); \
    } while (0);

  TEST_NUM_DIMS(1);
  TEST_NUM_DIMS(2);
  TEST_NUM_DIMS(3);
  TEST_NUM_DIMS(4);
  TEST_NUM_DIMS(5);
  #undef TEST_NUM_DIMS

  // 0D tensors are an exception to the above test because as scalars they can
  // only hold a single value. Thus, we create a 0D tensor with value TEST_VAL
  // and verify that the copy receives the same value.
  {
    auto in = make_tensor<TestType>();
    in() = TEST_VAL;
    sync();
    auto out = matx::copy(in, exec);
    sync();
    ASSERT_EQ(in(), out());
    ASSERT_EQ(out(), TEST_VAL);
  }

  // Also test that deep copying from a slice works as expected
  {
    auto in = make_tensor<TestType>({SZ, SZ, SZ});
    (in = DEFAULT).run(exec);
    sync();
    in(0, SZ/2, 0) = TEST_VAL;
    sync();
    auto out = matx::copy(slice<1>(in, {0,0,0}, {matxDropDim,matxEnd,matxDropDim}), exec);
    sync();
    ASSERT_EQ(out.Rank(), 1);
    ASSERT_EQ(out.Size(0), SZ);
    ASSERT_EQ(out(SZ/2), TEST_VAL);
    ASSERT_EQ(out(0), DEFAULT);
  }

  if constexpr (std::is_same_v<ExecType,cudaExecutor>) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  MATX_EXIT_HANDLER();
}