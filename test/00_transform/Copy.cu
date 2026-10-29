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
#include "gtest/gtest.h"
#include <vector>

using namespace matx;

template <typename TensorType>
class CopyTestsAll : public ::testing::Test
{
};

TYPED_TEST_SUITE(CopyTestsAll, MatXAllTypesAllExecs);

TYPED_TEST(CopyTestsAll, CopyOutParam)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  const int SZ = 5;
  TestType DEFAULT, TEST_VAL;
  if constexpr (cuda::std::is_same_v<TestType, bool>) {
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
      cuda::std::array<index_t, N> dims; \
      dims.fill(SZ); \
      auto in = make_tensor<TestType>(dims); \
      auto out = make_tensor<TestType>(dims); \
      (in = DEFAULT).run(exec); \
      exec.sync(); \
      cuda::std::array<index_t, N> inds; \
      inds.fill(SZ/2); \
      in(inds) = TEST_VAL; \
      exec.sync(); \
      matx::copy(out, in, exec); \
      exec.sync(); \
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
    auto in = make_tensor<TestType>({});
    auto out = make_tensor<TestType>({});
    in() = TEST_VAL;
    exec.sync();
    matx::copy(out, in, exec);
    exec.sync();
    ASSERT_EQ(in(), out());
    ASSERT_EQ(out(), TEST_VAL);
  }

  // Also test that deep copying from a slice works as expected
  {
    auto in = make_tensor<TestType>({SZ, SZ, SZ});
    auto out = make_tensor<TestType>({SZ});
    (in = DEFAULT).run(exec);
    exec.sync();
    in(0, SZ/2, 0) = TEST_VAL;
    exec.sync();
    matx::copy(out, slice<1>(in, {0,0,0}, {matxDropDim,matxEnd,matxDropDim}), exec);
    exec.sync();
    ASSERT_EQ(out.Rank(), 1);
    ASSERT_EQ(out.Size(0), SZ);
    ASSERT_EQ(out(SZ/2), TEST_VAL);
    ASSERT_EQ(out(0), DEFAULT);
  }

  if constexpr (cuda::std::is_same_v<ExecType,cudaExecutor>) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CopyTestsAll, CopyViewsAndExpressions)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};
  constexpr auto memory = is_cuda_executor_v<ExecType> ? MATX_MANAGED_MEMORY : MATX_HOST_MALLOC_MEMORY;

  auto in = make_tensor<TestType>({4, 6}, memory);
  auto out = make_tensor<TestType>({4, 6}, memory);
  auto expected = make_tensor<TestType>({4, 6}, memory);
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 6; j++) {
      in(i, j) = TestType{static_cast<detail::value_promote_t<TestType>>((i + j) % 2)};
    }
  }

  // A noncontiguous source must preserve the logical element order.
  auto transposed = permute(in, {1, 0});
  auto transposed_out = make_tensor<TestType>({6, 4}, memory);
  matx::copy(transposed_out, transposed, exec);

  // A noncontiguous destination must leave the gaps untouched.
  (out = TestType{0}).run(exec);
  (expected = TestType{0}).run(exec);
  auto strided_out = slice(out, {0, 0}, {matxEnd, matxEnd}, {1, 2});
  auto strided_in = slice(in, {0, 0}, {matxEnd, matxEnd}, {1, 2});
  auto strided_expected = slice(expected, {0, 0}, {matxEnd, matxEnd}, {1, 2});
  auto compact = make_tensor<TestType>({4, 3}, memory);
  (compact = strided_in).run(exec);
  (strided_expected = strided_in).run(exec);
  matx::copy(strided_out, compact, exec);
  exec.sync();
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 6; j++) {
      EXPECT_EQ(transposed_out(j, i), in(i, j));
      EXPECT_EQ(out(i, j), expected(i, j));
    }
  }

  // Expression inputs still evaluate through assignment. Bool addition promotes
  // to int, so explicitly convert the result back to the output element type.
  matx::copy(out, as_type<TestType>(in + TestType{0}), exec);
  exec.sync();
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 6; j++) {
      EXPECT_EQ(out(i, j), in(i, j));
    }
  }

  // Contiguous slices copy from their view pointers, including offsets.
  auto in_row = slice<1>(in, {2, 1}, {matxDropDim, 5});
  auto out_row = slice<1>(out, {1, 2}, {matxDropDim, matxEnd});
  (out = TestType{0}).run(exec);
  matx::copy(out_row, in_row, exec);
  exec.sync();
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 6; j++) {
      EXPECT_EQ(out(i, j), (i == 1 && j >= 2) ? in(2, j - 1) : TestType{0});
    }
  }
  MATX_EXIT_HANDLER();
}

TEST(CopyTests, CudaMemcpyOnExecutorStream)
{
  MATX_ENTER_HANDLER();
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  cudaExecutor exec{stream};
  auto in = make_tensor<float>({2, 3, 4}, MATX_DEVICE_MEMORY);
  auto out = make_tensor<float>({2, 3, 4}, MATX_DEVICE_MEMORY);
  auto stream_out = make_tensor<float>({2, 3, 4}, MATX_DEVICE_MEMORY);
  auto expected = make_tensor<float>({2, 3, 4});
  auto actual = make_tensor<float>({2, 3, 4});
  (in = 7.0f).run(exec);
  (expected = in).run(exec);
  exec.sync();

  // Capture verifies both dispatch to memcpy and use of the requested stream.
  cudaGraph_t graph;
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  matx::copy(out, in, exec);
  matx::copy(stream_out, out, stream);
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  size_t count = 0;
  ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &count), cudaSuccess);
  EXPECT_EQ(count, 2u);
  std::vector<cudaGraphNode_t> nodes(count);
  ASSERT_EQ(cudaGraphGetNodes(graph, nodes.data(), &count), cudaSuccess);
  for (auto node : nodes) {
    cudaGraphNodeType type;
    ASSERT_EQ(cudaGraphNodeGetType(node, &type), cudaSuccess);
    EXPECT_EQ(type, cudaGraphNodeTypeMemcpy);
  }
  cudaGraphExec_t graph_exec;
  ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
  (actual = stream_out).run(exec);
  exec.sync();
  for (index_t i = 0; i < actual.TotalSize(); i++) {
    EXPECT_EQ(actual.Data()[i], expected.Data()[i]);
  }
  ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  MATX_EXIT_HANDLER();
}

TEST(CopyTests, TypeConversion)
{
  MATX_ENTER_HANDLER();
  auto in = make_tensor<float>({4});
  auto out = make_tensor<int>({4});
  auto wide_out = make_tensor<double>({4});
  for (index_t i = 0; i < 4; i++) {
    in(i) = static_cast<float>(i) + 0.5f;
  }
  cudaExecutor exec{};
  matx::copy(wide_out, in, exec);
#if MATX_CHECK_NARROWING_CONVERSIONS
  matx::copy(out, as_type<int>(in), exec);
#else
  matx::copy(out, in, exec);
#endif
  exec.sync();
  for (index_t i = 0; i < 4; i++) {
    EXPECT_EQ(out(i), static_cast<int>(in(i)));
    EXPECT_EQ(wide_out(i), static_cast<double>(in(i)));
  }
  MATX_EXIT_HANDLER();
}

TEST(CopyTests, SparseSource)
{
  MATX_ENTER_HANDLER();
  auto values = make_tensor<float>({2});
  auto rows = make_tensor<index_t>({2});
  auto columns = make_tensor<index_t>({2});
  values(0) = 5.0f;
  values(1) = 7.0f;
  rows(0) = 0;
  rows(1) = 1;
  columns(0) = 2;
  columns(1) = 0;
  auto sparse = experimental::make_tensor_coo(values, rows, columns, {2, 3});
  auto out = make_tensor<float>({2, 3});
  auto assigned = make_tensor<float>({2, 3});
  auto expected = make_tensor<float>({2, 3});
  cudaExecutor exec{};
  // An expression forces evaluation of sparse elements, including implicit zeros.
  (expected = sparse + 0.0f).run(exec);
  (assigned = sparse).run(exec);
  matx::copy(out, sparse, exec);
  exec.sync();
  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 3; j++) {
      EXPECT_EQ(out(i, j), expected(i, j));
      EXPECT_EQ(assigned(i, j), expected(i, j));
    }
  }
  MATX_EXIT_HANDLER();
}

#ifdef MATX_EN_UNSAFE_ALIAS_DETECTION
TEST(CopyTests, OverlappingContiguousViews)
{
  // Host allocation allows verifying rejection before any CUDA work is issued.
  auto storage = make_tensor<float>({8}, MATX_HOST_MALLOC_MEMORY);
  auto first = slice(storage, {0}, {5});
  auto second = slice(storage, {3}, {8});
  ASSERT_TRUE(first.IsContiguous());
  ASSERT_TRUE(second.IsContiguous());
  auto expect_overlap = [](auto &out, const auto &in, auto exec) {
    try {
      matx::copy(out, in, exec);
      FAIL() << "Overlapping views must be rejected before copying";
    }
    catch (const detail::matxException &error) {
      EXPECT_EQ(error.e, matxInvalidParameter);
      EXPECT_NE(std::string(error.what()).find("memory ranges overlap"), std::string::npos);
    }
  };
  expect_overlap(second, first, cudaExecutor{});
  expect_overlap(first, second, cudaExecutor{});
  expect_overlap(second, first, cudaStream_t{});
}
#endif

#ifndef NDEBUG
TEST(CopyTests, MismatchedSizes)
{
  auto in = make_tensor<float>({2, 3}, MATX_HOST_MALLOC_MEMORY);
  auto out = make_tensor<float>({2, 4}, MATX_HOST_MALLOC_MEMORY);
  EXPECT_THROW(matx::copy(out, in, cudaExecutor{}), detail::matxException);
}
#endif

TYPED_TEST(CopyTestsAll, CopyReturn)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  const int SZ = 5;
  TestType DEFAULT, TEST_VAL;
  if constexpr (cuda::std::is_same_v<TestType, bool>) {
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
      cuda::std::array<index_t, N> dims; \
      dims.fill(SZ); \
      auto in = make_tensor<TestType>(dims); \
      (in = DEFAULT).run(exec); \
      exec.sync(); \
      cuda::std::array<index_t, N> inds; \
      inds.fill(SZ/2); \
      in(inds) = TEST_VAL; \
      exec.sync(); \
      auto out = matx::copy(in, exec); \
      exec.sync(); \
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
    auto in = make_tensor<TestType>({});
    in() = TEST_VAL;
    exec.sync();
    auto out = matx::copy(in, exec);
    exec.sync();
    ASSERT_EQ(in(), out());
    ASSERT_EQ(out(), TEST_VAL);
  }

  // Also test that deep copying from a slice works as expected
  {
    auto in = make_tensor<TestType>({SZ, SZ, SZ});
    (in = DEFAULT).run(exec);
    exec.sync();
    in(0, SZ/2, 0) = TEST_VAL;
    exec.sync();
    auto out = matx::copy(slice<1>(in, {0,0,0}, {matxDropDim,matxEnd,matxDropDim}), exec);
    exec.sync();
    ASSERT_EQ(out.Rank(), 1);
    ASSERT_EQ(out.Size(0), SZ);
    ASSERT_EQ(out(SZ/2), TEST_VAL);
    ASSERT_EQ(out(0), DEFAULT);
  }

  if constexpr (cuda::std::is_same_v<ExecType,cudaExecutor>) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  MATX_EXIT_HANDLER();
}
