////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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

#include "matx.h"
#include "prerun_tester.h"
#include "test_types.h"
#include "gtest/gtest.h"

#include <csignal>

using namespace matx;

namespace {

// The lifecycle probe intentionally does not support CUDAJITExecutor.
using ReductionLifecycleTypesWithoutJIT = TupleToTypes<TypedCartesianProduct<
    MatXNumericNonComplexTuple, ExecutorTypesAllWithoutJIT>::type>::type;

template <typename T>
class ReductionLifecycleTests : public ::testing::Test {};

TYPED_TEST_SUITE(ReductionLifecycleTests, ReductionLifecycleTypesWithoutJIT);

template <typename Exec>
Exec MakeLifecycleExecutor()
{
  if constexpr (std::is_same_v<Exec, SelectThreadsHostExecutor>) {
    return Exec{HostExecParams{2}};
  }
  else {
    return Exec{};
  }
}

template <bool Direct, typename Tensor>
auto MakeLifecycleInput(const Tensor &in, test::PreRunLifecycle &life)
{
  if constexpr (Direct) {
    // Keep the transform visible to the consumer; count lifecycle calls on its input.
    return cumsum(test::make_prerun_tester(in, life));
  }
  else {
    return test::make_prerun_tester(cumsum(in), life);
  }
}

enum class ArgReduction { MinMax, Max, Min };

template <ArgReduction Kind, bool Batched, typename Op>
auto MakeArgReduction(const Op &input)
{
  if constexpr (Kind == ArgReduction::MinMax) {
    if constexpr (Batched) { return argminmax(input, {1}); }
    else { return argminmax(input); }
  }
  else if constexpr (Kind == ArgReduction::Max) {
    if constexpr (Batched) { return argmax(input, {1}); }
    else { return argmax(input); }
  }
  else {
    if constexpr (Batched) { return argmin(input, {1}); }
    else { return argmin(input); }
  }
}

template <typename TypeParam, ArgReduction Kind, bool Direct = false, bool Batched = false>
void CheckArgReductionLifecycle()
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  using Exec = cuda::std::tuple_element_t<1, TypeParam>;
  auto exec = MakeLifecycleExecutor<Exec>();
  // Also exercise the parallel host scan/reduction paths, which use serial
  // fallbacks for the four-element inputs in the other cases.
  constexpr bool parallel_case = Kind == ArgReduction::MinMax && Direct && !Batched &&
      (std::is_same_v<Exec, AllThreadsHostExecutor> || std::is_same_v<Exec, SelectThreadsHostExecutor>);
  constexpr index_t width = parallel_case ? 32768 : 4;
  auto in = [] {
    if constexpr (Batched) { return make_tensor<T>({2, width}); }
    else { return make_tensor<T>({width}); }
  }();
  cuda::std::array<index_t, Batched ? 1 : 0> output_shape{};
  if constexpr (Batched) { output_shape[0] = 2; }
  auto min_values = make_tensor<T>(output_shape);
  auto min_indices = make_tensor<index_t>(output_shape);
  auto max_values = make_tensor<T>(output_shape);
  auto max_indices = make_tensor<index_t>(output_shape);
  test::PreRunLifecycle life;
  auto input = MakeLifecycleInput<Direct>(in, life);
  auto statement = [&] {
    auto reduction = MakeArgReduction<Kind, Batched>(input);
    if constexpr (Kind == ArgReduction::MinMax) {
      return (mtie(min_values, min_indices, max_values, max_indices) = reduction);
    }
    else if constexpr (Kind == ArgReduction::Max) {
      return (mtie(max_values, max_indices) = reduction);
    }
    else {
      return (mtie(min_values, min_indices) = reduction);
    }
  }();
  auto at = [](auto &tensor, index_t batch) -> decltype(auto) {
    if constexpr (Batched) { return tensor(batch); }
    else { return tensor(); }
  };

  for (int iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    const T scale = static_cast<T>(iteration + 1);
    constexpr index_t batches = Batched ? 2 : 1;
    for (index_t b = 0; b < batches; ++b) {
      for (index_t i = 0; i < width; ++i) {
        in.Data()[b * width + i] = static_cast<T>(b + 1) * scale;
      }
    }
    statement.run(exec);
    exec.sync();

    if constexpr (Kind == ArgReduction::MinMax || !is_cuda_executor_v<Exec>) {
      test::ExpectLifecycleClean(life, "arg reduction input", iteration + 1);
    }
    else {
      // CUDA argmax/argmin copy their input with an internal run(), which adds
      // at most one nested, balanced lifecycle per execution.
      test::ExpectLifecycleBalanced(life, "arg reduction input", iteration + 1, 2 * (iteration + 1));
    }
    for (index_t b = 0; b < batches; ++b) {
      const T row_scale = static_cast<T>(b + 1) * scale;
      if constexpr (Kind != ArgReduction::Max) {
        EXPECT_EQ(at(min_values, b), row_scale);
        EXPECT_EQ(at(min_indices, b), b * width);
      }
      if constexpr (Kind != ArgReduction::Min) {
        EXPECT_EQ(at(max_values, b), static_cast<T>(width) * row_scale);
        EXPECT_EQ(at(max_indices, b), b * width + width - 1);
      }
    }
  }
}

TYPED_TEST(ReductionLifecycleTests, ArgMinMaxTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::MinMax>();
}

TYPED_TEST(ReductionLifecycleTests, ArgMinMaxMaterializedTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::MinMax, true>();
}

TYPED_TEST(ReductionLifecycleTests, BatchedArgMinMaxTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::MinMax, false, true>();
}

TYPED_TEST(ReductionLifecycleTests, BatchedArgMinMaxMaterializedTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::MinMax, true, true>();
}

TYPED_TEST(ReductionLifecycleTests, ArgMaxTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::Max>();
}

TYPED_TEST(ReductionLifecycleTests, ArgMaxMaterializedTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::Max, true>();
}

TYPED_TEST(ReductionLifecycleTests, BatchedArgMaxTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::Max, false, true>();
}

TYPED_TEST(ReductionLifecycleTests, BatchedArgMaxMaterializedTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::Max, true, true>();
}

TYPED_TEST(ReductionLifecycleTests, ArgMinTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::Min>();
}

TYPED_TEST(ReductionLifecycleTests, ArgMinMaterializedTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::Min, true>();
}

TYPED_TEST(ReductionLifecycleTests, BatchedArgMinTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::Min, false, true>();
}

TYPED_TEST(ReductionLifecycleTests, BatchedArgMinMaterializedTransformInput)
{
  CheckArgReductionLifecycle<TypeParam, ArgReduction::Min, true, true>();
}

template <typename TypeParam, bool Indices, bool Direct = false>
void CheckFindLifecycle()
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  using Exec = cuda::std::tuple_element_t<1, TypeParam>;
  auto exec = MakeLifecycleExecutor<Exec>();
  auto in = make_tensor<T>({4});
  auto out = make_tensor<std::conditional_t<Indices, index_t, T>>({4});
  auto count = make_tensor<int>({});
  test::PreRunLifecycle life;
  auto input = MakeLifecycleInput<Direct>(in, life);
  auto statement = [&] {
    if constexpr (Indices) { return (mtie(out, count) = find_idx(input, GT{T(2)})); }
    else { return (mtie(out, count) = find(input, GT{T(2)})); }
  }();

  for (int iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    const T scale = static_cast<T>(iteration + 1);
    in.SetVals({scale, scale, scale, scale});
    statement.run(exec);
    exec.sync();

    test::ExpectLifecycleClean(life, "find input", iteration + 1);
    const int first = iteration == 0 ? 2 : 1;
    ASSERT_EQ(count(), 4 - first);
    for (int i = first; i < 4; ++i) {
      if constexpr (Indices) { EXPECT_EQ(out(i - first), i); }
      else { EXPECT_EQ(out(i - first), static_cast<T>(i + 1) * scale); }
    }
  }
}

TYPED_TEST(ReductionLifecycleTests, FindTransformInput)
{
  CheckFindLifecycle<TypeParam, false>();
}

TYPED_TEST(ReductionLifecycleTests, FindIdxTransformInput)
{
  CheckFindLifecycle<TypeParam, true>();
}

TYPED_TEST(ReductionLifecycleTests, FindMaterializedTransformInput)
{
  CheckFindLifecycle<TypeParam, false, true>();
}

TYPED_TEST(ReductionLifecycleTests, FindIdxMaterializedTransformInput)
{
  CheckFindLifecycle<TypeParam, true, true>();
}

// Transform implementations such as softmax run nested expressions holding
// copies of their prepared input. Those copies share the input's temporary
// storage, so their lifecycle hooks must not prepare or clean it up again.
// Run on a non-default stream so that cleanup must follow the stream's
// ordering rather than relying on legacy default-stream synchronization.
template <typename T>
class NestedTransformLifecycleTests : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess); }
  void TearDown() override {
    // Cached plans may hold resources bound to this stream, so release them
    // before destroying it.
    cudaStreamSynchronize(stream_);
    ClearCaches();
    cudaStreamDestroy(stream_);
  }

  cudaExecutor Executor() const { return cudaExecutor{stream_}; }

  cudaStream_t stream_ = nullptr;
};

TYPED_TEST_SUITE(NestedTransformLifecycleTests, MatXFloatNonComplexNonHalfTypesCUDAExec);

template <typename T>
cuda::std::array<T, 4> ExpectedSoftmaxOfCumsum(T scale)
{
  cuda::std::array<T, 4> expected{};
  T sum = 0;
  for (int i = 0; i < 4; ++i) {
    expected[i] = std::exp(static_cast<T>(i - 3) * scale);
    sum += expected[i];
  }
  for (auto &value : expected) {
    value /= sum;
  }
  return expected;
}

template <bool ExactLifecycle = true, typename T, typename Exec, typename Statement, typename Check>
void CheckNestedSoftmaxLifecycle(Exec &exec, tensor_t<T, 1> &in, const test::PreRunLifecycle &life,
                                 Statement &&statement, Check &&check)
{
  for (int iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    const T scale = static_cast<T>(iteration + 1);
    in.SetVals({scale, scale, scale, scale});
    statement.run(exec);
    exec.sync();

    if constexpr (ExactLifecycle) {
      test::ExpectLifecycleClean(life, "nested transform input", iteration + 1);
    }
    else {
      test::ExpectLifecycleBalanced(life, "nested transform input", iteration + 1, 2 * (iteration + 1));
    }
    check(ExpectedSoftmaxOfCumsum(scale));
  }
}

TYPED_TEST(NestedTransformLifecycleTests, Find)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  auto exec = this->Executor();
  auto in = make_tensor<T>({4});
  auto values = make_tensor<T>({4});
  auto count = make_tensor<int>({});
  test::PreRunLifecycle life;
  const T threshold = T(0.2);
  auto statement = (mtie(values, count) = find(softmax(cumsum(test::make_prerun_tester(in, life))), GT{threshold}));

  CheckNestedSoftmaxLifecycle(exec, in, life, statement, [&](const auto &expected) {
    int expected_count = 0;
    for (const T value : expected) {
      if (value > threshold) {
        EXPECT_NEAR(values(expected_count), value, 1e-6);
        ++expected_count;
      }
    }
    EXPECT_EQ(count(), expected_count);
  });
}

TYPED_TEST(NestedTransformLifecycleTests, FindIdx)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  auto exec = this->Executor();
  auto in = make_tensor<T>({4});
  auto indices = make_tensor<index_t>({4});
  auto count = make_tensor<int>({});
  test::PreRunLifecycle life;
  const T threshold = T(0.2);
  auto statement = (mtie(indices, count) = find_idx(softmax(cumsum(test::make_prerun_tester(in, life))), GT{threshold}));

  CheckNestedSoftmaxLifecycle(exec, in, life, statement, [&](const auto &expected) {
    int expected_count = 0;
    for (index_t i = 0; i < 4; ++i) {
      if (expected[i] > threshold) {
        EXPECT_EQ(indices(expected_count), i);
        ++expected_count;
      }
    }
    EXPECT_EQ(count(), expected_count);
  });
}

TYPED_TEST(NestedTransformLifecycleTests, ArgReductions)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  auto exec = this->Executor();
  auto in = make_tensor<T>({4});
  auto min_value = make_tensor<T>({});
  auto min_index = make_tensor<index_t>({});
  auto max_value = make_tensor<T>({});
  auto max_index = make_tensor<index_t>({});
  auto check_min = [&](const auto &expected) {
    EXPECT_NEAR(min_value(), expected[0], 1e-6);
    EXPECT_EQ(min_index(), 0);
  };
  auto check_max = [&](const auto &expected) {
    EXPECT_NEAR(max_value(), expected[3], 1e-6);
    EXPECT_EQ(max_index(), 3);
  };

  {
    SCOPED_TRACE("argminmax");
    test::PreRunLifecycle life;
    auto statement = (mtie(min_value, min_index, max_value, max_index) =
                          argminmax(softmax(cumsum(test::make_prerun_tester(in, life)))));
    CheckNestedSoftmaxLifecycle(exec, in, life, statement, [&](const auto &expected) {
      check_min(expected);
      check_max(expected);
    });
  }
  {
    SCOPED_TRACE("argmax");
    test::PreRunLifecycle life;
    auto statement = (mtie(max_value, max_index) = argmax(softmax(cumsum(test::make_prerun_tester(in, life)))));
    CheckNestedSoftmaxLifecycle<false>(exec, in, life, statement, check_max);
  }
  {
    SCOPED_TRACE("argmin");
    test::PreRunLifecycle life;
    auto statement = (mtie(min_value, min_index) = argmin(softmax(cumsum(test::make_prerun_tester(in, life)))));
    CheckNestedSoftmaxLifecycle<false>(exec, in, life, statement, check_min);
  }
}

TYPED_TEST(NestedTransformLifecycleTests, Assignment)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  auto exec = this->Executor();
  auto in = make_tensor<T>({4});
  auto out = make_tensor<T>({4});
  auto check = [&](const auto &expected) {
    for (index_t i = 0; i < 4; ++i) {
      EXPECT_NEAR(out(i), expected[i], 1e-6);
    }
  };

  {
    // Direct transform assignment writes into out without a temporary.
    SCOPED_TRACE("direct");
    test::PreRunLifecycle life;
    auto statement = (out = softmax(cumsum(test::make_prerun_tester(in, life))));
    CheckNestedSoftmaxLifecycle(exec, in, life, statement, check);
  }
  {
    SCOPED_TRACE("expression");
    test::PreRunLifecycle life;
    auto statement = (out = softmax(cumsum(test::make_prerun_tester(in, life))) * T(1));
    CheckNestedSoftmaxLifecycle(exec, in, life, statement, check);
  }
}

// Rank-2 input whose rows are softmaxed along axis 1. Strided inputs are a
// permuted view, so cumsum must materialize logical rows from strided storage.
template <typename T, bool Strided>
class AxisSoftmaxCase {
public:
  AxisSoftmaxCase() : storage_(make_tensor<T>({Strided ? 4 : 2, Strided ? 2 : 4})) {}

  auto Input() {
    if constexpr (Strided) { return storage_.Permute({1, 0}); }
    else { return storage_; }
  }

  // Row r holds (r + 1) * scale, so its softmax(cumsum) matches the rank-1
  // oracle evaluated at that row scale.
  void Fill(T scale) {
    auto in = Input();
    for (index_t r = 0; r < 2; ++r) {
      for (index_t i = 0; i < 4; ++i) {
        in(r, i) = static_cast<T>(r + 1) * scale;
      }
    }
  }

  static cuda::std::array<T, 4> Expected(index_t row, T scale) {
    return ExpectedSoftmaxOfCumsum(static_cast<T>(row + 1) * scale);
  }

private:
  tensor_t<T, 2> storage_;
};

template <typename T, bool Strided, typename Exec>
void CheckAxisSoftmaxLifecycle(Exec &exec)
{
  AxisSoftmaxCase<T, Strided> input;
  auto make = [&](test::PreRunLifecycle &life) {
    return softmax(cumsum(test::make_prerun_tester(input.Input(), life)), {1});
  };
  auto values = make_tensor<T>({8});
  auto count = make_tensor<int>({});
  auto min_values = make_tensor<T>({2});
  auto min_indices = make_tensor<index_t>({2});
  auto max_values = make_tensor<T>({2});
  auto max_indices = make_tensor<index_t>({2});
  auto out = make_tensor<T>({2, 4});
  const T threshold = T(0.2);
  test::PreRunLifecycle find_life, arg_life, direct_life, expression_life;
  auto find_statement = (mtie(values, count) = find(make(find_life), GT{threshold}));
  auto arg_statement = (mtie(min_values, min_indices, max_values, max_indices) = argminmax(make(arg_life), {1}));
  auto direct_statement = (out = make(direct_life));
  auto expression_statement = (out = make(expression_life) * T(1));

  auto check_out = [&](T scale) {
    for (index_t r = 0; r < 2; ++r) {
      const auto expected = input.Expected(r, scale);
      for (index_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(out(r, i), expected[i], 1e-6);
      }
    }
  };

  for (int iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    const T scale = static_cast<T>(iteration + 1);
    input.Fill(scale);
    find_statement.run(exec);
    arg_statement.run(exec);
    exec.sync();

    int expected_count = 0;
    for (index_t r = 0; r < 2; ++r) {
      const auto expected = input.Expected(r, scale);
      for (const T value : expected) {
        if (value > threshold) {
          EXPECT_NEAR(values(expected_count), value, 1e-6);
          ++expected_count;
        }
      }
      EXPECT_NEAR(min_values(r), expected[0], 1e-6);
      EXPECT_EQ(min_indices(r), r * 4);
      EXPECT_NEAR(max_values(r), expected[3], 1e-6);
      EXPECT_EQ(max_indices(r), r * 4 + 3);
    }
    EXPECT_EQ(count(), expected_count);

    direct_statement.run(exec);
    exec.sync();
    check_out(scale);
    expression_statement.run(exec);
    exec.sync();
    check_out(scale);

    test::ExpectLifecycleClean(find_life, "find input", iteration + 1);
    test::ExpectLifecycleClean(arg_life, "argminmax input", iteration + 1);
    test::ExpectLifecycleClean(direct_life, "direct assignment input", iteration + 1);
    test::ExpectLifecycleClean(expression_life, "expression assignment input", iteration + 1);
  }
}

TYPED_TEST(NestedTransformLifecycleTests, AxisSoftmax)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  auto exec = this->Executor();
  CheckAxisSoftmaxLifecycle<T, false>(exec);
}

TYPED_TEST(NestedTransformLifecycleTests, AxisSoftmaxStridedInput)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  auto exec = this->Executor();
  CheckAxisSoftmaxLifecycle<T, true>(exec);
}

// The lifecycle depth is per operator copy: a copy made after preparation
// borrows the original's storage, while a copy made before preparation owns
// its own. These call the hooks directly to pin down that contract.
class LifecycleDepthTests : public ::testing::Test {
protected:
  void SetUp() override { in_.SetVals({1, 2, 3, 4}); }

  template <typename Op>
  static void ExpectCumsum(const Op &op) {
    const float expected[] = {1, 3, 6, 10};
    for (index_t i = 0; i < 4; ++i) {
      EXPECT_EQ(op(i), expected[i]);
    }
  }

  static void ExpectCounts(const test::PreRunLifecycle &life, int prerun, int postrun) {
    EXPECT_EQ(life.prerun_count, prerun);
    EXPECT_EQ(life.postrun_count, postrun);
  }

  SingleThreadedHostExecutor exec_{};
  tensor_t<float, 1> in_ = make_tensor<float>({4}, MATX_HOST_MEMORY);
  detail::NoShape shape_{};
};

TEST_F(LifecycleDepthTests, NestedEntriesOnOneObject)
{
  test::PreRunLifecycle life;
  auto op = cumsum(test::make_prerun_tester(in_, life));

  op.PreRun(shape_, exec_);
  op.PreRun(shape_, exec_);
  ExpectCounts(life, 1, 0);
  op.PostRun(shape_, exec_);
  // The outer scope still owns the prepared result.
  ExpectCounts(life, 1, 0);
  ASSERT_NE(op.Data(), nullptr);
  ExpectCumsum(op);
  op.PostRun(shape_, exec_);
  test::ExpectLifecycleClean(life, "input");
  EXPECT_EQ(op.Data(), nullptr);
}

TEST_F(LifecycleDepthTests, CopiesAfterPreparationBorrowStorage)
{
  test::PreRunLifecycle life;
  auto op = cumsum(test::make_prerun_tester(in_, life));

  op.PreRun(shape_, exec_);
  auto first = op;
  auto second = op;
  first.PreRun(shape_, exec_);
  auto nested = first;
  nested.PreRun(shape_, exec_);
  second.PreRun(shape_, exec_);
  nested.PostRun(shape_, exec_);
  first.PostRun(shape_, exec_);
  second.PostRun(shape_, exec_);

  // Borrowed copies neither prepare nor clean up the input again.
  ExpectCounts(life, 1, 0);
  EXPECT_EQ(first.Data(), op.Data());
  EXPECT_EQ(second.Data(), op.Data());
  ExpectCumsum(op);
  op.PostRun(shape_, exec_);
  test::ExpectLifecycleClean(life, "input");
}

TEST_F(LifecycleDepthTests, CopyBeforePreparationOwnsStorage)
{
  test::PreRunLifecycle life;
  auto op = cumsum(test::make_prerun_tester(in_, life));
  auto copy = op;

  op.PreRun(shape_, exec_);
  copy.PreRun(shape_, exec_);
  ExpectCounts(life, 2, 0);
  EXPECT_NE(copy.Data(), op.Data());
  copy.PostRun(shape_, exec_);
  // Releasing the independent copy leaves the original prepared.
  ExpectCounts(life, 2, 1);
  ExpectCumsum(op);
  op.PostRun(shape_, exec_);
  test::ExpectLifecycleClean(life, "input", 2);
}

TEST_F(LifecycleDepthTests, UnmatchedPostRunChangesNothing)
{
  test::PreRunLifecycle life;
  auto op = cumsum(test::make_prerun_tester(in_, life));

  // Debug builds log this mismatch; it must not release or forward cleanup.
  op.PostRun(shape_, exec_);
  ExpectCounts(life, 0, 0);

  op.PreRun(shape_, exec_);
  ExpectCumsum(op);
  op.PostRun(shape_, exec_);
  test::ExpectLifecycleClean(life, "input");
}

TEST_F(LifecycleDepthTests, MaximumNestingDepthIsBalanced)
{
  test::PreRunLifecycle life;
  auto op = cumsum(test::make_prerun_tester(in_, life));
  constexpr int max_depth = cuda::std::numeric_limits<uint8_t>::max();

  for (int i = 0; i < max_depth; ++i) {
    op.PreRun(shape_, exec_);
  }
  ExpectCounts(life, 1, 0);
  ExpectCumsum(op);
  for (int i = 0; i < max_depth; ++i) {
    op.PostRun(shape_, exec_);
  }
  test::ExpectLifecycleClean(life, "input");
  EXPECT_EQ(op.Data(), nullptr);
}

TEST_F(LifecycleDepthTests, NestingDepthOverflowAborts)
{
  const auto previous_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  auto op = cumsum(in_);
  // The depth counter must not wrap; the noexcept hook terminates instead.
  EXPECT_EXIT({
    for (int i = 0; i <= cuda::std::numeric_limits<uint8_t>::max(); ++i) {
      op.PreRun(shape_, exec_);
    }
  }, testing::KilledBySignal(SIGABRT), "");
  GTEST_FLAG_SET(death_test_style, previous_style);
}

TEST_F(LifecycleDepthTests, RepeatedOperand)
{
  test::PreRunLifecycle life;
  auto op = cumsum(test::make_prerun_tester(in_, life));
  auto out = make_tensor<float>({4}, MATX_HOST_MEMORY);
  // Each operand is a separate copy taken before preparation.
  auto statement = (out = op + op);

  for (int iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    const float scale = static_cast<float>(iteration + 1);
    in_.SetVals({scale, 2 * scale, 3 * scale, 4 * scale});
    statement.run(exec_);

    test::ExpectLifecycleClean(life, "input", 2 * (iteration + 1));
    const float expected[] = {2, 6, 12, 20};
    for (index_t i = 0; i < 4; ++i) {
      EXPECT_EQ(out(i), expected[i] * scale);
    }
  }
  EXPECT_EQ(op.Data(), nullptr);
}

using SelectionTransformTypesWithoutJIT = TupleToTypes<TypedCartesianProduct<
    MatXFloatTuple, ExecutorTypesAllWithoutJIT>::type>::type;

template <typename T>
class SelectionTransformTests : public ::testing::Test {};

TYPED_TEST_SUITE(SelectionTransformTests, SelectionTransformTypesWithoutJIT);

TYPED_TEST(SelectionTransformTests, RankTwoMaterializedTransformInput)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  using Exec = cuda::std::tuple_element_t<1, TypeParam>;
  auto exec = MakeLifecycleExecutor<Exec>();
  // The transform must materialize logical rows from this strided input.
  auto in = make_tensor<T>({3, 2}).Permute({1, 0});
  auto values = make_tensor<T>({6});
  auto indices = make_tensor<index_t>({6});
  auto value_count = make_tensor<int>({});
  auto index_count = make_tensor<int>({});
  auto make_value = [](int real, [[maybe_unused]] int imag) {
    if constexpr (is_complex_v<T>) {
      using Scalar = typename T::value_type;
      return T(static_cast<Scalar>(real), static_cast<Scalar>(imag));
    }
    else { return T(real); }
  };
  test::PreRunLifecycle life;
  auto input = MakeLifecycleInput<true>(in, life);
  // Equal real parts with different imaginary parts make the complex
  // selection depend on both components of the materialized values.
  const auto predicate = NEQ{make_value(3, 1)};
  auto find_values = (mtie(values, value_count) = find(input, predicate));
  auto find_indices = (mtie(indices, index_count) = find_idx(input, predicate));

  for (int iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    const int scale = iteration + 1;
    const cuda::std::array<T, 6> source{
        make_value(scale, 2 * scale), make_value(2 * scale, -scale), make_value(3 * scale, 2 * scale),
        make_value(3 * scale, 2 * scale), make_value(-2 * scale, -scale), make_value(3 * scale, 3 * scale)};
    const cuda::std::array<T, 6> expected{
        make_value(scale, 2 * scale), make_value(3 * scale, scale), make_value(6 * scale, 3 * scale),
        make_value(3 * scale, 2 * scale), make_value(scale, scale), make_value(4 * scale, 4 * scale)};
    for (index_t i = 0; i < 6; ++i) {
      in(i / 3, i % 3) = source[i];
    }

    find_values.run(exec);
    find_indices.run(exec);
    exec.sync();

    test::ExpectLifecycleClean(life, "rank-2 selection input", 2 * (iteration + 1));
    index_t count = 0;
    for (index_t i = 0; i < 6; ++i) {
      if (predicate(expected[i])) {
        EXPECT_EQ(values(count), expected[i]);
        EXPECT_EQ(indices(count), i);
        ++count;
      }
    }
    EXPECT_EQ(value_count(), count);
    EXPECT_EQ(index_count(), count);
  }
}

TYPED_TEST(ReductionLifecycleTests, FindIdxExpressionInput)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  using Exec = cuda::std::tuple_element_t<1, TypeParam>;
  auto exec = MakeLifecycleExecutor<Exec>();
  auto in = make_tensor<T>({2, 3});
  auto out = make_tensor<index_t>({6});
  auto count = make_tensor<int>({});
  in.SetVals({{T(0), T(1), T(2)}, {T(3), T(4), T(5)}});
  auto input = in * T(2) + T(1);

  // Exercise partial, empty, and complete selections of a rank-2 expression.
  for (T threshold : {T(4), T(100), T(0)}) {
    SCOPED_TRACE(threshold);
    (mtie(out, count) = find_idx(input, GT{threshold})).run(exec);
    exec.sync();

    index_t expected_count = 0;
    for (index_t i = 0; i < 6; ++i) {
      if (in(i / 3, i % 3) * T(2) + T(1) > threshold) {
        EXPECT_EQ(out(expected_count), i);
        ++expected_count;
      }
    }
    EXPECT_EQ(count(), expected_count);
  }
}

TYPED_TEST(ReductionLifecycleTests, FindTensorLayouts)
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  using Exec = cuda::std::tuple_element_t<1, TypeParam>;
  auto exec = MakeLifecycleExecutor<Exec>();
  auto in = make_tensor<T>({2, 3});
  auto values = make_tensor<T>({6});
  auto indices = make_tensor<index_t>({6});
  auto value_count = make_tensor<int>({});
  auto index_count = make_tensor<int>({});
  in.SetVals({{T(1), T(2), T(3)}, {T(4), T(5), T(6)}});

  auto check = [&](const auto &input, const cuda::std::array<T, 6> &expected) {
    for (T threshold : {T(2), T(100), T(0)}) {
      SCOPED_TRACE(threshold);
      (mtie(values, value_count) = find(input, GT{threshold})).run(exec);
      (mtie(indices, index_count) = find_idx(input, GT{threshold})).run(exec);
      exec.sync();

      index_t count = 0;
      for (index_t i = 0; i < 6; ++i) {
        if (expected[i] > threshold) {
          EXPECT_EQ(values(count), expected[i]);
          EXPECT_EQ(indices(count), i);
          ++count;
        }
      }
      EXPECT_EQ(value_count(), count);
      EXPECT_EQ(index_count(), count);
    }
  };

  check(in, {T(1), T(2), T(3), T(4), T(5), T(6)});
  // Selection follows logical row-major order, not the backing buffer's order.
  check(in.Permute({1, 0}), {T(1), T(4), T(2), T(5), T(3), T(6)});
}

} // namespace
