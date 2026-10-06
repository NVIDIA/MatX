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

template <typename TypeParam, bool Direct = false, bool Batched = false>
void CheckArgMinMaxLifecycle()
{
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  using Exec = cuda::std::tuple_element_t<1, TypeParam>;
  auto exec = MakeLifecycleExecutor<Exec>();
  // Also exercise the parallel host scan/reduction paths, which use serial
  // fallbacks for the four-element inputs in the other cases.
  constexpr bool parallel_case = Direct && !Batched &&
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
  auto reduction = [&] {
    if constexpr (Batched) { return argminmax(input, {1}); }
    else { return argminmax(input); }
  }();
  auto statement = (mtie(min_values, min_indices, max_values, max_indices) = reduction);
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

    test::ExpectLifecycleClean(life, "argminmax input", iteration + 1);
    for (index_t b = 0; b < batches; ++b) {
      const T row_scale = static_cast<T>(b + 1) * scale;
      EXPECT_EQ(at(min_values, b), row_scale);
      EXPECT_EQ(at(min_indices, b), b * width);
      EXPECT_EQ(at(max_values, b), static_cast<T>(width) * row_scale);
      EXPECT_EQ(at(max_indices, b), b * width + width - 1);
    }
  }
}

TYPED_TEST(ReductionLifecycleTests, ArgMinMaxTransformInput)
{
  CheckArgMinMaxLifecycle<TypeParam>();
}

TYPED_TEST(ReductionLifecycleTests, ArgMinMaxMaterializedTransformInput)
{
  CheckArgMinMaxLifecycle<TypeParam, true>();
}

TYPED_TEST(ReductionLifecycleTests, BatchedArgMinMaxTransformInput)
{
  CheckArgMinMaxLifecycle<TypeParam, false, true>();
}

TYPED_TEST(ReductionLifecycleTests, BatchedArgMinMaxMaterializedTransformInput)
{
  CheckArgMinMaxLifecycle<TypeParam, true, true>();
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
