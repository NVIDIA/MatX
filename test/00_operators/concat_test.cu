#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;



TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Concatenate)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};
  index_t i, j;

  // example-begin concat-test-1
  auto t11 = make_tensor<TestType>({10});
  auto t12 = make_tensor<TestType>({5});
  auto t1o = make_tensor<TestType>({15});

  t11.SetVals({0,1,2,3,4,5,6,7,8,9});
  t12.SetVals({0,1,2,3,4});

  // Concatenate "t11" and "t12" into a new 1D tensor
  (t1o = concat(0, t11, t12)).run(exec);
  // example-end concat-test-1
  exec.sync();

  for (i = 0; i < t11.Size(0) + t12.Size(0); i++) {
    if (i < t11.Size(0)) {
      ASSERT_EQ(t11(i), t1o(i));
    }
    else {
      ASSERT_EQ(t12(i - t11.Size(0)), t1o(i));
    }
  }

  // Test contcat with nested transforms
  if constexpr (is_cuda_non_jit_executor<ExecType> && (cuda::std::is_same_v<TestType, float> || cuda::std::is_same_v<TestType, double>)) {
    auto delta = make_tensor<TestType>({1});
    delta.SetVals({static_cast<TestType>(1)});

    (t1o = static_cast<TestType>(0)).run(exec);
    (t1o = concat(0, conv1d(t11, delta, MATX_C_MODE_SAME), conv1d(t12, delta, MATX_C_MODE_SAME))).run(exec);

    exec.sync();

    for (i = 0; i < t11.Size(0) + t12.Size(0); i++) {
      if (i < t11.Size(0)) {
        ASSERT_EQ(t11(i), t1o(i));
      }
      else {
        ASSERT_EQ(t12(i - t11.Size(0)), t1o(i));
      }
    }
  }

  // 2D tensors
  auto t21 = make_tensor<TestType>({4, 4});
  auto t22 = make_tensor<TestType>({3, 4});
  auto t23 = make_tensor<TestType>({4, 3});

  auto t2o1 = make_tensor<TestType>({7,4});
  auto t2o2 = make_tensor<TestType>({4,7});
  t21.SetVals({{1,2,3,4},
               {2,3,4,5},
               {3,4,5,6},
               {4,5,6,7}} );
  t22.SetVals({{5,6,7,8},
               {6,7,8,9},
               {9,10,11,12}});
  t23.SetVals({{5,6,7},
               {6,7,8},
               {9,10,11},
               {10,11,12}});

  (t2o1 = concat(0, t21, t22)).run(exec);
  exec.sync();

  for (i = 0; i < t21.Size(0) + t22.Size(0); i++) {
    for (j = 0; j < t21.Size(1); j++) {
      if (i < t21.Size(0)) {
        ASSERT_EQ(t21(i,j), t2o1(i,j));
      }
      else {
        ASSERT_EQ(t22(i - t21.Size(0), j), t2o1(i,j));
      }
    }
  }

  (t2o2 = concat(1, t21, t23)).run(exec);
  exec.sync();

  for (j = 0; j < t21.Size(1) + t23.Size(1); j++) {
    for (i = 0; i < t21.Size(0); i++) {
      if (j < t21.Size(1)) {
        ASSERT_EQ(t21(i,j), t2o2(i,j));
      }
      else {
        ASSERT_EQ(t23(i, j - t21.Size(1)), t2o2(i,j));
      }
    }
  }

  auto t1o1 = make_tensor<TestType>({30});

  // Concatenating 3 tensors
  (t1o1 = concat(0, t11, t11, t11)).run(exec);
  exec.sync();

  for (i = 0; i < t1o1.Size(0); i++) {
    ASSERT_EQ(t1o1(i), t11(i % t11.Size(0)));
  }


  // Multiple concatenations
  {
    auto a = matx::make_tensor<float>({10});
    auto b = matx::make_tensor<float>({10});
    auto c = matx::make_tensor<float>({10});
    auto d = matx::make_tensor<float>({10});

    auto result = matx::make_tensor<float>({40});
    a.SetVals({1,2,3,4,5,6,7,8,9,10});
    b.SetVals({11,12,13,14,15,16,17,18,19,20});
    c.SetVals({21,22,23,24,25,26,27,28,29,30});
    d.SetVals({31,32,33,34,35,36,37,38,39,40});

    auto tempConcat1 = matx::concat(0, a, b);
    auto tempConcat2 = matx::concat(0, c, d);
    (result = matx::concat(0, tempConcat1, tempConcat2 )).run(exec);

    exec.sync();
    for (int cnt = 0; cnt < result.Size(0); cnt++) {
      ASSERT_EQ(result(cnt), cnt + 1);
    }
  }

  MATX_EXIT_HANDLER();
}

// Concatenates three rank-3 operators along each axis, reading and writing through concat
TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, ConcatAllAxes)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  // Values stay below 256 so they are exact in every tested type
  auto val = [](int id, index_t i, index_t j, index_t k) {
    return TestType(static_cast<float>(id * 64 + i * 16 + j * 4 + k));
  };
  const index_t axis_sizes[3] = {2, 3, 1};
  const index_t other_sizes[2] = {3, 4};

  for (int axis = 0; axis < 3; axis++) {
    auto shape_for = [&](index_t axis_size) {
      cuda::std::array<index_t, 3> s;
      for (int d = 0, od = 0; d < 3; d++) {
        s[d] = (d == axis) ? axis_size : other_sizes[od++];
      }
      return s;
    };
    auto a = make_tensor<TestType>(shape_for(axis_sizes[0]));
    auto b = make_tensor<TestType>(shape_for(axis_sizes[1]));
    auto c = make_tensor<TestType>(shape_for(axis_sizes[2]));
    auto out = make_tensor<TestType>(shape_for(axis_sizes[0] + axis_sizes[1] + axis_sizes[2]));

    auto fill = [&](auto &t, int id) {
      for (index_t i = 0; i < t.Size(0); i++)
        for (index_t j = 0; j < t.Size(1); j++)
          for (index_t k = 0; k < t.Size(2); k++)
            t(i, j, k) = val(id, i, j, k);
    };
    fill(a, 0);
    fill(b, 1);
    fill(c, 2);

    // Expected: which input an output element came from, and its index in that input
    auto check = [&](auto &o, int id_offset) {
      for (index_t i = 0; i < o.Size(0); i++) {
        for (index_t j = 0; j < o.Size(1); j++) {
          for (index_t k = 0; k < o.Size(2); k++) {
            index_t idx[3] = {i, j, k};
            int id = 0;
            while (idx[axis] >= axis_sizes[id]) {
              idx[axis] -= axis_sizes[id];
              id++;
            }
            ASSERT_EQ(o(i, j, k), val(id + id_offset, idx[0], idx[1], idx[2])) << "axis=" << axis;
          }
        }
      }
    };

    (out = concat(axis, a, b, c)).run(exec);
    exec.sync();
    check(out, 0);

    // Write through concat: shift each input's values to the next id
    (out = out + TestType(64)).run(exec);
    (concat(axis, a, b, c) = out).run(exec);
    exec.sync();
    (out = concat(axis, a, b, c)).run(exec);
    exec.sync();
    check(out, 1);
  }

  MATX_EXIT_HANDLER();
}

// concat rejects an axis outside [0, Rank()) in every build mode
TEST(OperatorValidationTests, ConcatInvalidAxis)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto a = make_tensor<TestType>({2, 3});
  auto b = make_tensor<TestType>({2, 3});

  EXPECT_THROW(concat(2, a, b), matx::detail::matxException);
  EXPECT_THROW(concat(-1, a, b), matx::detail::matxException);
  EXPECT_NO_THROW(concat(0, a, b));
  EXPECT_NO_THROW(concat(1, a, b));

  MATX_EXIT_HANDLER();
}

// Non-axis sizes must match the first operand in every build mode
TEST(OperatorValidationTests, ConcatMismatchedShapes)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto a = make_tensor<TestType>({2, 3});
  auto b = make_tensor<TestType>({2, 2});
  auto c = make_tensor<TestType>({4, 3});

  EXPECT_THROW(concat(0, a, b), matx::detail::matxException);
  EXPECT_THROW(concat(0, b, a), matx::detail::matxException);
  EXPECT_THROW(concat(0, a, c, b), matx::detail::matxException);
  EXPECT_THROW(concat(1, a, c), matx::detail::matxException);
  EXPECT_NO_THROW(concat(1, a, b));
  EXPECT_NO_THROW(concat(0, a, c));

  MATX_EXIT_HANDLER();
}

// Static operators can report a DynRank() different from Rank() (toeplitz reports its
// input's rank), so the rank check must use Rank() for them
TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, ConcatStaticOpWithDifferentDynRank)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto v = make_tensor<TestType>({3});
  auto a = make_tensor<TestType>({3, 3});
  for (index_t i = 0; i < 3; i++) {
    v(i) = static_cast<TestType>(static_cast<float>(i + 1));
    for (index_t j = 0; j < 3; j++) a(i, j) = static_cast<TestType>(static_cast<float>(10 + i * 3 + j));
  }

  auto out = make_tensor<TestType>({6, 3});
  (out = concat(0, toeplitz(v), a + a)).run(exec);
  exec.sync();

  for (index_t i = 0; i < 3; i++) {
    for (index_t j = 0; j < 3; j++) {
      const index_t d = i > j ? i - j : j - i;
      ASSERT_EQ(out(i, j), v(d));
      ASSERT_EQ(out(i + 3, j), a(i, j) + a(i, j));
    }
  }

  MATX_EXIT_HANDLER();
}
