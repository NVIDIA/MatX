#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "prerun_tester.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, Stack)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto t1a = make_tensor<TestType>({5});
  auto t1b = make_tensor<TestType>({5});
  auto t1c = make_tensor<TestType>({5});
 
  auto cop = concat(0, t1a, t1b, t1c);
  
  (cop = (TestType)2).run(exec);
  exec.sync();

  {
    // example-begin stack-test-1
    // Stack 1D operators "t1a", "t1b", and "t1c" together along the first dimension
    auto op = stack(0, t1a, t1b, t1c);
    // example-end stack-test-1
   
    for(int i = 0; i < t1a.Size(0); i++) {
      ASSERT_EQ(op(0,i), t1a(i));
      ASSERT_EQ(op(1,i), t1b(i));
      ASSERT_EQ(op(2,i), t1c(i));
    }
  }  
 
  {
    auto op = stack(1, t1a, t1b, t1c);
    
    for(int i = 0; i < t1a.Size(0); i++) {
      ASSERT_EQ(op(i,0), t1a(i));
      ASSERT_EQ(op(i,1), t1b(i));
      ASSERT_EQ(op(i,2), t1c(i));
    }
  }

  MATX_EXIT_HANDLER();
}

// Stacks three rank-2 tensors along each axis of the rank-3 output on the executor
TYPED_TEST(OperatorTestsNumericAllExecs, StackAllAxes)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  // Small values so they are exact in every numeric type
  using scalar_t = typename inner_op_type_t<TestType>::type;
  auto val = [](int id, index_t i, index_t j) {
    return TestType(static_cast<scalar_t>(id * 8 + static_cast<int>(i * 3 + j)));
  };

  auto a = make_tensor<TestType>({2, 3});
  auto b = make_tensor<TestType>({2, 3});
  auto c = make_tensor<TestType>({2, 3});
  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 3; j++) {
      a(i, j) = val(0, i, j);
      b(i, j) = val(1, i, j);
      c(i, j) = val(2, i, j);
    }
  }

  for (int axis = 0; axis < 3; axis++) {
    cuda::std::array<index_t, 3> shape;
    const index_t in_sizes[2] = {2, 3};
    for (int d = 0, id = 0; d < 3; d++) {
      shape[d] = (d == axis) ? 3 : in_sizes[id++];
    }
    auto out = make_tensor<TestType>(shape);

    (out = stack(axis, a, b, c)).run(exec);
    exec.sync();

    for (index_t i0 = 0; i0 < shape[0]; i0++) {
      for (index_t i1 = 0; i1 < shape[1]; i1++) {
        for (index_t i2 = 0; i2 < shape[2]; i2++) {
          const index_t idx[3] = {i0, i1, i2};
          index_t in_idx[2];
          for (int d = 0, id = 0; d < 3; d++) {
            if (d != axis) in_idx[id++] = idx[d];
          }
          ASSERT_EQ(out(i0, i1, i2), val(static_cast<int>(idx[axis]), in_idx[0], in_idx[1])) << "axis=" << axis;
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// Verifies that stack() correctly forwards PreRun()/PostRun() to its operands
// when a stack expression is materialized via run(). Each variadic operand is
// wrapped in a PreRunTesterOp lifecycle probe: stack()'s PreRun/PostRun fold
// must forward to every operand exactly once (an unforwarded operand leaves
// prerun_count == 0). The probe is a transparent pass-through, so the
// materialized result is still the correct stacked output and is checked against
// a reference. The cumsum() operands are real transforms whose temporaries are
// only allocated/filled if PreRun is forwarded.
TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecsWithoutJIT, StackOperatorInput)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({5});
  auto b = make_tensor<TestType>({5});
  auto c = make_tensor<TestType>({5});
  a.SetVals({1, 2, 3, 4, 5});       // cumsum(a) = {1, 3, 6, 10, 15}
  b.SetVals({10, 20, 30, 40, 50});  // leaf operand
  c.SetVals({2, 4, 6, 8, 10});      // cumsum(c) = {2, 6, 12, 20, 30}

  // Reference: materialize the transform operands into tensors first.
  auto ca = make_tensor<TestType>({5});
  auto cc = make_tensor<TestType>({5});
  (ca = cumsum(a)).run(exec);
  (cc = cumsum(c)).run(exec);
  auto out_ref = make_tensor<TestType>({3, 5});
  (out_ref = stack(0, ca, b, cc)).run(exec);

  // Under test: wrap each stacked operand in its own lifecycle probe. A mix of
  // two transform operands and a leaf exercises the variadic fold over more than
  // two operands.
  PreRunLifecycle s0, s1, s2;
  auto out_test = make_tensor<TestType>({3, 5});
  (out_test = stack(0,
                    make_prerun_tester(cumsum(a), s0),
                    make_prerun_tester(b, s1),
                    make_prerun_tester(cumsum(c), s2))).run(exec);

  exec.sync();

  // Correctness preserved (probe is a transparent pass-through).
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      ASSERT_EQ(out_test(i, j), out_ref(i, j)) << "mismatch at (" << i << "," << j << ")";
    }
  }

  // Lifecycle: stack() forwarded a balanced PreRun/PostRun to every operand.
  ExpectLifecycleClean(s0, "cumsum(a)");
  ExpectLifecycleClean(s1, "b");
  ExpectLifecycleClean(s2, "cumsum(c)");

  MATX_EXIT_HANDLER();
}

// stack rejects an axis outside [0, Rank()] in every build mode
TEST(OperatorValidationTests, StackInvalidAxis)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto a = make_tensor<TestType>({5});
  auto b = make_tensor<TestType>({5});

  EXPECT_THROW(stack(2, a, b), matx::detail::matxException);
  EXPECT_THROW(stack(-1, a, b), matx::detail::matxException);
  EXPECT_NO_THROW(stack(0, a, b));
  EXPECT_NO_THROW(stack(1, a, b));

  MATX_EXIT_HANDLER();
}

// All operands must have the same shape as the first in every build mode
TEST(OperatorValidationTests, StackMismatchedShapes)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto a = make_tensor<TestType>({2, 3});
  auto b = make_tensor<TestType>({2, 2});
  auto c = make_tensor<TestType>({2, 3});

  EXPECT_THROW(stack(0, a, b), matx::detail::matxException);
  EXPECT_THROW(stack(0, b, a), matx::detail::matxException);
  EXPECT_THROW(stack(2, a, c, b), matx::detail::matxException);
  EXPECT_NO_THROW(stack(0, a, c));

  MATX_EXIT_HANDLER();
}

// Static operators can report a DynRank() different from Rank() (toeplitz reports its
// input's rank), so the rank check must use Rank() for them
TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, StackStaticOpWithDifferentDynRank)
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

  auto out = make_tensor<TestType>({2, 3, 3});
  (out = stack(0, toeplitz(v), a + a)).run(exec);
  exec.sync();

  for (index_t i = 0; i < 3; i++) {
    for (index_t j = 0; j < 3; j++) {
      const index_t d = i > j ? i - j : j - i;
      ASSERT_EQ(out(0, i, j), v(d));
      ASSERT_EQ(out(1, i, j), a(i, j) + a(i, j));
    }
  }

  MATX_EXIT_HANDLER();
}
