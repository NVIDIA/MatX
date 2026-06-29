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
