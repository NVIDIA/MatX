#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericNonComplexAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};     
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  TestType d = c + 1;

  // example-begin max-el-test-1
  (tov0 = max(tiv0, d)).run(exec);
  // example-end max-el-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), std::max(c, d)));

  // example-begin min-el-test-1
  (tov0 = min(tiv0, d)).run(exec);
  // example-end min-el-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), std::min(c, d)));

  // These operators convert type T into bool
  auto tob = make_tensor<bool>({});

  // example-begin lt-test-1
  (tob = tiv0 < d).run(exec);
  // example-end lt-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c < d));

  // example-begin gt-test-1
  (tob = tiv0 > d).run(exec);
  // example-end gt-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c > d));

  // example-begin lte-test-1
  (tob = tiv0 <= d).run(exec);
  // example-end lte-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c <= d));

  // example-begin gte-test-1
  (tob = tiv0 >= d).run(exec);
  // example-end gte-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c >= d));

  // example-begin eq-test-1
  (tob = tiv0 == d).run(exec);
  // example-end eq-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c == d));

  // example-begin neq-test-1
  (tob = tiv0 != d).run(exec);
  // example-end neq-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c != d));

  MATX_EXIT_HANDLER();
}

