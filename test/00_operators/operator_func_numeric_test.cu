#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};    
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  // example-begin add-test-1
  (tov0 = tiv0 + tiv0).run(exec);
  // example-end add-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c + c));

  // example-begin sub-test-1
  (tov0 = tiv0 - tiv0).run(exec);
  // example-end sub-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c - c));

  // example-begin mul-test-1
  (tov0 = tiv0 * tiv0).run(exec);
  // example-end mul-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c * c));

  // example-begin div-test-1
  (tov0 = tiv0 / tiv0).run(exec);
  // example-end div-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c / c));

  // example-begin neg-test-1
  (tov0 = -tiv0).run(exec);
  // example-end neg-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), -c));

  // example-begin IF-test-1
  IF(tiv0 == tiv0, tov0 = c).run(exec);
  // example-end IF-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c));

  TestType p = 2.0f;
  // example-begin pow-test-1
  (tov0 = as_type<TestType>(pow(tiv0, p))).run(exec);
  // example-end pow-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_pow(c, p)));

  TestType three = 3.0f;

  (tov0 = tiv0 * tiv0 * (tiv0 + tiv0) / tiv0 + three).run(exec);
  exec.sync();

  TestType res;
  res = c * c * (c + c) / c + three;
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), res, 0.07));


  MATX_EXIT_HANDLER();
}

