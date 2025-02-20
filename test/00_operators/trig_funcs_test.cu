#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, TrigFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  // example-begin sin-test-1
  (tov0 = sin(tiv0)).run(exec);
  // example-end sin-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_sin(c)));

  // example-begin cos-test-1
  (tov0 = cos(tiv0)).run(exec);
  // example-end cos-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_cos(c)));

  // example-begin tan-test-1
  (tov0 = tan(tiv0)).run(exec);
  // example-end tan-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_tan(c)));

  // example-begin asin-test-1
  (tov0 = asin(tiv0)).run(exec);
  // example-end asin-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_asin(c)));

  // example-begin acos-test-1
  (tov0 = acos(tiv0)).run(exec);
  // example-end acos-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_acos(c)));

  // example-begin atan-test-1
  (tov0 = atan(tiv0)).run(exec);
  // example-end atan-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_atan(c)));

  // example-begin sinh-test-1
  (tov0 = sinh(tiv0)).run(exec);
  // example-end sinh-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_sinh(c)));

  // example-begin cosh-test-1
  (tov0 = cosh(tiv0)).run(exec);
  // example-end cosh-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_cosh(c)));

  // example-begin tanh-test-1
  (tov0 = tanh(tiv0)).run(exec);
  // example-end tanh-test-1  
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_tanh(c)));

  // example-begin asinh-test-1
  (tov0 = asinh(tiv0)).run(exec);
  // example-end asinh-test-1  
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_asinh(c)));

  // example-begin acosh-test-1
  (tov0 = acosh(tiv0)).run(exec);
  // example-end acosh-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_acosh(c)));

  // example-begin atanh-test-1
  (tov0 = atanh(tiv0)).run(exec);
  // example-end atanh-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_atanh(c)));

  MATX_EXIT_HANDLER();
} 