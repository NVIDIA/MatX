#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};    
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  // example-begin log10-test-1
  (tov0 = log10(tiv0)).run(exec);
  // example-end log10-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_log10(c)));

  // example-begin log-test-1
  (tov0 = log(tiv0)).run(exec);
  // example-end log-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_log(c)));

  // example-begin log2-test-1
  (tov0 = log2(tiv0)).run(exec);
  // example-end log2-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_log2(c)));

  // example-begin floor-test-1
  (tov0 = floor(tiv0)).run(exec);
  // example-end floor-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_floor(c)));

  // example-begin ceil-test-1
  (tov0 = ceil(tiv0)).run(exec);
  // example-end ceil-test-1  
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_ceil(c)));

  // example-begin round-test-1
  (tov0 = round(tiv0)).run(exec);
  // example-end round-test-1  
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_round(c)));

  // example-begin sqrt-test-1
  (tov0 = sqrt(tiv0)).run(exec);
  // example-end sqrt-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_sqrt(c)));      

  // example-begin rsqrt-test-1
  (tov0 = rsqrt(tiv0)).run(exec);
  // example-end rsqrt-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_rsqrt(c)));   

  MATX_EXIT_HANDLER();
}

