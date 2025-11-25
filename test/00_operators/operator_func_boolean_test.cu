#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsBooleanAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  TestType d = false;
  tiv0() = c;

  // example-begin land-test-1
  (tov0 = tiv0 && d).run(exec);
  // example-end land-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c && d));

  // example-begin lor-test-1
  (tov0 = tiv0 || d).run(exec);
  // example-end lor-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c || d));

  // example-begin lnot-test-1
  (tov0 = !tiv0).run(exec);
  // example-end lnot-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), !c));

  // example-begin xor-test-1
  (tov0 = tiv0 ^ d).run(exec);
  // example-end xor-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c ^ d));

  // example-begin or-test-1
  (tov0 = tiv0 | d).run(exec);
  // example-end or-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c | d));

  // example-begin and-test-1
  (tov0 = tiv0 & d).run(exec);
  // example-end and-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c & d));    

  MATX_EXIT_HANDLER();
}

