#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, FMod)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  // example-begin fmod-test-1
  auto tiv0 = make_tensor<TestType>({});
  auto tiv1 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  tiv0() = (TestType)5.0;
  tiv1() = (TestType)3.1;
  (tov0 = fmod(tiv0, tiv1)).run(exec);
  // example-end fmod-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_fmod((TestType)5.0, (TestType)3.1)));

  MATX_EXIT_HANDLER();
} 