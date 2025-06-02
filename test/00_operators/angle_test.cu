#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsComplexTypesAllExecs, AngleOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<typename TestType::value_type>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  // example-begin angle-test-1
  (tov0 = angle(tiv0)).run(exec);
  // example-end angle-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_angle(c)));  

  MATX_EXIT_HANDLER();
} 
