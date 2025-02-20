#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsComplexTypesAllExecs, RealImagOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<typename TestType::value_type>({});  

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  // example-begin real-test-1
  (tov0 = real(tiv0)).run(exec);
  // example-end real-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c.real()));  

  // example-begin imag-test-1
  (tov0 = imag(tiv0)).run(exec);
  // example-end imag-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c.imag()));   

  MATX_EXIT_HANDLER();
} 