#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsComplexTypesAllExecs, OperatorFuncDivComplex)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};  
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});
  typename TestType::value_type s = 5.0;

  TestType c = GenerateData<TestType>();  
  tiv0() = c;

  (tov0 = s / tiv0).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), s / tiv0()));

  MATX_EXIT_HANDLER();  
}

