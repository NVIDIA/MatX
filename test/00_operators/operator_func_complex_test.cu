#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsComplexTypesAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   

  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  // example-begin exp-test-1
  (tov0 = exp(tiv0)).run(exec);
  // example-end exp-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_exp(c)));

  // example-begin conj-test-1
  (tov0 = conj(tiv0)).run(exec);
  // example-end conj-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_conj(c)));

  // abs takes a complex and output a floating point value
  auto tdd0 = make_tensor<typename TestType::value_type>({});

  // example-begin abs-test-1
  (tdd0 = abs(tiv0)).run(exec);
  // example-end abs-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tdd0(), detail::scalar_internal_abs(c)));

  MATX_EXIT_HANDLER();
}

