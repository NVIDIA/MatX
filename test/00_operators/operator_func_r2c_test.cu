#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, OperatorFuncsR2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<typename detail::complex_from_scalar_t<TestType>>({});
  // example-begin expj-test-1
  // TestType is float, double, bf16, etc.
  tiv0() = static_cast<TestType>(M_PI/2.0);
  (tov0 = expj(tiv0)).run(exec);
  // tov0 is complex with value 0 + 1j
  // example-end expj-test-1

  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::complex_from_scalar_t<TestType>(0.0, 1.0)));

  tiv0() = static_cast<TestType>(-1.0 * M_PI);
  (tov0 = expj(tiv0)).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::complex_from_scalar_t<TestType>(-1.0, 0.0)));

  tiv0() = 0;
  (tov0 = expj(tiv0)).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::complex_from_scalar_t<TestType>(1.0, 0.0)));

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  (tov0 = expj(tiv0)).run(exec);
  exec.sync();

  EXPECT_TRUE(MatXUtils::MatXTypeCompare(
      tov0(),
      typename detail::complex_from_scalar_t<TestType>(detail::scalar_internal_cos(tiv0()), detail::scalar_internal_sin(tiv0()))));  
  MATX_EXIT_HANDLER();      
}

