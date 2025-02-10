#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, IsNanInf)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};    

  auto nan = make_tensor<TestType>({});
  using conversionType = typename matx::detail::value_promote_t<TestType>;  
  if constexpr(matx::is_complex_v<TestType>) {    
    nan() = TestType(std::numeric_limits<conversionType>::quiet_NaN());
  } else {
    nan() = std::numeric_limits<conversionType>::quiet_NaN();
  }
  auto tob = make_tensor<bool>({});
  // example-begin nan-test-1
  (tob = matx::isnan(nan)).run(exec); 
  // example-end nan-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), std::is_floating_point_v<conversionType> ? true : false));

  auto notnanorinf = make_tensor<TestType>({});
  if constexpr(matx::is_complex_v<TestType>) {    
    notnanorinf() = TestType(0);
  } else {
    notnanorinf() = 0;
  }  
  (tob = matx::isnan(notnanorinf)).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), false));

  auto inf = make_tensor<TestType>({});
  if constexpr(matx::is_complex_v<TestType>) {    
    inf() = TestType(std::numeric_limits<conversionType>::infinity());
  } else {
    inf() = std::numeric_limits<conversionType>::infinity();
  }  
  // example-begin inf-test-1
  (tob = matx::isinf(inf)).run(exec); 
  // example-end inf-test-1
  exec.sync(); 
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), std::is_floating_point_v<conversionType> ? true : false));

  (tob = matx::isinf(notnanorinf)).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), false));  


  MATX_EXIT_HANDLER();
} 