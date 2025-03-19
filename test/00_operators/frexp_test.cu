#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecs, Frexp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  // example-begin frexp-test-1
  // Input data
  auto tiv0  = make_tensor<TestType>({10});

  // Output fractional/integer parts
  auto tofrac = make_tensor<TestType>({10});
  auto toint  = make_tensor<int>({10});

  (tiv0 = random<TestType>(tiv0.Shape(), NORMAL)).run(exec);
  // Create operators representing fractional and integer
  const auto [ofrac, oint] = frexp(tiv0);
  (tofrac = ofrac, toint = oint).run(exec);
  // example-end frexp-test-1

  exec.sync();

  int texp;  
  for (int i = 0; i < tiv0.Size(0); i++) {
    if constexpr (std::is_same_v<TypeParam, float>) {
      float tfrac = cuda::std::frexpf(tiv0(i), &texp);
      ASSERT_EQ(tfrac, tofrac(i)); 
      ASSERT_EQ(texp,  toint(i)); 
    }
    else {
      double tfrac = cuda::std::frexp(tiv0(i), &texp);
      ASSERT_EQ(tfrac, tofrac(i)); 
      ASSERT_EQ(texp,  toint(i));   
    }
  }

  MATX_EXIT_HANDLER();
} 