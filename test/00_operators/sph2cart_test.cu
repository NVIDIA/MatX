#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Sphere2Cart)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  float thresh = 0.01f;
  if constexpr (is_matx_half_v<TestType>) {
    thresh = 0.35f;
  }  

  ExecType exec{}; 

  int n = 5;

  auto xi = range<0>({n},(TestType)1,(TestType)1);
  auto yi = range<0>({n},(TestType)1,(TestType)1);
  auto zi = range<0>({n},(TestType)1,(TestType)1);

  // example-begin cart2sph-test-1
  auto [theta, phi, r] = cart2sph(xi, yi, zi);
  // example-end cart2sph-test-1

  // example-begin sph2cart-test-1
  auto [x, y, z] = sph2cart(theta, phi, r);
  // example-end sph2cart-test-1

  for(int i=0; i<n; i++) {
    ASSERT_NEAR(xi(i), x(i), thresh);
  }

  MATX_EXIT_HANDLER();
}