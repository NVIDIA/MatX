#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecs, PolyVal)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  auto pb = std::make_unique<detail::MatXPybind>();
  pb->InitAndRunTVGenerator<TestType>("00_operators", "polyval_operator", "run", {4, 100});

  ExecType exec{};
  auto x = make_tensor<TestType>({100});
  auto c = make_tensor<TestType>({4});
  auto out = make_tensor<TestType>({100});

  pb->NumpyToTensorView(x, "x");
  pb->NumpyToTensorView(c, "c");

  // example-begin polyval-test-1
  (out = polyval(x, c)).run(exec);
  // example-end polyval-test-1
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, out, "out", 0.01);

  MATX_EXIT_HANDLER();
}