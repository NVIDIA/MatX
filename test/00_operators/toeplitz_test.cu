#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatAllExecs, Toeplitz)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  auto pb = std::make_unique<detail::MatXPybind>();
  pb->InitAndRunTVGenerator<TestType>("00_operators", "toeplitz", "run", {5, 5, 6});

  ExecType exec{};
  auto out1 = make_tensor<TestType>({5, 5});
  auto out2 = make_tensor<TestType>({5, 5});
  auto out3 = make_tensor<TestType>({5, 6});
  auto c = make_tensor<TestType>({5});
  auto r = make_tensor<TestType>({5});
  auto r2 = make_tensor<TestType>({6});

  pb->NumpyToTensorView(c, "c");
  pb->NumpyToTensorView(r, "r");
  pb->NumpyToTensorView(r2, "r2");

  // example-begin toeplitz-test-1
  (out1 = toeplitz(c)).run(exec);
  // example-end toeplitz-test-1
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, out1, "out1", 0.01);

  // example-begin toeplitz-test-2
  (out2 = toeplitz(c, r)).run(exec);
  // example-end toeplitz-test-2
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, out2, "out2", 0.01);  

  (out3 = toeplitz(c, r2)).run(exec);
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, out3, "out3", 0.01);  

  MATX_EXIT_HANDLER();
} 