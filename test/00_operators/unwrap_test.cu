#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecs, Unwrap)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  auto pb = std::make_unique<detail::MatXPybind>();
  pb->InitAndRunTVGenerator<TestType>("00_operators", "unwrap_operator", "run", {37, 11, 17});

  ExecType exec{};
  auto in1 = make_tensor<TestType>({37});
  auto in2 = make_tensor<TestType>({11, 17});
  auto out1_default = make_tensor<TestType>({37});
  auto out1_period = make_tensor<TestType>({37});
  auto out2_axis1 = make_tensor<TestType>({11, 17});
  auto out2_axis0 = make_tensor<TestType>({11, 17});

  pb->NumpyToTensorView(in1, "in1");
  pb->NumpyToTensorView(in2, "in2");

  // example-begin unwrap-test-1
  (out1_default = unwrap(in1)).run(exec);
  // example-end unwrap-test-1
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, out1_default, "out1_default", 0.01);

  (out1_period = unwrap(in1, -1, static_cast<TestType>(2.5), static_cast<TestType>(4.0))).run(exec);
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, out1_period, "out1_period", 0.01);

  (out2_axis1 = unwrap(in2, 1)).run(exec);
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, out2_axis1, "out2_axis1", 0.01);

  // discont < period/2 should behave like discont == period/2.
  (out2_axis0 = unwrap(in2, 0, static_cast<TestType>(1.0), static_cast<TestType>(6.0))).run(exec);
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, out2_axis0, "out2_axis0", 0.01);

  MATX_EXIT_HANDLER();
}
