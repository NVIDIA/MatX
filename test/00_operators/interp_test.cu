#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Interp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  // example-begin interp-test-1
  auto x = make_tensor<TestType>({5});
  x.SetVals({0.0, 1.0, 3.0, 3.5, 4.0});

  auto v = make_tensor<TestType>({5});
  v.SetVals({0.0, 2.0, 1.0, 3.0, 4.0});

  auto xq = make_tensor<TestType>({6});
  xq.SetVals({-1.0, 0.0, 0.25, 1.0, 1.5, 5.0});

  auto out_linear = make_tensor<TestType>({xq.Size(0)});
  (out_linear = interp(x, v, xq)).run(exec);
  // example-end interp-test-1
  exec.sync();

  ASSERT_EQ(out_linear(0), 0.0);
  ASSERT_EQ(out_linear(1), 0.0);
  ASSERT_EQ(out_linear(2), 0.5);
  ASSERT_EQ(out_linear(3), 2.0);
  ASSERT_EQ(out_linear(4), 1.75);
  ASSERT_EQ(out_linear(5), 4.0);

  // example-begin interp-test-2
  auto out_nearest = make_tensor<TestType>({xq.Size(0)});
  (out_nearest = interp<InterpMethodNearest>(x, v, xq)).run(exec);
  // example-end interp-test-2
  exec.sync();

  ASSERT_EQ(out_nearest(0), 0.0);
  ASSERT_EQ(out_nearest(1), 0.0);
  ASSERT_EQ(out_nearest(2), 0.0);
  ASSERT_EQ(out_nearest(3), 2.0);
  ASSERT_EQ(out_nearest(4), 2.0);
  ASSERT_EQ(out_nearest(5), 4.0);

  auto out_next = make_tensor<TestType>({xq.Size(0)});
  (out_next = interp<InterpMethodNext>(x, v, xq)).run(exec);
  exec.sync();

  ASSERT_EQ(out_next(0), 0.0);
  ASSERT_EQ(out_next(1), 0.0);
  ASSERT_EQ(out_next(2), 2.0);
  ASSERT_EQ(out_next(3), 2.0);
  ASSERT_EQ(out_next(4), 1.0);
  ASSERT_EQ(out_next(5), 4.0);

  auto out_prev = make_tensor<TestType>({xq.Size(0)});
  (out_prev = interp<InterpMethodPrev>(x, v, xq)).run(exec);
  exec.sync();

  ASSERT_EQ(out_prev(0), 0.0);
  ASSERT_EQ(out_prev(1), 0.0);
  ASSERT_EQ(out_prev(2), 0.0);
  ASSERT_EQ(out_prev(3), 2.0);
  ASSERT_EQ(out_prev(4), 2.0);
  ASSERT_EQ(out_prev(5), 4.0);


  auto out_spline = make_tensor<TestType>({xq.Size(0)});
  (out_spline = interp<InterpMethodSpline>(x, v, xq)).run(exec);
  exec.sync();

  ASSERT_EQ(out_spline(5), 7.0);


  MATX_EXIT_HANDLER();
}
