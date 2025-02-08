#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Cross)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};   
  auto pb = std::make_unique<detail::MatXPybind>();
  // Half precision needs a bit more tolerance when compared to fp32
  float thresh = 0.01f;
  if constexpr (is_matx_half_v<TestType>) {
    thresh = 0.08f;
  }

  {//batched 4 x 3
    pb->InitAndRunTVGenerator<TestType>("00_operators", "cross_operator", "run", {4, 3});

    auto a = make_tensor<TestType>({4, 3});
    auto b = make_tensor<TestType>({4, 3});
    auto out = make_tensor<TestType>({4, 3});

    pb->NumpyToTensorView(a, "a");
    pb->NumpyToTensorView(b, "b");

    // example-begin cross-test-1
    (out = cross(a, b)).run(exec);
    // example-end cross-test-1
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, out, "out", thresh);
  }

  {//non-batched 3
    pb->InitAndRunTVGenerator<TestType>("00_operators", "cross_operator", "run", {3});
    auto a = make_tensor<TestType>({3});
    auto b = make_tensor<TestType>({3});
    auto out = make_tensor<TestType>({3});

    pb->NumpyToTensorView(a, "a");
    pb->NumpyToTensorView(b, "b");
    
    (out = cross(a, b)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, out, "out", thresh);
  }

  {//non-batched 2
    pb->InitAndRunTVGenerator<TestType>("00_operators", "cross_operator", "run", {2});
    auto a = make_tensor<TestType>({2});
    auto b = make_tensor<TestType>({2});
    auto out = make_tensor<TestType>({1});

    pb->NumpyToTensorView(a, "a");
    pb->NumpyToTensorView(b, "b");
    
    (out = cross(a, b)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, out, "out", thresh);
  }

  MATX_EXIT_HANDLER();
} 