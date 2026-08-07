#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "prerun_tester.h"

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

// Verify polyval forwards PreRun/PostRun to both of its operands (input values
// and coefficients) by wrapping each in a lifecycle probe.
TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecsWithoutJIT, PolyValOperatorCoeffs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  constexpr int N = 5;
  constexpr int NC = 4;

  auto x = make_tensor<TestType>({N});
  auto c = make_tensor<TestType>({NC});
  x.SetVals({0.5, 1.0, 1.5, 2.0, 2.5});
  c.SetVals({1, 2, 3, 4});

  // Reference from the raw operands.
  auto out_ref = make_tensor<TestType>({N});
  (out_ref = polyval(x, c)).run(exec);

  // Wrap both operands in lifecycle probes.
  PreRunLifecycle sx, sc;
  auto out_test = make_tensor<TestType>({N});
  (out_test = polyval(make_prerun_tester(x, sx), make_prerun_tester(c, sc))).run(exec);
  exec.sync();

  ExpectLifecycleClean(sx, "input");
  ExpectLifecycleClean(sc, "coeffs");

  for (int i = 0; i < N; i++) {
    ASSERT_NEAR(static_cast<double>(out_test(i)), static_cast<double>(out_ref(i)),
                1e-4) << "mismatch at index " << i;
  }

  MATX_EXIT_HANDLER();
}
