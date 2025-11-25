#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsNumericNonComplexAllExecs, AtOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};
  auto t2 = make_tensor<TestType>({2,10});

  // example-begin at-test-1
  auto t1 = make_tensor<TestType>({10});
  auto t0 = make_tensor<TestType>({});

  t1.SetVals({10, 20, 30, 40, 50, 60, 70, 80, 90, 100});
  (t2 = t1).run(exec);

  // Select the fourth element from `t1` as part of the execution. Value should match 
  // `t1(3)` after execution
  (t0 = at(t1, 3)).run(exec);
  // example-end at-test-1
  exec.sync();

  ASSERT_EQ(t0(), t1(3));

  (t0 = at(t2, 1, 4)).run(exec);
  exec.sync();

  ASSERT_EQ(t0(), t2(1, 4));  

  if constexpr (is_cuda_non_jit_executor_v<ExecType> && (std::is_same_v<TestType, float> || std::is_same_v<TestType, double>)) {
    using ComplexType = detail::complex_from_scalar_t<TestType>;
    auto c0 = make_tensor<ComplexType>({});
    (c0 = at(fft(t1), 0)).run(exec);
    exec.sync();

    // The first component of the FFT output (DC) is the sum of all elements, so
    // 10+20+...+100 = 550. The imaginary component should be 0.
    ASSERT_NEAR(c0().real(), static_cast<TestType>(550.0), static_cast<TestType>(1.0e-6));
    ASSERT_NEAR(c0().imag(), static_cast<TestType>(0.0), static_cast<TestType>(1.0e-6));
  }

  MATX_EXIT_HANDLER();
} 