#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecs, R2COp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{}; 
  using ComplexType = detail::complex_from_scalar_t<TestType>;

  // r2c requires FFT support, so we need to check the executor here
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  }

#ifndef MATX_EN_MATHDX  
  if constexpr (is_cuda_jit_executor_v<ExecType>) {
    GTEST_SKIP();
  }
#endif

  const int N1 = 5;
  const int N2 = 6;

  auto t1 = make_tensor<TestType>({N1});
  auto t2 = make_tensor<TestType>({N2});
  auto T1 = make_tensor<ComplexType>({N1});
  auto T2 = make_tensor<ComplexType>({N2});

  for (int i = 0; i < N1; i++) { t1(i) = static_cast<TestType>(i+1); }
  for (int i = 0; i < N2; i++) { t2(i) = static_cast<TestType>(i+1); }
  exec.sync();

  const cuda::std::array<ComplexType, N1> T1_expected = {{
    { 15.0, 0.0 }, { -2.5, static_cast<TestType>(3.4409548) }, { -2.5, static_cast<TestType>(0.81229924) },
    { -2.5, static_cast<TestType>(-0.81229924) }, { -2.5, static_cast<TestType>(-3.4409548) }
  }};

  const cuda::std::array<ComplexType, N2> T2_expected = {{
    { 21.0, 0.0 }, { -3.0, static_cast<TestType>(5.19615242) }, { -3.0, static_cast<TestType>(1.73205081) },
    { -3.0, static_cast<TestType>(-4.44089210e-16) }, { -3.0, static_cast<TestType>(-1.73205081) },
    { -3.0, static_cast<TestType>(-5.19615242) }
  }};

  const TestType thresh = static_cast<TestType>(1.0e-6);

  // Test the regular r2c path with fft() deducing the transform size
  (T1 = r2c(fft(t1), N1)).run(exec);
  (T2 = r2c(fft(t2), N2)).run(exec);

  exec.sync();

  for (int i = 0; i < N1; i++) {
    ASSERT_NEAR(T1(i).real(), T1_expected[i].real(), thresh);
    ASSERT_NEAR(T1(i).imag(), T1_expected[i].imag(), thresh);
  }

  for (int i = 0; i < N2; i++) {
    ASSERT_NEAR(T2(i).real(), T2_expected[i].real(), thresh);
    ASSERT_NEAR(T2(i).imag(), T2_expected[i].imag(), thresh);
  }

  // Test the r2c path when specifying the fft() transform size
  (T1 = r2c(fft(t1, N1), N1)).run(exec);
  (T2 = r2c(fft(t2, N2), N2)).run(exec);

  exec.sync();

  for (int i = 0; i < N1; i++) {
    ASSERT_NEAR(T1(i).real(), T1_expected[i].real(), thresh);
    ASSERT_NEAR(T1(i).imag(), T1_expected[i].imag(), thresh);
  }

  for (int i = 0; i < N2; i++) {
    ASSERT_NEAR(T2(i).real(), T2_expected[i].real(), thresh);
    ASSERT_NEAR(T2(i).imag(), T2_expected[i].imag(), thresh);
  }

  // Add an ifft to the composition to return the original tensor,
  // but now in complex rather than real form. The imaginary components
  // should be ~0.
  (T1 = ifft(r2c(fft(t1), N1))).run(exec);
  (T2 = ifft(r2c(fft(t2), N2))).run(exec);

  exec.sync();

  for (int i = 0; i < N1; i++) {
    ASSERT_NEAR(T1(i).real(), t1(i), thresh);
    ASSERT_NEAR(T1(i).imag(), static_cast<TestType>(0.0), thresh);
  }

  for (int i = 0; i < N2; i++) {
    ASSERT_NEAR(T2(i).real(), t2(i), thresh);
    ASSERT_NEAR(T2(i).imag(), static_cast<TestType>(0.0), thresh);
  }

  MATX_EXIT_HANDLER();
}