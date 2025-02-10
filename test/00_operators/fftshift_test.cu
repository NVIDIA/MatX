#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;



TYPED_TEST(OperatorTestsFloatNonHalf, FFTShiftWithTransform)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  
  using inner_type = typename inner_op_type_t<TestType>::type;
  using complex_type = detail::complex_from_scalar_t<inner_type>;

  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  }
  
  ExecType exec{};

  [[maybe_unused]] const inner_type thresh = static_cast<inner_type>(1.0e-6);

  // Verify that fftshift1D/ifftshift1D work with nested transforms.
  // These tests are limited to complex-to-complex transforms where we have matched
  // dimensions and types for the inputs/outputs. Adding tests that include real-to-complex
  // or complex-to-real fft compositions is TBD.
  if constexpr (is_complex_v<TestType>)
  {
    const int N1 = 3;
    const int N2 = 4;

    auto t3 = make_tensor<complex_type>({N1});
    auto t4 = make_tensor<complex_type>({N2});
    auto T3 = make_tensor<complex_type>({N1});
    auto T4 = make_tensor<complex_type>({N2});

    const cuda::std::array<complex_type, N1> t3_vals = {{ { 1.0, 0.0 }, { 2.0, 0.0 }, { 3.0, 0.0 } }};
    const cuda::std::array<complex_type, N2> t4_vals = {{ { 1.0, 0.0 }, { 2.0, 0.0 }, { 3.0, 0.0 }, { 4.0, 0.0 } }};

    for (int i = 0; i < N1; i++) { t3(i) = t3_vals[i]; };
    for (int i = 0; i < N2; i++) { t4(i) = t4_vals[i]; };

    exec.sync();

    (T3 = fftshift1D(fft(t3))).run(exec);
    (T4 = fftshift1D(fft(t4))).run(exec);

    exec.sync();

    const cuda::std::array<complex_type, N1> T3_expected = {{
      { -1.5, static_cast<inner_type>(-0.8660254) }, { 6.0, 0.0 }, { -1.5, static_cast<inner_type>(0.8660254) }
    }};
    const cuda::std::array<complex_type, N2> T4_expected = {{
      { -2.0, 0.0 }, { -2.0, -2.0 }, { 10.0, 0.0 }, { -2.0, 2.0 }
    }};

    for (int i = 0; i < N1; i++) {
      ASSERT_NEAR(T3(i).real(), T3_expected[i].real(), thresh);
      ASSERT_NEAR(T3(i).imag(), T3_expected[i].imag(), thresh);
    }

    for (int i = 0; i < N2; i++) {
      ASSERT_NEAR(T4(i).real(), T4_expected[i].real(), thresh);
      ASSERT_NEAR(T4(i).imag(), T4_expected[i].imag(), thresh);
    }

    (T3 = ifftshift1D(fft(t3))).run(exec);
    (T4 = ifftshift1D(fft(t4))).run(exec);

    exec.sync();

    const cuda::std::array<complex_type, N1> T3_ifftshift_expected = {{
      { -1.5, static_cast<inner_type>(0.8660254) }, { -1.5, static_cast<inner_type>(-0.8660254) }, { 6.0, 0.0 }
    }};

    for (int i = 0; i < N1; i++) {
      ASSERT_NEAR(T3(i).real(), T3_ifftshift_expected[i].real(), thresh);
      ASSERT_NEAR(T3(i).imag(), T3_ifftshift_expected[i].imag(), thresh);
    }

    // For even length vectors, fftshift() and ifftshift() are identical
    for (int i = 0; i < N2; i++) {
      ASSERT_NEAR(T4(i).real(), T4_expected[i].real(), thresh);
      ASSERT_NEAR(T4(i).imag(), T4_expected[i].imag(), thresh);
    }
  }

  // Verify that fftshift2D/ifftshift2D work with nested transforms. We do not
  // check correctness here, but there are fftshift2D correctness tests elsewhere.
  if constexpr (is_complex_v<TestType>) {
    [[maybe_unused]] const int N = 4;

    auto x = make_tensor<complex_type>({N,N});
    auto X = make_tensor<complex_type>({N,N});

    (x = static_cast<TestType>(0)).run(exec);

    (X = fftshift2D(fft2(x))).run(exec);
    (X = fftshift2D(ifft2(x))).run(exec);
    (X = ifftshift2D(fft2(x))).run(exec);
    (X = ifftshift2D(ifft2(x))).run(exec);

    exec.sync();
  }

  MATX_EXIT_HANDLER();
}