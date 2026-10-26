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
    
    // Skip if using JIT executor but expression doesn't support JIT
    if constexpr (is_cuda_jit_executor_v<ExecType>) {
      auto test_t3 = make_tensor<complex_type>({N1});
      auto test_expr = fftshift1D(fft(test_t3));
      if (!jit_supported(test_expr)) {
        GTEST_SKIP();
      }
    }

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

    // Skip if using JIT executor but expression doesn't support JIT
    if constexpr (is_cuda_jit_executor_v<ExecType>) {
      auto test_x = make_tensor<complex_type>({N,N});
      auto test_expr = fftshift2D(fft2(test_x));
      if (!jit_supported(test_expr)) {
        GTEST_SKIP();
      }
    }

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

TYPED_TEST(OperatorTestsFloatNonHalf, FFTShiftDifferentSizesInOneExpression)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  // Shifts of inputs with different sizes can appear in the same expression. An odd length makes
  // fftshift and ifftshift differ.
  const index_t na = 7;
  const index_t nb = 16;
  auto a = make_tensor<TestType>({na});
  auto b = make_tensor<TestType>({nb});
  auto out = make_tensor<TestType>({na + nb});
  for (index_t i = 0; i < na; i++) {
    a(i) = static_cast<detail::value_promote_t<TestType>>(i);
  }
  for (index_t i = 0; i < nb; i++) {
    b(i) = static_cast<detail::value_promote_t<TestType>>(na + i);
  }

  (out = concat(0, fftshift1D(a), fftshift1D(b))).run(exec);
  exec.sync();
  for (index_t i = 0; i < na; i++) {
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(out(i), a((i + (na + 1) / 2) % na)));
  }
  for (index_t i = 0; i < nb; i++) {
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(out(na + i), b((i + (nb + 1) / 2) % nb)));
  }

  (out = concat(0, ifftshift1D(a), ifftshift1D(b))).run(exec);
  exec.sync();
  for (index_t i = 0; i < na; i++) {
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(out(i), a((i + na / 2) % na)));
  }
  for (index_t i = 0; i < nb; i++) {
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(out(na + i), b((i + nb / 2) % nb)));
  }

  // 2D shifts move both dimensions
  const index_t cols = 4;
  auto c = make_tensor<TestType>({na, cols});
  auto d = make_tensor<TestType>({nb, cols});
  auto out2 = make_tensor<TestType>({na + nb, cols});
  for (index_t i = 0; i < na; i++) {
    for (index_t j = 0; j < cols; j++) {
      c(i, j) = static_cast<detail::value_promote_t<TestType>>(i * cols + j);
    }
  }
  for (index_t i = 0; i < nb; i++) {
    for (index_t j = 0; j < cols; j++) {
      d(i, j) = static_cast<detail::value_promote_t<TestType>>(100 + i * cols + j);
    }
  }

  (out2 = concat(0, fftshift2D(c), fftshift2D(d))).run(exec);
  exec.sync();
  for (index_t j = 0; j < cols; j++) {
    for (index_t i = 0; i < na; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(out2(i, j), c((i + (na + 1) / 2) % na, (j + (cols + 1) / 2) % cols)));
    }
    for (index_t i = 0; i < nb; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(out2(na + i, j), d((i + (nb + 1) / 2) % nb, (j + (cols + 1) / 2) % cols)));
    }
  }

  (out2 = concat(0, ifftshift2D(c), ifftshift2D(d))).run(exec);
  exec.sync();
  for (index_t j = 0; j < cols; j++) {
    for (index_t i = 0; i < na; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(out2(i, j), c((i + na / 2) % na, (j + cols / 2) % cols)));
    }
    for (index_t i = 0; i < nb; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(out2(na + i, j), d((i + nb / 2) % nb, (j + cols / 2) % cols)));
    }
  }

  MATX_EXIT_HANDLER();
}
