#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsCastToFloatAllExecs, ComplexCast)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  // 0D tensor tests
  {
    // example-begin as_complex_double-test-2
    auto c64 = make_tensor<cuda::std::complex<double>>({});
    auto in_real = make_tensor<TestType>({});
    auto in_imag = make_tensor<TestType>({});
    in_real.SetVals({3});
    in_imag.SetVals({5});
    (c64 = as_complex_double(in_real, in_imag)).run(exec);
    // c64() will be (3.0, 5.0)
    // example-end as_complex_double-test-2
    exec.sync();

    ASSERT_EQ(c64().real(), 3.0);
    ASSERT_EQ(c64().imag(), 5.0);
  }
  {
    // example-begin as_complex_float-test-2
    auto c32 = make_tensor<cuda::std::complex<float>>({});
    auto in_real = make_tensor<TestType>({});
    auto in_imag = make_tensor<TestType>({});
    in_real.SetVals({3});
    in_imag.SetVals({5});
    (c32 = as_complex_float(in_real, in_imag)).run(exec);
    // c32() will be (3.0f, 5.0f)
    // example-end as_complex_float-test-2
    exec.sync();

    ASSERT_EQ(c32().real(), 3.0f);
    ASSERT_EQ(c32().imag(), 5.0f);
  }

  // 2D tensor tests
  {
    const int N = 4;
    auto c32 = make_tensor<cuda::std::complex<float>>({N,N});
    auto c64 = make_tensor<cuda::std::complex<double>>({N,N});
    auto in_real = make_tensor<TestType>({N,N});
    auto in_imag = make_tensor<TestType>({N,N});
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        in_real(i,j) = static_cast<TestType>(4);
        in_imag(i,j) = static_cast<TestType>(6);
      }
    }

    exec.sync();

    (c32 = as_complex_float(in_real, in_imag)).run(exec);
    (c64 = as_complex_double(in_real, in_imag)).run(exec);

    exec.sync();

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_EQ(c32(i,j).real(), 4.0f);
        ASSERT_EQ(c32(i,j).imag(), 6.0f);
        ASSERT_EQ(c64(i,j).real(), 4.0);
        ASSERT_EQ(c64(i,j).imag(), 6.0);
      }
    }
  }

  MATX_EXIT_HANDLER();
}