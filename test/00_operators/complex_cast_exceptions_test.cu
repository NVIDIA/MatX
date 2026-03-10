#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TEST(OperatorTests, ComplexCastExceptions)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 4;
  auto t = make_tensor<int8_t>({count0});
  auto t2 = make_tensor<int8_t>({count0});
  auto to = make_tensor<float>({count0});

  cudaExecutor exec{};

  const int N = 3;
  cuda::std::array<index_t, N> real_dims, imag_dims;
  real_dims.fill(5);
  imag_dims.fill(5);

  auto out = make_tensor<cuda::std::complex<float>>(real_dims);
  auto test_code = [&real_dims, &imag_dims]() {
      auto re = make_tensor<float>(real_dims);
      auto im = make_tensor<float>(imag_dims);
      [[maybe_unused]] auto op = as_complex_float(re, im);
  };

  for (int i = 0; i < N; i++) {
    real_dims[i] = 6;
    ASSERT_THROW({ test_code(); }, matx::detail::matxException);
    real_dims[i] = 5;

    imag_dims[i] = 6;
    ASSERT_THROW({ test_code(); }, matx::detail::matxException);
    imag_dims[i] = 5;
  }

  ASSERT_NO_THROW({ test_code(); });

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, OperatorSizeCompatibility)
{
  MATX_ENTER_HANDLER();

  // Test that MATX_ASSERT_COMPATIBLE_OP_SIZES correctly checks all dimensions.

  // Test 2D tensors with mismatch in first dimension
  {
    auto t1 = make_tensor<float>({4, 7});
    auto t2 = make_tensor<float>({5, 7});
    ASSERT_THROW({ [[maybe_unused]] auto op = t1 + t2; }, matx::detail::matxException);
  }

  // Test 3D tensors with mismatch in first dimension (last dimension matches)
  {
    auto t1 = make_tensor<float>({4, 3, 7});
    auto t2 = make_tensor<float>({5, 3, 7});
    ASSERT_THROW({ [[maybe_unused]] auto op = t1 + t2; }, matx::detail::matxException);
  }

  // Test 3D tensors with mismatch in middle dimension (last dimension matches)
  {
    auto t1 = make_tensor<float>({4, 3, 7});
    auto t2 = make_tensor<float>({4, 5, 7});
    ASSERT_THROW({ [[maybe_unused]] auto op = t1 + t2; }, matx::detail::matxException);
  }

  // Test 4D tensors with mismatch in first dimension (last dimension matches)
  {
    auto t1 = make_tensor<float>({2, 3, 4, 7});
    auto t2 = make_tensor<float>({3, 3, 4, 7});
    ASSERT_THROW({ [[maybe_unused]] auto op = t1 + t2; }, matx::detail::matxException);
  }

  // Test 4D tensors with mismatch in second dimension (last dimension matches)
  {
    auto t1 = make_tensor<float>({2, 3, 4, 7});
    auto t2 = make_tensor<float>({2, 5, 4, 7});
    ASSERT_THROW({ [[maybe_unused]] auto op = t1 + t2; }, matx::detail::matxException);
  }

  // Test 4D tensors with mismatch in third dimension (last dimension matches)
  {
    auto t1 = make_tensor<float>({2, 3, 4, 7});
    auto t2 = make_tensor<float>({2, 3, 6, 7});
    ASSERT_THROW({ [[maybe_unused]] auto op = t1 + t2; }, matx::detail::matxException);
  }

  // Test that compatible sizes don't throw
  {
    auto t1 = make_tensor<float>({4, 3, 7});
    auto t2 = make_tensor<float>({4, 3, 7});
    ASSERT_NO_THROW({ [[maybe_unused]] auto op = t1 + t2; });
  }

  // Test with multiplication operator
  {
    auto t1 = make_tensor<float>({4, 3, 7});
    auto t2 = make_tensor<float>({5, 3, 7});
    ASSERT_THROW({ [[maybe_unused]] auto op = t1 * t2; }, matx::detail::matxException);
  }

  // Test with subtraction operator
  {
    auto t1 = make_tensor<float>({4, 5, 7});
    auto t2 = make_tensor<float>({4, 3, 7});
    ASSERT_THROW({ [[maybe_unused]] auto op = t1 - t2; }, matx::detail::matxException);
  }

  MATX_EXIT_HANDLER();
}
