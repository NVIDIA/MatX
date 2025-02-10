#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, Abs2)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};

  if constexpr (std::is_same_v<TestType, cuda::std::complex<float>> &&
    std::is_same_v<ExecType,cudaExecutor>) {
    // example-begin abs2-test-1
    auto x = make_tensor<cuda::std::complex<float>>({});
    auto y = make_tensor<float>({});
    x() = { 1.5f, 2.5f };
    (y = abs2(x)).run(exec);
    exec.sync();
    ASSERT_NEAR(y(), 1.5f*1.5f+2.5f*2.5f, 1.0e-6);
    // example-end abs2-test-1
  }

  auto x = make_tensor<TestType>({});
  auto y = make_tensor<inner_type>({});
  if constexpr (is_complex_v<TestType>) {
    x() = TestType{2.0, 2.0};
    (y = abs2(x)).run(exec);
    exec.sync();
    ASSERT_NEAR(y(), 8.0, 1.0e-6);
  } else {
    x() = 2.0;
    (y = abs2(x)).run(exec);
    exec.sync();
    ASSERT_NEAR(y(), 4.0, 1.0e-6);

    // Test with higher rank tensor
    auto x3 = make_tensor<TestType>({3,3,3});
    auto y3 = make_tensor<TestType>({3,3,3});
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          x3(i,j,k) = static_cast<TestType>(i*9 + j*3 + k);
        }
      }
    }

    (y3 = abs2(x3)).run(exec);
    exec.sync();

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          TestType v = static_cast<TestType>(i*9 + j*3 + k);
          ASSERT_NEAR(y3(i,j,k), v*v, 1.0e-6);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}
