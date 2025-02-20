#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;



TYPED_TEST(OperatorTestsNumericAllExecs, Upsample)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  

  ExecType exec{}; 

  {
    // example-begin upsample-test-1
    // Upsample a signal of length 100 by 5
    int n = 5;

    auto t1 = make_tensor<TestType>({100});
    (t1 = static_cast<TestType>(1)).run(exec);
    auto us_op = upsample(t1, 0, n);
    // example-end upsample-test-1
    exec.sync();

    ASSERT_TRUE(us_op.Size(0) == t1.Size(0) * n);
    for (index_t i = 0; i < us_op.Size(0); i++) {
      if ((i % n) == 0) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(us_op(i), t1(i / n)));
      }
      else {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(us_op(i), static_cast<TestType>(0)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, Downsample)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  

  ExecType exec{}; 

  {
    // example-begin downsample-test-1
    int n = 5;

    auto t1 = make_tensor<TestType>({100});
    (t1 = static_cast<TestType>(1)).run(exec);
    auto ds_op = downsample(t1, 0, n);
    // example-end downsample-test-1
    exec.sync();

    ASSERT_TRUE(ds_op.Size(0) == t1.Size(0) / n);
    for (index_t i = 0; i < ds_op.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(ds_op(i), t1(i * n)));
    }
  }

  {
    int n = 3;

    auto t1 = make_tensor<TestType>({100});
    (t1 = static_cast<TestType>(1)).run(exec);
    auto ds_op = downsample(t1, 0, n);

    exec.sync();

    ASSERT_TRUE(ds_op.Size(0) == t1.Size(0) / n + 1);
    for (index_t i = 0; i < ds_op.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(ds_op(i), t1(i * n)));
    }
  }  

  MATX_EXIT_HANDLER();
}