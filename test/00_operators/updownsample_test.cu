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
// Upsample a rank-3 tensor along each axis on the executor
TYPED_TEST(OperatorTestsNumericAllExecs, UpsampleAllAxes)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_t = typename inner_op_type_t<TestType>::type;

  ExecType exec{};

  const index_t n = 3;
  const cuda::std::array<index_t, 3> in_shape{2, 3, 4};
  auto in = make_tensor<TestType>(in_shape);
  for (index_t i = 0; i < 2; i++)
    for (index_t j = 0; j < 3; j++)
      for (index_t k = 0; k < 4; k++)
        in(i, j, k) = static_cast<inner_t>(i * 12 + j * 4 + k + 1);

  for (int axis = 0; axis < 3; axis++) {
    cuda::std::array<index_t, 3> out_shape = in_shape;
    out_shape[axis] *= n;
    auto out = make_tensor<TestType>(out_shape);
    (out = upsample(in, axis, n)).run(exec);
    exec.sync();

    for (index_t i = 0; i < out_shape[0]; i++) {
      for (index_t j = 0; j < out_shape[1]; j++) {
        for (index_t k = 0; k < out_shape[2]; k++) {
          index_t idx[3] = {i, j, k};
          TestType expected = static_cast<inner_t>(0);
          if (idx[axis] % n == 0) {
            idx[axis] /= n;
            expected = in(idx[0], idx[1], idx[2]);
          }
          ASSERT_TRUE(MatXUtils::MatXTypeCompare(out(i, j, k), expected)) << "axis=" << axis;
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// The upsample and downsample dim must be within the operator's rank, and the rate must be
// positive, in every build mode
TEST(OperatorValidationTests, UpDownsampleInvalidArgs)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto in = make_tensor<TestType>({2, 3, 4});

  EXPECT_THROW(upsample(in, 3, 2), matx::detail::matxException);
  EXPECT_THROW(upsample(in, -1, 2), matx::detail::matxException);
  EXPECT_NO_THROW(upsample(in, 2, 2));

  EXPECT_THROW(downsample(in, 3, 2), matx::detail::matxException);
  EXPECT_THROW(downsample(in, -1, 2), matx::detail::matxException);
  EXPECT_NO_THROW(downsample(in, 2, 2));

  EXPECT_THROW(upsample(in, 0, 0), matx::detail::matxException);
  EXPECT_THROW(upsample(in, 0, -2), matx::detail::matxException);
  EXPECT_THROW(downsample(in, 0, 0), matx::detail::matxException);
  EXPECT_THROW(downsample(in, 0, -2), matx::detail::matxException);
  EXPECT_NO_THROW(upsample(in, 0, 1));
  EXPECT_NO_THROW(downsample(in, 0, 1));

  MATX_EXIT_HANDLER();
}
