#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatAllExecs, Print)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t1 = make_tensor<TestType>({3});
  auto r1 = ones<TestType>(t1.Shape());
  print(r1);

  auto t3 = matx::make_tensor<TestType>({3, 2, 20});
  print(matx::ones(t3.Shape()), 1, 0, 2);

  MATX_EXIT_HANDLER();
} 