#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, SliceStrideOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};
  // example-begin slice-test-2
  auto t1 = make_tensor<TestType>({10});

  t1.SetVals({10, 20, 30, 40, 50, 60, 70, 80, 90, 100});

  // Slice every other element from a 1D tensor (stride of two)
  auto t1t = slice(t1, {0}, {matxEnd}, {2});
  // example-end slice-test-2
 
  for (index_t i = 0; i < t1.Size(0); i += 2) {
    ASSERT_EQ(t1(i), t1t(i / 2));
  }

  auto t1t2 = slice(t1, {2}, {matxEnd}, {2});

  for (index_t i = 0; i < t1t2.Size(0); i++) {
    ASSERT_EQ(TestType(30 + 20 * i), t1t2(i));
  }

  MATX_EXIT_HANDLER();
} 