#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsAllExecs, BaseOp)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto A = make_tensor<TestType>({10,20});
  auto op = A + A;

  EXPECT_TRUE(op.Size(0) == A.Size(0));
  EXPECT_TRUE(op.Size(1) == A.Size(1));

  auto shape = op.Shape();

  EXPECT_TRUE(shape[0] == A.Size(0));
  EXPECT_TRUE(shape[1] == A.Size(1));
 
  EXPECT_TRUE(A.TotalSize() == op.TotalSize());

  MATX_EXIT_HANDLER();
} 