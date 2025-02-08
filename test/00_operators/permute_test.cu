#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsAllExecs, PermuteOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto A = make_tensor<TestType>({10,20,30});
  for(int i=0; i < A.Size(0); i++) {
    for(int j=0; j < A.Size(1); j++) {
      for(int k=0; k < A.Size(2); k++) {
        A(i,j,k) = static_cast<typename inner_op_type_t<TestType>::type>( i * A.Size(1)*A.Size(2) +
         j * A.Size(2) + k);  
      }
    }
  }

  // example-begin permute-test-1
  // Permute from dims {0, 1, 2} to {2, 0, 1}
  auto op = permute(A, {2, 0, 1});
  // example-end permute-test-1
  auto At = A.Permute({2, 0, 1});

  ASSERT_TRUE(op.Size(0) == A.Size(2));
  ASSERT_TRUE(op.Size(1) == A.Size(0));
  ASSERT_TRUE(op.Size(2) == A.Size(1));
  
  ASSERT_TRUE(op.Size(0) == At.Size(0));
  ASSERT_TRUE(op.Size(1) == At.Size(1));
  ASSERT_TRUE(op.Size(2) == At.Size(2));

  for(int i=0; i < op.Size(0); i++) {
    for(int j=0; j < op.Size(1); j++) {
      for(int k=0; k < op.Size(2); k++) {
        ASSERT_TRUE( op(i,j,k) == A(j,k,i));  
        ASSERT_TRUE( op(i,j,k) == At(i,j,k));
      }
    }
  }

  MATX_EXIT_HANDLER();
} 