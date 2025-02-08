#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatAllExecs, IsClose)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  // example-begin isclose-test-1
  auto A = make_tensor<TestType>({5, 5, 5});
  auto B = make_tensor<TestType>({5, 5, 5});
  auto C = make_tensor<int>({5, 5, 5});

  (A = ones<TestType>(A.Shape())).run(exec);
  (B = ones<TestType>(B.Shape())).run(exec);
  (C = isclose(A, B)).run(exec);
  // example-end isclose-test-1
  exec.sync();

  for(int i=0; i < A.Size(0); i++) {
    for(int j=0; j < A.Size(1); j++) {
      for(int k=0; k < A.Size(2); k++) {
        ASSERT_EQ(C(i,j,k), 1);
      }
    }
  }

  B(1,1,1) = 2;
  (C = isclose(A, B)).run(exec);
  exec.sync();

  for(int i=0; i < A.Size(0); i++) {
    for(int j=0; j < A.Size(1); j++) {
      for(int k=0; k < A.Size(2); k++) {
        if (i == 1 && j == 1 && k == 1) {
          ASSERT_EQ(C(i,j,k), 0); 
        }
        else {
          ASSERT_EQ(C(i,j,k), 1);
        }
      }
    }
  }  

  MATX_EXIT_HANDLER();
} 