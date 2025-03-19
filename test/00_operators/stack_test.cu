#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, Stack)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto t1a = make_tensor<TestType>({5});
  auto t1b = make_tensor<TestType>({5});
  auto t1c = make_tensor<TestType>({5});
 
  auto cop = concat(0, t1a, t1b, t1c);
  
  (cop = (TestType)2).run(exec);
  exec.sync();

  {
    // example-begin stack-test-1
    // Stack 1D operators "t1a", "t1b", and "t1c" together along the first dimension
    auto op = stack(0, t1a, t1b, t1c);
    // example-end stack-test-1
   
    for(int i = 0; i < t1a.Size(0); i++) {
      ASSERT_EQ(op(0,i), t1a(i));
      ASSERT_EQ(op(1,i), t1b(i));
      ASSERT_EQ(op(2,i), t1c(i));
    }
  }  
 
  {
    auto op = stack(1, t1a, t1b, t1c);
    
    for(int i = 0; i < t1a.Size(0); i++) {
      ASSERT_EQ(op(i,0), t1a(i));
      ASSERT_EQ(op(i,1), t1b(i));
      ASSERT_EQ(op(i,2), t1c(i));
    }
  }  
  
  MATX_EXIT_HANDLER();
}