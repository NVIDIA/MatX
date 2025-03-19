#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsAllExecs, GetString)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto A = make_tensor<TestType>({10,20});
  auto B = make_tensor<TestType>({20});
  auto C = make_tensor<TestType>({10,20});

  auto op1 = C = A;
  auto op2 = C = A + B + (TestType)5;
  auto op3 = C = A / B;
  auto op4 = (op1,op2,op3);

  std::cout << "op1: " << op1.str() << std::endl;
  std::cout << "op2: " << op2.str() << std::endl;
  std::cout << "op3: " << op3.str() << std::endl;
  std::cout << "op4: " << op4.str() << std::endl;

  MATX_EXIT_HANDLER();
} 