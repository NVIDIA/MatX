#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsAllExecs, Flatten)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};  

  // example-begin flatten-test-1
  auto t2 = make_tensor<TestType>({10, 2});
  auto val = GenerateData<TestType>();

  for (index_t i = 0; i < t2.Size(0); i++) {
    for (index_t j = 0; j < t2.Size(1); j++) {
      t2(i,j) = val;
    }
  }

  auto t1 = make_tensor<TestType>({t2.Size(0)*t2.Size(1)});
  (t1 = flatten(t2)).run(exec);
  // example-end flatten-test-1
  exec.sync();
  
  for (index_t i = 0; i < t2.Size(0)*t2.Size(1); i++) {
    ASSERT_EQ(t1(i), val);
  }

  MATX_EXIT_HANDLER();
}