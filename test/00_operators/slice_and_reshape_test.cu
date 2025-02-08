#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecs, SliceAndReshape)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  {
    // Unit test combining slice with reshape which showed a bug in the past
    auto t = make_tensor<TestType>({100});
    (t = linspace<0>(t.Shape(), (TestType)0, (TestType)99)).run(exec);
    auto rs = reshape(t, {2, 10, 5});
    auto s = slice(rs, {0, 0, 2}, {matxEnd, matxEnd, matxEnd});
    exec.sync();
    
    for (index_t i = 0; i < s.Size(0); i++) {
      for (index_t j = 0; j < s.Size(1); j++) {
        for (index_t k = 0; k < s.Size(2); k++) {
          ASSERT_EQ(t(i*50 + j*5 + k+2), s(i,j,k));
        }
      }
    }
  } 

  MATX_EXIT_HANDLER(); 
} 