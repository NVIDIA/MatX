#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsComplexTypesAllExecs, InterleavedTransform)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  index_t m = 10;
  index_t k = 20;
  tensor_t<TestType, 2> t2({m, k});
  tensor_t<typename TestType::value_type, 2> t2p({m * 2, k});
  for (index_t i = 0; i < 2 * m; i++) {
    for (index_t j = 0; j < k; j++) {
      if (i >= m) {
        t2p(i, j) = 2.0f;
      }
      else {
        t2p(i, j) = -1.0f;
      }
    }
  }

  (t2 = interleaved(t2p)).run(exec);
  exec.sync();

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j).real(), t2p(i, j)));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2(i, j).imag(), t2p(i + t2.Size(0), j)));
    }
  }
  MATX_EXIT_HANDLER();
}
