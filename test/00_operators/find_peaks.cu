#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexSingleThreadedHostAllExecs, FindPeaks)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  // example-begin findpeaks-test-1
  auto tiv = make_tensor<TestType>({32});
  auto tidx = make_tensor<index_t>({32});
  tiv.SetVals({ 3.9905e-01, 5.1668e-01, 2.4930e-02, 9.4008e-01, 9.4585e-01, 7.9673e-01, 4.1501e-01, 8.2026e-01, 2.2904e-01, 9.0959e-01, 1.0000e+01,
    7.5222e-02, 4.0922e-01, 9.6007e-01, 2.0930e-01, 1.9395e-01, 8.9094e-01, 4.3867e-01, 3.5698e-01, 5.4537e-01, 8.2992e-01, 2.0994e-01, 7.6842e-01,
    4.2899e-01, 2.1167e-01, 6.6055e-01, 1.6536e-01, 4.2499e-01, 9.9267e-01, 6.9642e-01, 2.4719e-01, 7.0281e-01});
  auto num_found = make_tensor<int>({});

  (mtie(tidx, num_found) = find_peaks(tiv, static_cast<TestType>(0.5), static_cast<TestType>(0.1))).run(exec);
  // example-end findpeaks-test-1

  exec.sync();
  ASSERT_TRUE(num_found() == 9);

  std::vector<int> expected_idxs = {1, 7, 10, 13, 16, 20, 22, 25, 28};
  for (int i = 0; i < num_found(); i++) {
    ASSERT_TRUE(tidx(i) == expected_idxs[i]);
  }

  MATX_EXIT_HANDLER();
} 