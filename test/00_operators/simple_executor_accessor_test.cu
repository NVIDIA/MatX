#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsAllExecsWithoutJIT, SimpleExecutorAccessorTests)
{
  MATX_ENTER_HANDLER();

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  const int N = 3;

  auto t0 = make_tensor<TestType>({});
  auto t1 = make_tensor<TestType>({N});
  auto t2 = make_tensor<TestType>({N,N});
  auto t3 = make_tensor<TestType>({N,N,N});
  auto t4 = make_tensor<TestType>({N,N,N,N});
  auto t5 = make_tensor<TestType>({N,N,N,N,N});
  auto t6 = make_tensor<TestType>({N,N,N,N,N,N});

  // Simple executor accessor tests for tensors of increasing dimensions.
  // These tests verify that we can read from and write to all elements of
  // a tensor, which in turn tests the multi-dimensional indexing logic.
  // These tests do not assign unique values to each unique index, so they
  // are not exhaustive tests. For the CUDA executor, the 5D and higher-rank
  // tensors will use matxOpTDKernel whereas  the smaller tensors each have
  // custom implementations.
  const TestType init = TestType(5);
  const TestType expected = TestType(7);
  (t0 = init).run(exec);
  (t1 = init).run(exec);
  (t2 = init).run(exec);
  (t3 = init).run(exec);
  (t4 = init).run(exec);
  (t5 = init).run(exec);
  (t6 = init).run(exec);

  // We use IF to generate both a read and a write for each tensor index.
  IF(t0 == init, t0 = expected).run(exec);
  IF(t1 == init, t1 = expected).run(exec);
  IF(t2 == init, t2 = expected).run(exec);
  IF(t3 == init, t3 = expected).run(exec);
  IF(t4 == init, t4 = expected).run(exec);
  IF(t5 == init, t5 = expected).run(exec);
  IF(t6 == init, t6 = expected).run(exec);

  cudaStreamSynchronize(0);

  ASSERT_EQ(t0(), expected);
  for (int i0 = 0; i0 < N; i0++) {
    ASSERT_EQ(t1(i0), expected);
    for (int i1 = 0; i1 < N; i1++) {
      ASSERT_EQ(t2(i0,i1), expected);
      for (int i2 = 0; i2 < N; i2++) {
        ASSERT_EQ(t3(i0,i1,i2), expected);
        for (int i3 = 0; i3 < N; i3++) {
          ASSERT_EQ(t4(i0,i1,i2,i3), expected);
          for (int i4 = 0; i4 < N; i4++) {
            ASSERT_EQ(t5(i0,i1,i2,i3,i4), expected);
            for (int i5 = 0; i5 < N; i5++) {
              ASSERT_EQ(t6(i0,i1,i2,i3,i4,i5), expected);
            }
          }
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
} 