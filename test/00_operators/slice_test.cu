#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, SliceOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  // example-begin slice-test-1
  auto t2 = make_tensor<TestType>({20, 10});
  auto t3 = make_tensor<TestType>({30, 20, 10});
  auto t4 = make_tensor<TestType>({40, 30, 20, 10});

  (t2 = linspace<1>(t2.Shape(), (inner_type)0, (inner_type)10)).run(exec);
  (t3 = linspace<2>(t3.Shape(), (inner_type)0, (inner_type)10)).run(exec);
  (t4 = linspace<3>(t4.Shape(), (inner_type)0, (inner_type)10)).run(exec);
  exec.sync();

  // Slice with different start and end points in each dimension
  auto t2t = slice(t2, {1, 2}, {3, 5});
  auto t3t = slice(t3, {1, 2, 3}, {3, 5, 7});
  auto t4t = slice(t4, {1, 2, 3, 4}, {3, 5, 7, 9});
  // example-end slice-test-1

  ASSERT_EQ(t2t.Size(0), 2);
  ASSERT_EQ(t2t.Size(1), 3);

  ASSERT_EQ(t3t.Size(0), 2);
  ASSERT_EQ(t3t.Size(1), 3);
  ASSERT_EQ(t3t.Size(2), 4);

  ASSERT_EQ(t4t.Size(0), 2);
  ASSERT_EQ(t4t.Size(1), 3);
  ASSERT_EQ(t4t.Size(2), 4);
  ASSERT_EQ(t4t.Size(3), 5);

  for (index_t i = 0; i < t2t.Size(0); i++) {
    for (index_t j = 0; j < t2t.Size(1); j++) {
      ASSERT_EQ(t2t(i, j), t2(i + 1, j + 2));
    }
  }

  for (index_t i = 0; i < t3t.Size(0); i++) {
    for (index_t j = 0; j < t3t.Size(1); j++) {
      for (index_t k = 0; k < t3t.Size(2); k++) {
        ASSERT_EQ(t3t(i, j, k), t3(i + 1, j + 2, k + 3));
      }
    }
  }

  for (index_t i = 0; i < t4t.Size(0); i++) {
    for (index_t j = 0; j < t4t.Size(1); j++) {
      for (index_t k = 0; k < t4t.Size(2); k++) {
        for (index_t l = 0; l < t4t.Size(3); l++) {
          ASSERT_EQ(t4t(i, j, k, l), t4(i + 1, j + 2, k + 3, l + 4));
        }
      }
    }
  }

  // Test SliceOp applied to a transform, using transpose() as an example transform
  auto t2trans = make_tensor<TestType>({3, 2});
  (t2trans = slice(transpose(t2), {2, 1}, {5, 3})).run(exec);
  exec.sync();

  ASSERT_EQ(t2trans.Size(0), 3);
  ASSERT_EQ(t2trans.Size(1), 2);
  for (index_t i = 0; i < t2trans.Size(0); i++) {
    for (index_t j = 0; j < t2trans.Size(1); j++) {
      ASSERT_EQ(t2trans(i, j), t2(j + 1, i + 2));
    }
  }

  // Negative indexing. These should give the same results
  // example-begin slice-test-4
  auto t2sn = slice(t2, {-4, -5}, {matxEnd, matxEnd});
  auto t2s = slice(t2, {t2.Size(0) - 4, t2.Size(1) - 5}, {matxEnd, matxEnd});

  // example-end slice-test-4
  exec.sync();
  ASSERT_EQ(t2sn.Size(0), t2s.Size(0));
  ASSERT_EQ(t2sn.Size(1), t2s.Size(1));
  for (index_t i = 0; i < t2sn.Size(0); i++) {
    for (index_t j = 0; j < t2sn.Size(1); j++) {
      ASSERT_EQ(t2sn(i, j), t2s(i, j));
    }
  }  

  MATX_EXIT_HANDLER();
} 