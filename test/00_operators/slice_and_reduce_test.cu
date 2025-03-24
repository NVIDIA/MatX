#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, SliceAndReduceOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;
  ExecType exec{}; 

  tensor_t<TestType, 2> t2t{{20, 10}};
  tensor_t<TestType, 3> t3t{{30, 20, 10}};
  (t2t = clone<2>(linspace((inner_type)0, (inner_type)10, 10), {t2t.Size(0), matxKeepDim})).run(exec);
  (t3t = clone<3>(linspace((inner_type)0, (inner_type)10, 10), {t3t.Size(0), t3t.Size(1), matxKeepDim})).run(exec);
  exec.sync();

  {
    index_t j = 0;
    // example-begin slice-test-3
    // Slice "t2t" by selecting column "j" from a 2D operator and converting to a 1D operator
    auto t2sly = slice<1>(t2t, {0, j}, {matxEnd, matxDropDim});
    // example-end slice-test-3
    for (index_t i = 0; i < t2sly.Size(0); i++) {
      ASSERT_EQ(t2sly(i), t2t(i, j));
    }
  }

  {
    index_t i = 0;
    auto t2slx = slice<1>(t2t, {i, 0}, {matxDropDim, matxEnd});
    for (index_t j = 0; j < t2slx.Size(0); j++) {
      ASSERT_EQ(t2slx(j), t2t(i, j));
    }
  }

  {
    index_t j = 0;
    index_t k = 0;
    auto t3slz = slice<1>(t3t, {0, j, k}, {matxEnd, matxDropDim, matxDropDim});
    for (index_t i = 0; i < t3slz.Size(0); i++) {
      ASSERT_EQ(t3slz(i), t3t(i, j, k));
    }
  }

  {
    index_t i = 0;
    index_t k = 0;
    auto t3sly = slice<1>(t3t, {i, 0, k}, {matxDropDim, matxEnd, matxDropDim});
    for (index_t j = 0; j < t3sly.Size(0); j++) {
      ASSERT_EQ(t3sly(j), t3t(i, j, k));
    }
  }

  {
    index_t i = 0;
    index_t j = 0;
    auto t3slx = slice<1>(t3t, {i, j, 0}, {matxDropDim, matxDropDim, matxEnd});
    for (index_t k = 0; k < t3slx.Size(0); k++) {
      ASSERT_EQ(t3slx(k), t3t(i, j, k));
    }
  }

  {
    index_t k = 0;
    auto t3slzy = slice<2>(t3t, {0, 0, k}, {matxEnd, matxEnd, matxDropDim});
    for (index_t i = 0; i < t3slzy.Size(0); i++) {
      for (index_t j = 0; j < t3slzy.Size(1); j++) {
        ASSERT_EQ(t3slzy(i, j), t3t(i, j, k));
      }
    }
  }

  {
    index_t j = 0;
    auto t3slzx = slice<2>(t3t, {0, j, 0}, {matxEnd, matxDropDim, matxEnd});
    for (index_t i = 0; i < t3slzx.Size(0); i++) {
      for (index_t k = 0; k < t3slzx.Size(1); k++) {
        ASSERT_EQ(t3slzx(i, k), t3t(i, j, k));
      }
    }
  }

  {
    index_t i = 0;
    auto t3slyx = slice<2>(t3t, {i, 0, 0}, {matxDropDim, matxEnd, matxEnd});
    for (index_t j = 0; j < t3slyx.Size(0); j++) {
      for (index_t k = 0; k < t3slyx.Size(1); k++) {
        ASSERT_EQ(t3slyx(j, k), t3t(i, j, k));
      }
    }
  }
  MATX_EXIT_HANDLER();
} 