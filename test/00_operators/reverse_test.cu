#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, Reverse)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  index_t count0 = 100;
  index_t count1 = 200;
  tensor_t<TestType, 1> t1({count0});
  tensor_t<TestType, 1> t1r({count0});
  tensor_t<TestType, 2> t2({count0, count1});
  tensor_t<TestType, 2> t2r({count0, count1});

  for (index_t i = 0; i < count0; i++) {
    t1(i) = static_cast<detail::value_promote_t<TestType>>(i);
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i * count1 + j);
    }
  }

  {
    (t1r = reverse<0>(t1)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1r(i), t1(count0 - i - 1)));
    }

    // example-begin reverse-test-1
    // Reverse the values of t2 along dimension 0
    (t2r = reverse<0>(t2)).run(exec);
    // example-end reverse-test-1
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(count0 - i - 1, j)));
      }
    }
  }

  {
    (t2r = reverse<1>(t2)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(i, count1 - j - 1)));
      }
    }
  }

  {
    (t2r = reverse<0,1>(t2)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2r(i, j), t2(count0 - i - 1, count1 - j - 1)));
      }
    }
  }

  // Flip versions
  {
    (t1r = flipud(t1)).run(exec);
    exec.sync();
    for (index_t i = 0; i < count0; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1r(i), t1(count0 - i - 1)));
    }

    // example-begin flipud-test-1
    (t2r = flipud(t2)).run(exec);
    // example-end flipud-test-1
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(count0 - i - 1, j)));
      }
    }
  }

  {
    (t1r = fliplr(t1)).run(exec);
    exec.sync();
    for (index_t i = 0; i < count0; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1r(i), t1(count0 - i - 1)));
    }

    // example-begin fliplr-test-1
    (t2r = fliplr(t2)).run(exec);
    // example-end fliplr-test-1
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(i, count1 - j - 1)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}