#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsNumericNoHalfAllExecs, AdvancedOperators)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  index_t count = 100;

  tensor_t<TestType, 1> a({count});
  tensor_t<TestType, 1> b({count});
  tensor_t<TestType, 1> c({count});

  for (index_t i = 0; i < count; i++) {
    a(i) = static_cast<detail::value_promote_t<TestType>>(i);
    b(i) = static_cast<detail::value_promote_t<TestType>>(i + 100);
  }

  {
    (c = a + b).run(exec);

    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt + (tcnt + (TestType)100)));
    }
  }

  {
    (c = a * b).run(exec);

    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt * (tcnt + (TestType)100)));
    }
  }

  {
    (c = a * b + a).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TestType)100) + tcnt));
    }
  }

  {

    (c = a * b + a * (TestType)4.0f).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TestType)100.0f) + tcnt * (TestType)4));
    }
  }
  MATX_EXIT_HANDLER();
}



TYPED_TEST(OperatorTestsNumericNonComplexAllExecs, AdvancedOperators)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  index_t count = 10;

  tensor_t<TestType, 1> a({count});
  tensor_t<TestType, 1> b({count});
  tensor_t<TestType, 1> c({count});

  for (index_t i = 0; i < count; i++) {
    a(i) = (TestType)i;
    b(i) = (TestType)(i + 2);
  }

  {
    (c = a + b).run(exec);

    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = (TestType)i;
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(c(i), (TestType)((float)tcnt + ((float)tcnt + 2.0f))));
    }
  }

  {
    (c = a * b).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = (TestType)i;
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), (float)tcnt * ((float)tcnt + 2.0f)));
    }
  }

  {
    (c = a * b + a).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = (TestType)i;
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), (float)tcnt * ((float)tcnt + 2.0f) + (float)tcnt));
    }
  }

  {

    (c = a * b + a * (TestType)2.0f).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = (TestType)i;
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), (float)tcnt * ((float)tcnt + 2.0f) + (float)tcnt * 2.0f));
    }
  }
  MATX_EXIT_HANDLER();
}