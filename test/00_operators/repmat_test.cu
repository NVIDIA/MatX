#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsAllExecs, RepMat)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  // example-begin repmat-test-1
  const index_t count0 = 4;
  const index_t count1 = 4;
  const index_t same_reps = 10;
  tensor_t<TestType, 2> t2({count0, count1});
  tensor_t<TestType, 2> t2s({count0 * same_reps, count1 * same_reps});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i);
    }
  }

  auto repop = repmat(t2, same_reps);
  // example-end repmat-test-1
  ASSERT_TRUE(repop.Size(0) == same_reps * t2.Size(0));
  ASSERT_TRUE(repop.Size(1) == same_reps * t2.Size(1));

  (t2s = repop).run(exec);
  exec.sync();

  for (index_t i = 0; i < count0 * same_reps; i++) {
    for (index_t j = 0; j < count1 * same_reps; j++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2s(i, j), t2(i % count0, j % count1)));
    }
  }

  // Now a rectangular repmat
  tensor_t<TestType, 2> t2r({count0 * same_reps, count1 * same_reps * 2});

  auto rrepop = repmat(t2, {same_reps, same_reps * 2});
  ASSERT_TRUE(rrepop.Size(0) == same_reps * t2.Size(0));
  ASSERT_TRUE(rrepop.Size(1) == same_reps * 2 * t2.Size(1));

  (t2r = rrepop).run(exec);
  exec.sync();

  for (index_t i = 0; i < count0 * same_reps; i++) {
    for (index_t j = 0; j < count1 * same_reps * 2; j++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2r(i, j), t2(i % count0, j % count1)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAllExecs, RepMatDifferentSizesInOneExpression)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  // Repetitions of inputs with different sizes can appear in the same expression
  const index_t na = 8;
  const index_t nb = 16;
  const index_t reps = 2;
  auto a = make_tensor<TestType>({na});
  auto b = make_tensor<TestType>({nb});
  auto out = make_tensor<TestType>({(na + nb) * reps});
  for (index_t i = 0; i < na; i++) {
    a(i) = static_cast<detail::value_promote_t<TestType>>(i);
  }
  for (index_t i = 0; i < nb; i++) {
    b(i) = static_cast<detail::value_promote_t<TestType>>(na + i);
  }

  (out = concat(0, repmat(a, reps), repmat(b, reps))).run(exec);
  exec.sync();

  for (index_t i = 0; i < na * reps; i++) {
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(out(i), a(i % na)));
  }
  for (index_t i = 0; i < nb * reps; i++) {
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(out(na * reps + i), b(i % nb)));
  }

  MATX_EXIT_HANDLER();
}
