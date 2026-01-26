#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsNumericNonComplexAllExecs, Overlap)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 


  tensor_t<TestType, 1> a{{10}};
  a.SetVals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto ao = overlap(a, {4}, {2});

  tensor_t<TestType, 2> b{{4, 4}};
  b.SetVals({{0, 1, 2, 3}, {2, 3, 4, 5}, {4, 5, 6, 7}, {6, 7, 8, 9}});
  for (index_t i = 0; i < b.Size(0); i++) {
    for (index_t j = 0; j < b.Size(1); j++) {
      ASSERT_EQ(ao(i, j), b(i, j));
    }
  }

  auto ao2 = overlap(a, {4}, {1});

  tensor_t<TestType, 2> b2{{7, 4}};
  b2.SetVals({{0, 1, 2, 3},
              {1, 2, 3, 4},
              {2, 3, 4, 5},
              {3, 4, 5, 6},
              {4, 5, 6, 7},
              {5, 6, 7, 8},
              {6, 7, 8, 9}});
  for (index_t i = 0; i < b2.Size(0); i++) {
    for (index_t j = 0; j < b2.Size(1); j++) {
      ASSERT_EQ(ao2(i, j), b2(i, j));
    }
  }

  auto ao3 = overlap(a, {4}, {3});
  tensor_t<TestType, 2> b3{{3, 4}};
  b3.SetVals({{0, 1, 2, 3}, {3, 4, 5, 6}, {6, 7, 8, 9}});
  for (index_t i = 0; i < b3.Size(0); i++) {
    for (index_t j = 0; j < b3.Size(1); j++) {
      ASSERT_EQ(ao3(i, j), b3(i, j));
    }
  }

  auto ao4 = overlap(a, {3}, {2});
  tensor_t<TestType, 2> b4{{4, 3}};
  b4.SetVals({{0, 1, 2}, {2, 3, 4}, {4, 5, 6}, {6, 7, 8}});
  for (index_t i = 0; i < b4.Size(0); i++) {
    for (index_t j = 0; j < b4.Size(1); j++) {
      ASSERT_EQ(ao4(i, j), b4(i, j));
    }
  }

  // Test with an operator input
  // example-begin overlap-test-1  
  auto aop = linspace((TestType)0, (TestType)9, a.Size(0));
  tensor_t<TestType, 2> b4out{{4, 3}};

  // Input is {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  // Output is: {{0, 1, 2}, {2, 3, 4}, {4, 5, 6}, {6, 7, 8}}
  (b4out = overlap(aop, {3}, {2})).run(exec);
  // example-end overlap-test-1  

  ASSERT_EQ(b4out.Size(0), 4);
  ASSERT_EQ(b4out.Size(1), 3);

  exec.sync();
  for (index_t i = 0; i < b4.Size(0); i++) {
    for (index_t j = 0; j < b4.Size(1); j++) {
      ASSERT_EQ(b4out(i, j), b4(i, j));
    }
  }  

  MATX_EXIT_HANDLER();
}