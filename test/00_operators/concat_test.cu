#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;



TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Concatenate)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 
  index_t i, j;

  // example-begin concat-test-1
  auto t11 = make_tensor<TestType>({10});
  auto t12 = make_tensor<TestType>({5});
  auto t1o = make_tensor<TestType>({15});

  t11.SetVals({0,1,2,3,4,5,6,7,8,9});
  t12.SetVals({0,1,2,3,4});

  // Concatenate "t11" and "t12" into a new 1D tensor
  (t1o = concat(0, t11, t12)).run(exec);
  // example-end concat-test-1
  exec.sync();

  for (i = 0; i < t11.Size(0) + t12.Size(0); i++) {
    if (i < t11.Size(0)) {
      ASSERT_EQ(t11(i), t1o(i));
    }
    else {
      ASSERT_EQ(t12(i - t11.Size(0)), t1o(i));
    }
  }

  // Test contcat with nested transforms
  if constexpr (is_cuda_non_jit_executor<ExecType> && (std::is_same_v<TestType, float> || std::is_same_v<TestType, double>)) {
    auto delta = make_tensor<TestType>({1});
    delta.SetVals({1.0});

    (t1o = 0).run(exec);
    (t1o = concat(0, conv1d(t11, delta, MATX_C_MODE_SAME), conv1d(t12, delta, MATX_C_MODE_SAME))).run(exec);

    exec.sync();

    for (i = 0; i < t11.Size(0) + t12.Size(0); i++) {
      if (i < t11.Size(0)) {
        ASSERT_EQ(t11(i), t1o(i));
      }
      else {
        ASSERT_EQ(t12(i - t11.Size(0)), t1o(i));
      }
    }
  }

  // 2D tensors
  auto t21 = make_tensor<TestType>({4, 4});
  auto t22 = make_tensor<TestType>({3, 4});
  auto t23 = make_tensor<TestType>({4, 3});

  auto t2o1 = make_tensor<TestType>({7,4});  
  auto t2o2 = make_tensor<TestType>({4,7});  
  t21.SetVals({{1,2,3,4},
               {2,3,4,5},
               {3,4,5,6},
               {4,5,6,7}} );
  t22.SetVals({{5,6,7,8},
               {6,7,8,9},
               {9,10,11,12}});
  t23.SetVals({{5,6,7},
               {6,7,8},
               {9,10,11},
               {10,11,12}});

  (t2o1 = concat(0, t21, t22)).run(exec);
  exec.sync();

  for (i = 0; i < t21.Size(0) + t22.Size(0); i++) {
    for (j = 0; j < t21.Size(1); j++) {
      if (i < t21.Size(0)) {
        ASSERT_EQ(t21(i,j), t2o1(i,j));
      }
      else {
        ASSERT_EQ(t22(i - t21.Size(0), j), t2o1(i,j));
      }
    }
  }

  (t2o2 = concat(1, t21, t23)).run(exec); 
  exec.sync();
  
  for (j = 0; j < t21.Size(1) + t23.Size(1); j++) {
    for (i = 0; i < t21.Size(0); i++) {
      if (j < t21.Size(1)) {
        ASSERT_EQ(t21(i,j), t2o2(i,j));
      }
      else {
        ASSERT_EQ(t23(i, j - t21.Size(1)), t2o2(i,j));
      }
    }
  }  

  auto t1o1 = make_tensor<TestType>({30});  

  // Concatenating 3 tensors
  (t1o1 = concat(0, t11, t11, t11)).run(exec);
  exec.sync();

  for (i = 0; i < t1o1.Size(0); i++) {
    ASSERT_EQ(t1o1(i), t11(i % t11.Size(0)));
  }


  // Multiple concatenations
  {
    auto a = matx::make_tensor<float>({10});
    auto b = matx::make_tensor<float>({10});
    auto c = matx::make_tensor<float>({10});
    auto d = matx::make_tensor<float>({10});
    
    auto result = matx::make_tensor<float>({40});
    a.SetVals({1,2,3,4,5,6,7,8,9,10});
    b.SetVals({11,12,13,14,15,16,17,18,19,20});
    c.SetVals({21,22,23,24,25,26,27,28,29,30});
    d.SetVals({31,32,33,34,35,36,37,38,39,40});
    
    auto tempConcat1 = matx::concat(0, a, b);
    auto tempConcat2 = matx::concat(0, c, d);
    (result = matx::concat(0, tempConcat1, tempConcat2 )).run(exec);

    exec.sync();
    for (int cnt = 0; cnt < result.Size(0); cnt++) {
      ASSERT_EQ(result(cnt), cnt + 1);
    }    
  }  
  
  MATX_EXIT_HANDLER();
}