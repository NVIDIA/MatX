#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, CollapseOp)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  int N = 10;
  int M = 12;
  int K = 14;


  MATX_ENTER_HANDLER();
  auto tiv = make_tensor<TestType>({N,M,K});

  for(int n = 0; n < N; n++) {
    for(int m = 0; m < M; m++) {
      for(int k = 0; k < K; k++) {
        tiv(n,m,k) = inner_type(n*M*K + m*K + k);
      }
    }
  }

  { // rcollapse 2
    auto tov = make_tensor<TestType>({N,M*K});
  
    // example-begin rcollapse-test-1
    // Collapse two right-most dimensions together
    auto op = rcollapse<2>(tiv);
    // example-end rcollapse-test-1

    EXPECT_TRUE(op.Rank() == 2);
    EXPECT_TRUE(op.Size(0) == N);
    EXPECT_TRUE(op.Size(1) == M*K);

    (tov = (TestType)0).run(exec);
    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_TRUE(tiv(n,m,k) == tov(n,m*K+k));
        }
      }
    }
  }
  
  { // lcollapse 12
    auto tov = make_tensor<TestType>({N*M,K});
  
    // example-begin lcollapse-test-1
    // Collapse two left-most dimensions together
    auto op = lcollapse<2>(tiv);
    // example-end lcollapse-test-1

    EXPECT_TRUE(op.Rank() == 2);
    EXPECT_TRUE(op.Size(0) == N*M);
    EXPECT_TRUE(op.Size(1) == K);
    
    
    (tov = (TestType)0).run(exec);
    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_TRUE(tiv(n,m,k) == tov(n*M+m,k));
        }
      }
    }
  }
  
  { // rcollapse 3
    auto tov = make_tensor<TestType>({N*M*K});
  
    auto op = rcollapse<3>(tiv);

    EXPECT_TRUE(op.Rank() == 1);
    EXPECT_TRUE(op.Size(0) == N*M*K);

    (tov = (TestType)0).run(exec);
    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_TRUE(tiv(n,m,k) == tov(n*M*K+m*K+k));
        }
      }
    }
  }

  { // lcollapse 3 
    auto tov = make_tensor<TestType>({N*M*K});
  
    auto op = lcollapse<3>(tiv);

    EXPECT_TRUE(op.Rank() == 1);
    EXPECT_TRUE(op.Size(0) == N*M*K);

    (tov = (TestType)0).run(exec);
    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_TRUE(tiv(n,m,k) == tov(n*M*K+m*K+k));
        }
      }
    }
  }

  if constexpr (is_cuda_non_jit_executor<ExecType> && (std::is_same_v<TestType, float> || std::is_same_v<TestType, double>))
  { // rcollapse with nested transform operator
    auto tov = make_tensor<TestType>({N,M*K});
    auto delta = make_tensor<TestType>({1,1});
    delta(0,0) = static_cast<typename inner_op_type_t<TestType>::type>(1.0);

    auto op = rcollapse<2>(conv2d(tiv, delta, MATX_C_MODE_SAME));

    EXPECT_TRUE(op.Rank() == 2);
    EXPECT_TRUE(op.Size(0) == N);
    EXPECT_TRUE(op.Size(1) == M*K);

    (tov = (TestType)0).run(exec);
    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_TRUE(tiv(n,m,k) == tov(n,m*K+k));
        }
      }
    }
  }

  if constexpr (is_cuda_non_jit_executor<ExecType> && (std::is_same_v<TestType, float> || std::is_same_v<TestType, double>))
  { // lcollapse with nested transform operator
    auto tov = make_tensor<TestType>({N*M,K});
    auto delta = make_tensor<TestType>({1,1});
    delta(0,0) = static_cast<typename inner_op_type_t<TestType>::type>(1.0);

    auto op = lcollapse<2>(conv2d(tiv, delta, MATX_C_MODE_SAME));

    EXPECT_TRUE(op.Rank() == 2);
    EXPECT_TRUE(op.Size(0) == N*M);
    EXPECT_TRUE(op.Size(1) == K);

    (tov = (TestType)0).run(exec);
    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_TRUE(tiv(n,m,k) == tov(n*M+m,k));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
} 