#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, RemapOp)
{
  int N = 10;

  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  auto tiv = make_tensor<TestType>({N,N});

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      tiv(i,j) = inner_type(i*N+j);
    }
  }

  { // Identity Gather test

    // example-begin remap-test-1
    auto tov = make_tensor<TestType>({N, N});
    auto idx = make_tensor<int>({N});
    
    for(int i = 0; i < N; i++) {
      idx(i) = i;
    }

    // Remap 2D operator "tiv" by selecting elements from dimension 0 stored in "idx"
    (tov = remap<0>(tiv, idx)).run(exec);
    // example-end remap-test-1
    exec.sync();

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }
    
    (tov = remap<1>(tiv, idx)).run(exec);
    exec.sync();
    
    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }

    // example-begin remap-test-2
    // Remap 2D operator "tiv" by selecting elements from dimensions 0 and 1 stored in "idx"
    (tov = remap<0,1>(tiv, idx, idx)).run(exec);
    // example-end remap-test-2
    exec.sync();
    
    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }
  }
  
  { // Identity lvalue test

    auto tov = make_tensor<TestType>({N, N});
    auto idx = make_tensor<int>({N});
    
    for(int i = 0; i < N; i++) {
      idx(i) = i;
    }

    (tov = (TestType)0).run(exec);

    (remap<0>(tov, idx) = tiv).run(exec);
    
    exec.sync();

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }
    
    (tov = (TestType)0).run(exec);
    (remap<1>(tov, idx) = tiv).run(exec);
    exec.sync();
    
    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }
    
    (tov = (TestType)0).run(exec);
    (remap<0,1>(tov, idx, idx) = tiv).run(exec);
    exec.sync();
    
    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }
  }

  { // Reverse test
    
    auto tov = make_tensor<TestType>({N,N});
    auto idx = make_tensor<int>({N});
    
    for(int i = 0; i < N; i++) {
      idx(i) = N-i-1;
    }

    (tov = remap<0>(tiv, idx)).run(exec);
    exec.sync();

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(N-i-1,j));
      }
    }
    
    (tov = remap<1>(tiv, idx)).run(exec);
    exec.sync();

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i, N-j-1));
      }
    }
    
    (tov = remap<0,1>(tiv, idx, idx)).run(exec);
    exec.sync();

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(N-i-1, N-j-1));
      }
    }
  }
  
  { // Reverse lvalue test
    
    auto tov = make_tensor<TestType>({N,N});
    auto idx = make_tensor<int>({N});
    
    for(int i = 0; i < N; i++) {
      idx(i) = N-i-1;
    }

    (remap<0>(tov, idx) = tiv).run(exec);
    exec.sync();

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(N-i-1,j));
      }
    }
    
    (remap<1>(tov, idx) = tiv).run(exec);
    exec.sync();

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i, N-j-1));
      }
    }
    
    (remap<0,1>(tov, idx, idx) = tiv).run(exec);
    exec.sync();

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(N-i-1, N-j-1));
      }
    }
  }
  
  { // Even test
    int M = N/2;
    auto idx = make_tensor<int>({M});
    
    for(int i = 0; i < M; i++) {
      idx(i) = i*2;
    }

    {
      auto tov = make_tensor<TestType>({M, N});

      (tov = remap<0>(tiv, idx)).run(exec);
      exec.sync();

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < N ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i*2,j));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({N, M});

      (tov = remap<1>(tiv, idx)).run(exec);
      exec.sync();

      for( int i = 0; i < N ; i++) {
        for( int j = 0; j < M ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i,j*2));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({M, M});

      (tov = remap<0,1>(tiv, idx, idx)).run(exec);
      exec.sync();

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < M ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i*2,j*2));
	}
      }
    }
  }
  
  { // Braodcast test
    int M = N*2;
    auto idx = make_tensor<int>({M});
    
    for(int i = 0; i < M; i++) {
      idx(i) = 1;
    }

    {
      auto tov = make_tensor<TestType>({M, N});

      (tov = remap<0>(tiv, idx)).run(exec);
      exec.sync();

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < N ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(1,j));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({N, M});

      (tov = remap<1>(tiv, idx)).run(exec);
      exec.sync();

      for( int i = 0; i < N ; i++) {
        for( int j = 0; j < M ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i,1));
        }
      }
    }
  }

  { // Advanced test
    int M = N*2;
    auto idx = make_tensor<int>({M});
    
    for(int i = 0; i < M; i++) {
      idx(i) = i/4;
    }

    {
      auto tov = make_tensor<TestType>({M, N});

      (tov = remap<0>(tiv, idx)).run(exec);
      exec.sync();

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < N ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i/4,j));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({N, M});

      (tov = remap<1>(tiv, idx)).run(exec);
      exec.sync();

      for( int i = 0; i < N ; i++) {
        for( int j = 0; j < M ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i,j/4));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({M, M});

      (tov = remap<0,1>(tiv, idx, idx)).run(exec);
      exec.sync();

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < M ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i/4,j/4));
        }
      }
    }
  }

  {
    // Remap as both LHS and RHS
    auto in = make_tensor<TestType>({4,4,4});
    auto out = make_tensor<TestType>({4,4,4});
    TestType c = GenerateData<TestType>();
    for (int i = 0; i < in.Size(0); i++){
      for (int j = 0; j < in.Size(1); j++){
        for (int k = 0; k < in.Size(2); k++){
          in(i,j,k) = c;
        }
      }
    }

    auto map1 = matx::make_tensor<int>({2});
    auto map2 = matx::make_tensor<int>({2});
    map1(0) = 1;
    map1(1) = 2;
    map2(0) = 0;
    map2(1) = 1;

    (out = static_cast<TestType>(0)).run(exec);
    (matx::remap<2>(out, map2) = matx::remap<2>(in, map1)).run(exec);
    exec.sync();

    for (int i = 0; i < in.Size(0); i++){
      for (int j = 0; j < in.Size(1); j++){
        for (int k = 0; k < in.Size(2); k++){
          if (k > 1) {
            ASSERT_EQ(out(i,j,k), (TestType)0);
          }
          else {
            ASSERT_EQ(out(i,j,k), in(i,j,k));
          }
        }
      }
    } 
  }

  MATX_EXIT_HANDLER();
} 