#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, CloneOp)
{
  constexpr int N = 10;
  constexpr int M = 12;
  constexpr int K = 14;

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  MATX_ENTER_HANDLER();
  { // clone from 0D
    // example-begin clone-test-1
    auto tiv = make_tensor<TestType>({});
    auto tov = make_tensor<TestType>({N,M,K});

    tiv() = 3;

    // Clone "tiv" from a 0D tensor to a 3D tensor
    auto op = clone<3>(tiv, {N, M, K});
    // example-end clone-test-1

    ASSERT_EQ(op.Size(0), N);
    ASSERT_EQ(op.Size(1), M);
    ASSERT_EQ(op.Size(2), K);

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(op(n,m,k) , tiv());
        }
      }
    }

    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(tov(n,m,k) , tiv());
        }
      }
    }
  }    

  { // clone from 1D
    // example-begin clone-test-2
    auto tiv = make_tensor<TestType>({K});
    auto tov = make_tensor<TestType>({N,M,K});

    for(int k = 0; k < K; k++) {
      tiv(k) = static_cast<typename inner_op_type_t<TestType>::type>(k);
    }

    // Clone "tiv" from a 1D tensor to a 3D tensor
    // matxKeepDim is used to indicate where the 1D tensor should be placed in the 3D tensor
    auto op = clone<3>(tiv, {N, M, matxKeepDim});
    // example-end clone-test-2

    ASSERT_EQ(op.Size(0), N);
    ASSERT_EQ(op.Size(1), M);
    ASSERT_EQ(op.Size(2), K);


    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(op(n,m,k) , tiv(k));
        }
      }
    }

    (tov = op).run(exec);
    exec.sync();
    print(op);
    print(tov);
    print(tiv);

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(tov(n,m,k) , tiv(k));
        }
      }
    }
  }    

  { // clone from 1D
    auto tiv = make_tensor<TestType>({M});
    auto tov = make_tensor<TestType>({N,M,K});

    for(int m = 0; m < K; m++) {
      tiv(m) = static_cast<typename inner_op_type_t<TestType>::type>(m);
    }

    auto op = clone<3>(tiv, {N, matxKeepDim, K});

    ASSERT_EQ(op.Size(0), N);
    ASSERT_EQ(op.Size(1), M);
    ASSERT_EQ(op.Size(2), K);


    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(op(n,m,k) , tiv(m));
        }
      }
    }

    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(tov(n,m,k) , tiv(m));
        }
      }
    }
  }    

  { // clone from 2D and operator
    auto tiv = make_tensor<TestType>({M,K});
    auto tov = make_tensor<TestType>({N,M,K});

    for(int m = 0; m < M; m++) {
      for(int k = 0; k < K; k++) {
        tiv(m,k) = static_cast<typename inner_op_type_t<TestType>::type>(m*K)+static_cast<typename inner_op_type_t<TestType>::type>(k);
      }
    }

    auto op = clone<3>(tiv, {N, matxKeepDim, matxKeepDim});

    ASSERT_EQ(op.Size(0), N);
    ASSERT_EQ(op.Size(1), M);
    ASSERT_EQ(op.Size(2), K);


    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(op(n,m,k) , tiv(m,k));
        }
      }
    }

    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(tov(n,m,k) , tiv(m,k));
        }
      }
    }
  }    

  { // clone from 2D
    auto tiv = make_tensor<TestType>({M,K});
    auto tov = make_tensor<TestType>({N,M,K});

    for(int m = 0; m < M; m++) {
      for(int k = 0; k < K; k++) {
        tiv(m,k) = static_cast<typename inner_op_type_t<TestType>::type>(m*K)+static_cast<typename inner_op_type_t<TestType>::type>(k);
      }
    }

    const auto op = clone<3>(static_cast<typename inner_op_type_t<TestType>::type>(2)*tiv, {N, matxKeepDim, matxKeepDim});

    ASSERT_EQ(op.Size(0), N);
    ASSERT_EQ(op.Size(1), M);
    ASSERT_EQ(op.Size(2), K);


    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(op(n,m,k), TestType(2)*tiv(m,k));
        }
      }
    }

    (tov = op).run(exec);
    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(tov(n,m,k) , TestType(2)*tiv(m,k));
        }
      }
    }
  }    

  if constexpr (is_cuda_executor_v<ExecType>)
  { // clone of a nested transform; conv2d currently only has a device executor
    auto tiv = make_tensor<TestType>({M,K});
    auto tov = make_tensor<TestType>({N,M,K});
    auto delta = make_tensor<TestType>({1,1});

    for(int m = 0; m < M; m++) {
      for(int k = 0; k < K; k++) {
        tiv(m,k) = static_cast<typename inner_op_type_t<TestType>::type>(m*K)+static_cast<typename inner_op_type_t<TestType>::type>(k);
      }
    }

    delta(0,0) = static_cast<typename inner_op_type_t<TestType>::type>(1.0);

    exec.sync();

    (tov = clone<3>(conv2d(tiv, delta, MATX_C_MODE_SAME), {N, matxKeepDim, matxKeepDim})).run(exec);

    exec.sync();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(tov(n,m,k) , tiv(m,k));
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
} 