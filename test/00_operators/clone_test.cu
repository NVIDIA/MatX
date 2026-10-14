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

    if (jit_supported(conv2d(tiv, delta, MATX_C_MODE_SAME))) {
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
  }

  MATX_EXIT_HANDLER();
}

// Clones of expressions (which use CloneOp rather than a strided tensor view)
// with kept dimensions in leading, non-adjacent, and interleaved positions
TYPED_TEST(OperatorTestsNumericAllExecs, CloneOpKeepDimPatterns)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_t = typename inner_op_type_t<TestType>::type;

  ExecType exec{};

  auto t1 = make_tensor<TestType>({4});
  auto t2 = make_tensor<TestType>({3, 4});
  auto t3 = make_tensor<TestType>({2, 3, 4});
  for (index_t i = 0; i < 4; i++) t1(i) = static_cast<inner_t>(i);
  for (index_t i = 0; i < 3; i++)
    for (index_t j = 0; j < 4; j++) t2(i, j) = static_cast<inner_t>(i * 4 + j);
  for (index_t i = 0; i < 2; i++)
    for (index_t j = 0; j < 3; j++)
      for (index_t k = 0; k < 4; k++) t3(i, j, k) = static_cast<inner_t>(i * 12 + j * 4 + k);
  const auto two = static_cast<inner_t>(2);

  { // rank 2 -> 3, kept dims {0, 2}
    auto out = make_tensor<TestType>({3, 5, 4});
    (out = clone<3>(two * t2, {matxKeepDim, 5, matxKeepDim})).run(exec);
    exec.sync();
    for (index_t a = 0; a < 3; a++)
      for (index_t b = 0; b < 5; b++)
        for (index_t c = 0; c < 4; c++)
          ASSERT_EQ(out(a, b, c), TestType(two) * t2(a, c));
  }

  { // rank 2 -> 4, kept dims {0, 3}
    auto out = make_tensor<TestType>({3, 2, 5, 4});
    (out = clone<4>(two * t2, {matxKeepDim, 2, 5, matxKeepDim})).run(exec);
    exec.sync();
    for (index_t a = 0; a < 3; a++)
      for (index_t b = 0; b < 2; b++)
        for (index_t c = 0; c < 5; c++)
          for (index_t d = 0; d < 4; d++)
            ASSERT_EQ(out(a, b, c, d), TestType(two) * t2(a, d));
  }

  { // rank 1 -> 4, kept dim {1}
    auto out = make_tensor<TestType>({2, 4, 3, 5});
    (out = clone<4>(two * t1, {2, matxKeepDim, 3, 5})).run(exec);
    exec.sync();
    for (index_t a = 0; a < 2; a++)
      for (index_t b = 0; b < 4; b++)
        for (index_t c = 0; c < 3; c++)
          for (index_t d = 0; d < 5; d++)
            ASSERT_EQ(out(a, b, c, d), TestType(two) * t1(b));
  }

  { // rank 3 -> 4, kept dims {0, 1, 3}
    auto out = make_tensor<TestType>({2, 3, 5, 4});
    (out = clone<4>(two * t3, {matxKeepDim, matxKeepDim, 5, matxKeepDim})).run(exec);
    exec.sync();
    for (index_t a = 0; a < 2; a++)
      for (index_t b = 0; b < 3; b++)
        for (index_t c = 0; c < 5; c++)
          for (index_t d = 0; d < 4; d++)
            ASSERT_EQ(out(a, b, c, d), TestType(two) * t3(a, b, d));
  }

  MATX_EXIT_HANDLER();
}

// The number of matxKeepDim entries must match the operator's rank in every build mode
TEST(OperatorValidationTests, CloneOpInvalidKeepDims)
{
  MATX_ENTER_HANDLER();
  using TestType = float;
  using inner_t = typename inner_op_type_t<TestType>::type;

  auto t2 = make_tensor<TestType>({3, 4});
  const auto two = static_cast<inner_t>(2);

  EXPECT_THROW(clone<3>(two * t2, {3, 5, matxKeepDim}), matx::detail::matxException);
  EXPECT_THROW(clone<3>(two * t2, {matxKeepDim, matxKeepDim, matxKeepDim}), matx::detail::matxException);
  EXPECT_THROW(clone<3>(two * t2, {3, 5, 4}), matx::detail::matxException);
  EXPECT_NO_THROW(clone<3>(two * t2, {matxKeepDim, 5, matxKeepDim}));

  // Tensors take a separate path (a strided view) with the same requirement
  EXPECT_THROW(clone<3>(t2, {3, 5, matxKeepDim}), matx::detail::matxException);
  EXPECT_THROW(clone<3>(t2, {matxKeepDim, matxKeepDim, matxKeepDim}), matx::detail::matxException);
  EXPECT_NO_THROW(clone<3>(t2, {matxKeepDim, 5, matxKeepDim}));

  // ...as does the tensor's Clone() member that clone() uses
  EXPECT_THROW(t2.template Clone<3>({3, 5, matxKeepDim}), matx::detail::matxException);
  EXPECT_NO_THROW(t2.template Clone<3>({matxKeepDim, 5, matxKeepDim}));

  // Both paths report the same error code
  auto error_of = [](auto &&f) {
    try { f(); } catch (const matx::detail::matxException &ex) { return ex.e; }
    return matxSuccess;
  };
  EXPECT_EQ(error_of([&] { clone<3>(two * t2, {3, 5, matxKeepDim}); }), matxInvalidDim);
  EXPECT_EQ(error_of([&] { clone<3>(t2, {3, 5, matxKeepDim}); }), matxInvalidDim);

  MATX_EXIT_HANDLER();
} 