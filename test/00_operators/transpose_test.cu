#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsNumericAllExecs, Transpose3D)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};     

  index_t num_rows = 5998;
  index_t num_cols = 64;

  tensor_t<TestType, 3> t3 ({1, num_rows, num_cols});
  tensor_t<TestType, 3> t3t({1, num_cols, num_rows});

  for (index_t i = 0; i < num_rows; i++) {
    for (index_t j = 0; j < num_cols; j++) {
       t3(0, i, j) = static_cast<detail::value_promote_t<TestType>>(i * num_cols + j);
    }
  }

  (t3t = transpose_matrix(t3)).run(exec);
  exec.sync();

  if constexpr (std::is_same_v<ExecType,cudaExecutor>) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  for (index_t i = 0; i < num_rows; i++) {
    for (index_t j = 0; j < num_cols; j++) {
        EXPECT_EQ(t3(0, i, j), t3t(0, j, i));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, TransposeVsTransposeMatrix)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  // example-begin transpose-test-1
  // ExecType is an executor type (e.g. matx::cudaExecutor for executing on the GPU).
  ExecType exec{};

  const index_t m = 3;
  const index_t n = 5;
  const index_t p = 7;

  // TestType is the tensor data type
  tensor_t<TestType, 3> t3  ({m,n,p});
  tensor_t<TestType, 3> t3t ({p,n,m});
  tensor_t<TestType, 3> t3tm({m,p,n});

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      for (index_t k = 0; k < p; k++) {
        t3(i, j, k) = static_cast<detail::value_promote_t<TestType>>(i*n*p + j*p + k);
      }
    }
  }

  (t3t = transpose(t3)).run(exec);
  (t3tm = transpose_matrix(t3)).run(exec);

  exec.sync();
  if constexpr (std::is_same_v<ExecType,cudaExecutor>) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      for (index_t k = 0; k < p; k++) {
        // transpose() permutes all dimensions whereas transpose_matrix() only permutes the
        // last two dimensions.
        EXPECT_EQ(t3(i,j,k), t3t(k,j,i));
        EXPECT_EQ(t3(i,j,k), t3tm(i,k,j));
      }
    }
  }
  // example-end transpose-test-1

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplexTypesAllExecs, HermitianTranspose)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  index_t count0 = 100;
  index_t count1 = 200;
  tensor_t<TestType, 2> t2({count0, count1});
  tensor_t<TestType, 2> t2s({count1, count0});
  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      TestType tmp = {(float)i, (float)-j};
      t2(i, j) = tmp;
    }
  }

  // example-begin hermitianT-test-1
  (t2s = hermitianT(t2)).run(exec);
  // example-end hermitianT-test-1
  exec.sync();

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(static_cast<double>(t2s(j, i).real()),
                                     static_cast<double>(t2(i, j).real())));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(-static_cast<double>(t2s(j, i).imag()),
                                     static_cast<double>(t2(i, j).imag())));
    }
  }
  MATX_EXIT_HANDLER();
}
