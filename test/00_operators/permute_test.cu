#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsAllExecs, PermuteOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto A = make_tensor<TestType>({10,20,30});
  for(int i=0; i < A.Size(0); i++) {
    for(int j=0; j < A.Size(1); j++) {
      for(int k=0; k < A.Size(2); k++) {
        A(i,j,k) = static_cast<typename inner_op_type_t<TestType>::type>( i * A.Size(1)*A.Size(2) +
         j * A.Size(2) + k);  
      }
    }
  }

  // example-begin permute-test-1
  // Permute from dims {0, 1, 2} to {2, 0, 1}
  auto op = permute(A, {2, 0, 1});
  // example-end permute-test-1
  auto At = A.Permute({2, 0, 1});

  ASSERT_TRUE(op.Size(0) == A.Size(2));
  ASSERT_TRUE(op.Size(1) == A.Size(0));
  ASSERT_TRUE(op.Size(2) == A.Size(1));
  
  ASSERT_TRUE(op.Size(0) == At.Size(0));
  ASSERT_TRUE(op.Size(1) == At.Size(1));
  ASSERT_TRUE(op.Size(2) == At.Size(2));

  for(int i=0; i < op.Size(0); i++) {
    for(int j=0; j < op.Size(1); j++) {
      for(int k=0; k < op.Size(2); k++) {
        ASSERT_TRUE( op(i,j,k) == A(j,k,i));  
        ASSERT_TRUE( op(i,j,k) == At(i,j,k));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// Invalid permutations are rejected in every build mode, for both tensors (strided views)
// and expressions (PermuteOp)
TEST(OperatorValidationTests, PermuteInvalidDims)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto A = make_tensor<TestType>({2, 3, 4});
  auto expr = reverse<0>(A);

  EXPECT_THROW(permute(A, {0, 1, 3}), matx::detail::matxException);
  EXPECT_THROW(permute(A, {0, -1, 2}), matx::detail::matxException);
  EXPECT_THROW(permute(A, {0, 1, 1}), matx::detail::matxException);
  EXPECT_NO_THROW(permute(A, {1, 2, 0}));

  EXPECT_THROW(permute(expr, {0, 1, 3}), matx::detail::matxException);
  EXPECT_THROW(permute(expr, {0, -1, 2}), matx::detail::matxException);
  EXPECT_THROW(permute(expr, {0, 1, 1}), matx::detail::matxException);
  EXPECT_NO_THROW(permute(expr, {1, 2, 0}));

  // Both paths report the same error code
  auto error_of = [](auto &&f) {
    try { f(); } catch (const matx::detail::matxException &ex) { return ex.e; }
    return matxSuccess;
  };
  EXPECT_EQ(error_of([&] { permute(A, {0, 1, 1}); }), matxInvalidDim);
  EXPECT_EQ(error_of([&] { permute(expr, {0, 1, 1}); }), matxInvalidDim);

  MATX_EXIT_HANDLER();
}
TYPED_TEST(OperatorTestsNumericAllExecsWithoutJIT, NestedPermuteOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};
  auto a = make_tensor<TestType>({2, 3, 4});
  for (index_t i = 0; i < 2; ++i) {
    for (index_t j = 0; j < 3; ++j) {
      for (index_t k = 0; k < 4; ++k) {
        a(i, j, k) = static_cast<TestType>(i + 2 * j + 3 * k);
      }
    }
  }
  const cuda::std::array<int32_t, 3> outer{2, 1, 0};
  auto combined = permute(permute(a + a, {2, 0, 1}), outer);
  EXPECT_EQ(combined.Size(0), 3);
  EXPECT_EQ(combined.Size(1), 2);
  EXPECT_EQ(combined.Size(2), 4);
  auto out = make_tensor<TestType>({3, 2, 4});
  (out = combined).run(exec);
  exec.sync();
  for (index_t i = 0; i < 3; ++i) {
    for (index_t j = 0; j < 2; ++j) {
      for (index_t k = 0; k < 4; ++k) {
        EXPECT_EQ(out(i, j, k), static_cast<TestType>(2 * (j + 2 * i + 3 * k)));
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TEST(OperatorValidationTests, NestedPermuteInvalidDims)
{
  auto a = make_tensor<float>({2, 3, 4});
  auto inner = permute(a + a, {2, 0, 1});
  EXPECT_THROW(permute(inner, {0, 1, 3}), detail::matxException);
  EXPECT_THROW(permute(inner, {0, -1, 2}), detail::matxException);
  EXPECT_THROW(permute(inner, {0, 1, 1}), detail::matxException);
}
