#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsAllExecs, ReshapeOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  // example-begin reshape-test-1
  auto A = make_tensor<TestType>({2*4*8*16});
  for(int i = 0; i < A.Size(0); i++) {
    A(i) = static_cast<typename inner_op_type_t<TestType>::type>(i);
  }

  // op is a 4D operator
  auto op = reshape(A, {2, 4, 8, 16});

  // op2 is a 1D operator
  auto op2 = reshape(op, {2 * 4 * 8 * 16});
  // example-end reshape-test-1

  ASSERT_TRUE(op.Rank() == 4);
  ASSERT_TRUE(op2.Rank() == 1);

  ASSERT_TRUE(op.Size(0) == 2 );
  ASSERT_TRUE(op.Size(1) == 4 );
  ASSERT_TRUE(op.Size(2) == 8 );
  ASSERT_TRUE(op.Size(3) == 16 );
  
  ASSERT_TRUE(op2.Size(0) == A.TotalSize() );

  int idx = 0;
  for(int i=0; i < op.Size(0); i++) {
    for(int j=0; j < op.Size(1); j++) {
      for(int k=0; k < op.Size(2); k++) {
        for(int l=0; l < op.Size(3); l++) {
          ASSERT_TRUE( A(idx) == op(i,j,k,l) );
          ASSERT_TRUE( A(idx) == op2(idx));
          idx++;
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(OperatorTestsNumericAllExecs, Reshape)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  index_t count = 10;
  tensor_t<TestType, 4> t4({count, count, count, count});
  tensor_t<TestType, 1> t1({count * count * count * count});

  for (index_t i = 0; i < t4.Size(0); i++) {
    for (index_t j = 0; j < t4.Size(1); j++) {
      for (index_t k = 0; k < t4.Size(2); k++) {
        for (index_t l = 0; l < t4.Size(3); l++) {
          t4(i, j, k, l) =
              static_cast<detail::value_promote_t<TestType>>(i + j + k + l);
          t1(l + k * t4.Size(3) + j * t4.Size(3) * t4.Size(2) +
             i * t4.Size(3) * t4.Size(2) * t4.Size(1)) =
              static_cast<detail::value_promote_t<TestType>>(i + j + k + l);
        }
      }
    }
  }

  // Drop to a single dimension of same original total size
  auto rsv1 = t4.View({count * count * count * count});
  for (index_t i = 0; i < t4.Size(0); i++) {
    for (index_t j = 0; j < t4.Size(1); j++) {
      for (index_t k = 0; k < t4.Size(2); k++) {
        for (index_t l = 0; l < t4.Size(3); l++) {
          MATX_ASSERT_EQ(rsv1(l + k * t4.Size(3) + j * t4.Size(3) * t4.Size(2) +
                              i * t4.Size(3) * t4.Size(2) * t4.Size(1)),
                         (TestType)(i + j + k + (double)l));
        }
      }
    }
  }

  // Drop to 2D with a subset of the original size
  auto rsv2 = t4.View({2, 2});
  for (index_t i = 0; i < rsv2.Size(0); i++) {
    for (index_t j = 0; j < rsv2.Size(1); j++) {
      MATX_ASSERT_EQ(rsv2(i, j), t4(0, 0, 0, i * rsv2.Size(1) + j));
    }
  }

  // Create a 4D tensor from the 1D
  auto rsv4 = t1.View({count, count, count, count});
  for (index_t i = 0; i < rsv4.Size(0); i++) {
    for (index_t j = 0; j < rsv4.Size(1); j++) {
      for (index_t k = 0; k < rsv4.Size(2); k++) {
        for (index_t l = 0; l < rsv4.Size(3); l++) {
          MATX_ASSERT_EQ(rsv4(i, j, k, l),
                         t1(l + k * t4.Size(3) + j * t4.Size(3) * t4.Size(2) +
                            i * t4.Size(3) * t4.Size(2) * t4.Size(1)));
        }
      }
    }
  }


  // Test if oversized views throw
#ifndef NDEBUG  
  try {
    t4.View({1000, 1000, 100});
    FAIL() << "Oversized views not throwing";
  } catch (detail::matxException &e) {}
#endif

  MATX_EXIT_HANDLER();
}