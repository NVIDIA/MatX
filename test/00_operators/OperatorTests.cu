////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include "assert.h"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"
#include <type_traits>
#include <iostream>

using namespace matx;

template <typename TensorType>
class OperatorTestsComplex : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloat : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsNumeric : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsNumericNonComplex : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloatNonComplex : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloatNonComplexNonHalf : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsIntegral : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsBoolean : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloatHalf : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsNumericNoHalf : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsAll : public ::testing::Test {
};

template <typename TensorType>
class OperatorTestsNumericNonComplexAllExecs : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloatNonComplexNonHalfAllExecs : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsNumericNoHalfAllExecs : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsComplexNonHalfTypesAllExecs : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsComplexTypesAllExecs : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsAllExecs : public ::testing::Test {
};

template <typename TensorType>
class OperatorTestsFloatAllExecs : public ::testing::Test {
};

template <typename TensorType>
class OperatorTestsFloatNonComplexAllExecs : public ::testing::Test {
};

template <typename TensorType>
class OperatorTestsNumericAllExecs : public ::testing::Test {
};

template <typename TensorType>
class OperatorTestsIntegralAllExecs : public ::testing::Test {
};

template <typename TensorType>
class OperatorTestsBooleanAllExecs : public ::testing::Test {
};

TYPED_TEST_SUITE(OperatorTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(OperatorTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(OperatorTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(OperatorTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(OperatorTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(OperatorTestsNumericNonComplex,
                 MatXNumericNonComplexTypes);
TYPED_TEST_SUITE(OperatorTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexNonHalf,
                 MatXFloatNonComplexNonHalfTypes);
TYPED_TEST_SUITE(OperatorTestsBoolean, MatXBoolTypes);
TYPED_TEST_SUITE(OperatorTestsFloatHalf, MatXFloatHalfTypes);
TYPED_TEST_SUITE(OperatorTestsNumericNoHalf, MatXNumericNoHalfTypes);


TYPED_TEST_SUITE(OperatorTestsNumericNonComplexAllExecs,
                 MatXNumericNonComplexTypesAllExecs);  
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexNonHalfAllExecs,
                 MatXFloatNonComplexNonHalfTypesAllExecs);  
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexAllExecs,
                 MatXTypesFloatNonComplexAllExecs);
TYPED_TEST_SUITE(OperatorTestsNumericAllExecs,
                 MatXTypesNumericAllExecs);                                
TYPED_TEST_SUITE(OperatorTestsNumericNoHalfAllExecs, MatXNumericNoHalfTypesAllExecs);          
TYPED_TEST_SUITE(OperatorTestsComplexNonHalfTypesAllExecs, MatXComplexNonHalfTypesAllExecs);
TYPED_TEST_SUITE(OperatorTestsComplexTypesAllExecs, MatXComplexTypesAllExecs);
TYPED_TEST_SUITE(OperatorTestsAllExecs, MatXTypesAllExecs);
TYPED_TEST_SUITE(OperatorTestsFloatAllExecs, MatXTypesFloatAllExecs);
TYPED_TEST_SUITE(OperatorTestsIntegralAllExecs, MatXTypesIntegralAllExecs);
TYPED_TEST_SUITE(OperatorTestsBooleanAllExecs, MatXTypesBooleanAllExecs);

TYPED_TEST(OperatorTestsAllExecs, BaseOp)
{
  MATX_ENTER_HANDLER();

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto A = make_tensor<TestType>({10,20});
  auto op = A + A;

  EXPECT_TRUE(op.Size(0) == A.Size(0));
  EXPECT_TRUE(op.Size(1) == A.Size(1));

  auto shape = op.Shape();

  EXPECT_TRUE(shape[0] == A.Size(0));
  EXPECT_TRUE(shape[1] == A.Size(1));
 
  EXPECT_TRUE(A.TotalSize() == op.TotalSize());

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAllExecs, GetString)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto A = make_tensor<TestType>({10,20});
  auto B = make_tensor<TestType>({20});
  auto C = make_tensor<TestType>({10,20});

  auto op1 = C = A;
  auto op2 = C = A + B + (TestType)5;
  auto op3 = C = A / B;
  auto op4 = C = cos(A * B) + sin(C);
  auto op6 = (op1,op2,op3,op4);

  std::cout << "op1: " << op1.str() << std::endl;
  std::cout << "op2: " << op2.str() << std::endl;
  std::cout << "op3: " << op3.str() << std::endl;
  std::cout << "op4: " << op4.str() << std::endl;
  std::cout << "op6: " << op6.str() << std::endl;

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAllExecs, PermuteOp)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

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

  auto op = permute(A, {2, 0, 1});
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

TYPED_TEST(OperatorTestsAllExecs, ReshapeOp)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  auto A = make_tensor<TestType>({2*4*8*16});
  for(int i = 0; i < A.Size(0); i++) {
    A(i) = static_cast<typename inner_op_type_t<TestType>::type>(i);
  }

  auto op = reshape(A, {2, 4, 8, 16});
  auto op2 = reshape(op, {2*4*8*16});

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

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, FMod)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tiv1;
  tensor_t<TestType, 0> tov0;

  tiv0() = (TestType)5.0;
  tiv1() = (TestType)3.1;
  (tov0 = fmod(tiv0, tiv1)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_fmod((TestType)5.0, (TestType)3.1)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatAllExecs, TrigFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0;

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  (tov0 = sin(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_sin(c)));

  (tov0 = cos(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_cos(c)));

  (tov0 = tan(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_tan(c)));

  (tov0 = asin(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_asin(c)));

  (tov0 = acos(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_acos(c)));

  (tov0 = atan(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_atan(c)));

  (tov0 = sinh(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_sinh(c)));

  (tov0 = cosh(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_cosh(c)));

  (tov0 = tanh(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_tanh(c)));

  (tov0 = asinh(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_asinh(c)));

  (tov0 = acosh(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_acosh(c)));

  (tov0 = atanh(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_atanh(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplexTypesAllExecs, AngleOp)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  tensor_t<TestType, 0> tiv0;
  tensor_t<typename TestType::value_type, 0> tov0;

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  (tov0 = angle(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_angle(c)));  

  MATX_EXIT_HANDLER();
}
TYPED_TEST(OperatorTestsNumericAllExecs, CloneOp)
{
  constexpr int N = 10;
  constexpr int M = 12;
  constexpr int K = 14;

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   

  MATX_ENTER_HANDLER();
  { // clone from 0D
    auto tiv = make_tensor<TestType>();
    auto tov = make_tensor<TestType>({N,M,K});

    tiv() = 3;

    auto op = clone<3>(tiv, {N, M, K});

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
    cudaDeviceSynchronize();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(tov(n,m,k) , tiv());
        }
      }
    }
  }    

  { // clone from 1D
    auto tiv = make_tensor<TestType>({K});
    auto tov = make_tensor<TestType>({N,M,K});

    for(int k = 0; k < K; k++) {
      tiv(k) = static_cast<typename inner_op_type_t<TestType>::type>(k);
    }

    auto op = clone<3>(tiv, {N, M, matxKeepDim});

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
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();

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

    auto op = clone<3>(static_cast<typename inner_op_type_t<TestType>::type>(2)*tiv, {N, matxKeepDim, matxKeepDim});

    ASSERT_EQ(op.Size(0), N);
    ASSERT_EQ(op.Size(1), M);
    ASSERT_EQ(op.Size(2), K);


    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(op(n,m,k) , TestType(2)*tiv(m,k));
        }
      }
    }

    (tov = op).run(exec);
    cudaDeviceSynchronize();

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_EQ(tov(n,m,k) , TestType(2)*tiv(m,k));
        }
      }
    }
  }    

  MATX_EXIT_HANDLER();
}



TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, SliceStrideOp)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  tensor_t<TestType, 1> t1{{10}};

  t1.SetVals({10, 20, 30, 40, 50, 60, 70, 80, 90, 100});
  auto t1t = slice(t1, {0}, {matxEnd}, {2});
 
  for (index_t i = 0; i < t1.Size(0); i += 2) {
    ASSERT_EQ(t1(i), t1t(i / 2));
  }

  auto t1t2 = slice(t1, {2}, {matxEnd}, {2});

  for (index_t i = 0; i < t1t2.Size(0); i++) {
    ASSERT_EQ(TestType(30 + 20 * i), t1t2(i));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, SliceOp)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  tensor_t<TestType, 2> t2{{20, 10}};
  tensor_t<TestType, 3> t3{{30, 20, 10}};
  tensor_t<TestType, 4> t4{{40, 30, 20, 10}};

  (t2 = linspace<1>(t2.Shape(), (inner_type)0, (inner_type)10)).run(exec);
  (t3 = linspace<2>(t3.Shape(), (inner_type)0, (inner_type)10)).run(exec);
  (t4 = linspace<3>(t4.Shape(), (inner_type)0, (inner_type)10)).run(exec);
  cudaStreamSynchronize(0);

  auto t2t = slice(t2, {1, 2}, {3, 5});
  auto t3t = slice(t3, {1, 2, 3}, {3, 5, 7});
  auto t4t = slice(t4, {1, 2, 3, 4}, {3, 5, 7, 9});

  ASSERT_EQ(t2t.Size(0), 2);
  ASSERT_EQ(t2t.Size(1), 3);

  ASSERT_EQ(t3t.Size(0), 2);
  ASSERT_EQ(t3t.Size(1), 3);
  ASSERT_EQ(t3t.Size(2), 4);

  ASSERT_EQ(t4t.Size(0), 2);
  ASSERT_EQ(t4t.Size(1), 3);
  ASSERT_EQ(t4t.Size(2), 4);
  ASSERT_EQ(t4t.Size(3), 5);

  for (index_t i = 0; i < t2t.Size(0); i++) {
    for (index_t j = 0; j < t2t.Size(1); j++) {
      ASSERT_EQ(t2t(i, j), t2(i + 1, j + 2));
    }
  }

  for (index_t i = 0; i < t3t.Size(0); i++) {
    for (index_t j = 0; j < t3t.Size(1); j++) {
      for (index_t k = 0; k < t3t.Size(2); k++) {
        ASSERT_EQ(t3t(i, j, k), t3(i + 1, j + 2, k + 3));
      }
    }
  }

  for (index_t i = 0; i < t4t.Size(0); i++) {
    for (index_t j = 0; j < t4t.Size(1); j++) {
      for (index_t k = 0; k < t4t.Size(2); k++) {
        for (index_t l = 0; l < t4t.Size(3); l++) {
          ASSERT_EQ(t4t(i, j, k, l), t4(i + 1, j + 2, k + 3, l + 4));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, SliceAndReduceOp)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;
  ExecType exec{}; 

  tensor_t<TestType, 2> t2t{{20, 10}};
  tensor_t<TestType, 3> t3t{{30, 20, 10}};
  (t2t = linspace<1>(t2t.Shape(), (inner_type)0, (inner_type)10)).run(exec);
  (t3t = linspace<2>(t3t.Shape(), (inner_type)0, (inner_type)10)).run(exec);
  cudaStreamSynchronize(0);

  {
    index_t j = 0;
    auto t2sly = slice<1>(t2t, {0, j}, {matxEnd, matxDropDim});
    for (index_t i = 0; i < t2sly.Size(0); i++) {
      ASSERT_EQ(t2sly(i), t2t(i, j));
    }
  }

  {
    index_t i = 0;
    auto t2slx = slice<1>(t2t, {i, 0}, {matxDropDim, matxEnd});
    for (index_t j = 0; j < t2slx.Size(0); j++) {
      ASSERT_EQ(t2slx(j), t2t(i, j));
    }
  }

  {
    index_t j = 0;
    index_t k = 0;
    auto t3slz = slice<1>(t3t, {0, j, k}, {matxEnd, matxDropDim, matxDropDim});
    for (index_t i = 0; i < t3slz.Size(0); i++) {
      ASSERT_EQ(t3slz(i), t3t(i, j, k));
    }
  }

  {
    index_t i = 0;
    index_t k = 0;
    auto t3sly = slice<1>(t3t, {i, 0, k}, {matxDropDim, matxEnd, matxDropDim});
    for (index_t j = 0; j < t3sly.Size(0); j++) {
      ASSERT_EQ(t3sly(j), t3t(i, j, k));
    }
  }

  {
    index_t i = 0;
    index_t j = 0;
    auto t3slx = slice<1>(t3t, {i, j, 0}, {matxDropDim, matxDropDim, matxEnd});
    for (index_t k = 0; k < t3slx.Size(0); k++) {
      ASSERT_EQ(t3slx(k), t3t(i, j, k));
    }
  }

  {
    index_t k = 0;
    auto t3slzy = slice<2>(t3t, {0, 0, k}, {matxEnd, matxEnd, matxDropDim});
    for (index_t i = 0; i < t3slzy.Size(0); i++) {
      for (index_t j = 0; j < t3slzy.Size(1); j++) {
        ASSERT_EQ(t3slzy(i, j), t3t(i, j, k));
      }
    }
  }

  {
    index_t j = 0;
    auto t3slzx = slice<2>(t3t, {0, j, 0}, {matxEnd, matxDropDim, matxEnd});
    for (index_t i = 0; i < t3slzx.Size(0); i++) {
      for (index_t k = 0; k < t3slzx.Size(1); k++) {
        ASSERT_EQ(t3slzx(i, k), t3t(i, j, k));
      }
    }
  }

  {
    index_t i = 0;
    auto t3slyx = slice<2>(t3t, {i, 0, 0}, {matxDropDim, matxEnd, matxEnd});
    for (index_t j = 0; j < t3slyx.Size(0); j++) {
      for (index_t k = 0; k < t3slyx.Size(1); k++) {
        ASSERT_EQ(t3slyx(j, k), t3t(i, j, k));
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, CollapseOp)
{
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
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
  
    auto op = rcollapse<2>(tiv);

    EXPECT_TRUE(op.Rank() == 2);
    EXPECT_TRUE(op.Size(0) == N);
    EXPECT_TRUE(op.Size(1) == M*K);

    (tov = (TestType)0).run(exec);
    (tov = op).run(exec);
    cudaStreamSynchronize(0);

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
  
    auto op = lcollapse<2>(tiv);

    EXPECT_TRUE(op.Rank() == 2);
    EXPECT_TRUE(op.Size(0) == N*M);
    EXPECT_TRUE(op.Size(1) == K);
    
    
    (tov = (TestType)0).run(exec);
    (tov = op).run(exec);
    cudaStreamSynchronize(0);

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
    cudaStreamSynchronize(0);

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
    cudaStreamSynchronize(0);

    for(int n = 0; n < N; n++) {
      for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
          ASSERT_TRUE(tiv(n,m,k) == tov(n*M*K+m*K+k));
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(OperatorTestsNumericAllExecs, RemapOp)
{
  int N = 10;

  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  auto tiv = make_tensor<TestType>({N,N});

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      tiv(i,j) = inner_type(i*N+j);
    }
  }

  { // Identity Gather test

    auto tov = make_tensor<TestType>({N, N});
    auto idx = make_tensor<int>({N});
    
    for(int i = 0; i < N; i++) {
      idx(i) = i;
    }

    (tov = remap<0>(tiv, idx)).run(exec);
    cudaStreamSynchronize(0);

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }
    
    (tov = remap<1>(tiv, idx)).run(exec);
    cudaStreamSynchronize(0);
    
    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }

    (tov = remap<0,1>(tiv, idx, idx)).run(exec);
    cudaStreamSynchronize(0);
    
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
    cudaStreamSynchronize(0);

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }
    
    (tov = (TestType)0).run(exec);
    (remap<1>(tov, idx) = tiv).run(exec);
    cudaStreamSynchronize(0);
    
    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i,j));
      }
    }
    
    (tov = (TestType)0).run(exec);
    (remap<0,1>(tov, idx, idx) = tiv).run(exec);
    cudaStreamSynchronize(0);
    
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
    cudaStreamSynchronize(0);

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(N-i-1,j));
      }
    }
    
    (tov = remap<1>(tiv, idx)).run(exec);
    cudaStreamSynchronize(0);

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i, N-j-1));
      }
    }
    
    (tov = remap<0,1>(tiv, idx, idx)).run(exec);
    cudaStreamSynchronize(0);

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
    cudaStreamSynchronize(0);

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(N-i-1,j));
      }
    }
    
    (remap<1>(tov, idx) = tiv).run(exec);
    cudaStreamSynchronize(0);

    for( int i = 0; i < N ; i++) {
      for( int j = 0; j < N ; j++) {
        EXPECT_TRUE(tov(i,j) == tiv(i, N-j-1));
      }
    }
    
    (remap<0,1>(tov, idx, idx) = tiv).run(exec);
    cudaStreamSynchronize(0);

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
      cudaStreamSynchronize(0);

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < N ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i*2,j));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({N, M});

      (tov = remap<1>(tiv, idx)).run(exec);
      cudaStreamSynchronize(0);

      for( int i = 0; i < N ; i++) {
        for( int j = 0; j < M ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i,j*2));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({M, M});

      (tov = remap<0,1>(tiv, idx, idx)).run(exec);
      cudaStreamSynchronize(0);

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
      cudaStreamSynchronize(0);

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < N ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(1,j));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({N, M});

      (tov = remap<1>(tiv, idx)).run(exec);
      cudaStreamSynchronize(0);

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
      cudaStreamSynchronize(0);

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < N ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i/4,j));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({N, M});

      (tov = remap<1>(tiv, idx)).run(exec);
      cudaStreamSynchronize(0);

      for( int i = 0; i < N ; i++) {
        for( int j = 0; j < M ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i,j/4));
        }
      }
    }
    
    {
      auto tov = make_tensor<TestType>({M, M});

      (tov = remap<0,1>(tiv, idx, idx)).run(exec);
      cudaStreamSynchronize(0);

      for( int i = 0; i < M ; i++) {
        for( int j = 0; j < M ; j++) {
          EXPECT_TRUE(tov(i,j) == tiv(i/4,j/4));
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, RemapRankZero)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto sync = [&exec]() constexpr {
    if constexpr (std::is_same_v<ExecType,cudaExecutor>) {
      cudaDeviceSynchronize();
    }
  };

  const int N = 16;

  // 1D source tensor cases
  {
    auto from = make_tensor<int>({N});
    (from = range<0>({N}, 0, 1)).run(exec);
    sync();
    auto ind = make_tensor<int>();
    auto r = remap<0>(from, ind);
    auto to = make_tensor<int>({1});

    ind() = N/2;
    (to = r).run(exec);
    sync();

    ASSERT_EQ(to(0), N/2);

    ind() = N/4;
    (to = r).run(exec);
    sync();

    ASSERT_EQ(to(0), N/4);
  }

  // 2D source tensor cases
  {
    auto from = make_tensor<int>({N,N});
    (from = ones(from.Shape())).run(exec);
    sync();

    auto i0 = make_tensor<int>();
    auto i1 = make_tensor<int>();
    auto r0 = remap<0>(from, i0);
    auto r1 = remap<1>(from, i0);

    auto to0 = make_tensor<int>({1,N});
    auto to1 = make_tensor<int>({N,1});

    i0() = N/2;
    from(N/2,0) = 2;
    from(0,N/2) = 3;
    (to0 = r0).run(exec);
    (to1 = r1).run(exec);
    sync();

    ASSERT_EQ(to0(0,0), 2);
    ASSERT_EQ(to0(0,1), 1);
    ASSERT_EQ(to1(0,0), 3);
    ASSERT_EQ(to1(1,0), 1);

    i0() = N/3;
    i1() = 2*(N/3);
    from(N/3, 2*(N/3)) = 11;

    // Select a single entry from the 2D input tensor
    auto entry = make_tensor<int>({1,1});
    (entry = remap<0,1>(from, i0, i1)).run(exec);
    sync();

    ASSERT_EQ(entry(0,0), 11);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplexTypesAllExecs, RealImagOp)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  tensor_t<TestType, 0> tiv0;
  tensor_t<typename TestType::value_type, 0> tov0;

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  (tov0 = real(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c.real()));  

  (tov0 = imag(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c.imag()));   

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0;

  TestType c = GenerateData<TestType>();
  TestType d = c;
  TestType z = 0;
  tiv0() = c;

  tensor_t<TestType, 0> tov00;

  IFELSE(tiv0 == d, tov0 = z, tov0 = d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), z));

  IFELSE(tiv0 == d, tov0 = tiv0, tov0 = d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), tiv0()));

  IFELSE(tiv0 != d, tov0 = d, tov0 = z).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), z));

  (tov0 = c, tov00 = c).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c));
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov00(), c));  

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, OperatorFuncsR2C)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  tensor_t<TestType, 0> tiv0;
  tensor_t<typename detail::complex_from_scalar_t<TestType>, 0> tov0;
  TestType c = GenerateData<TestType>();
  tiv0() = c;

  (tov0 = expj(tiv0)).run(exec);
  cudaStreamSynchronize(0);

  EXPECT_TRUE(MatXUtils::MatXTypeCompare(
      tov0(),
      typename detail::complex_from_scalar_t<TestType>(detail::_internal_cos(tiv0()), detail::_internal_sin(tiv0()))));  
  MATX_EXIT_HANDLER();      
}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};    
  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0;

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  (tov0 = log10(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_log10(c)));

  (tov0 = log(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_log(c)));

  (tov0 = log2(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_log2(c)));

  (tov0 = floor(tiv0)).run(exec);   
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_floor(c)));

  (tov0 = ceil(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_ceil(c)));

  (tov0 = round(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_round(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, NDOperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   

  auto a = make_tensor<TestType>({1,2,3,4,5});
  auto b = make_tensor<TestType>({1,2,3,4,5});
  (a = ones<TestType>(a.Shape())).run(exec);
  cudaDeviceSynchronize();
  (b = ones<TestType>(b.Shape())).run(exec);
  cudaDeviceSynchronize();
  (a = a + b).run(exec);

  auto t0 = make_tensor<TestType>();
  sum(t0, a, exec);
  cudaStreamSynchronize(0);
  ASSERT_EQ(t0(), static_cast<TestType>(2 * a.TotalSize()));
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericNonComplexAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};     
  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0;

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  TestType d = c + 1;

  (tov0 = max(tiv0, d)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), max(c, d)));

  (tov0 = min(tiv0, d)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), min(c, d)));

  // These operators convert type T into bool
  tensor_t<bool, 0> tob;

  (tob = tiv0 < d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c < d));

  (tob = tiv0 > d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c > d));

  (tob = tiv0 <= d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c <= d));

  (tob = tiv0 >= d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c >= d));

  (tob = tiv0 == d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c == d));

  (tob = tiv0 != d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c != d));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplexTypesAllExecs, OperatorFuncDivComplex)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};  
  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0; 
  typename TestType::value_type s = 5.0;

  TestType c = GenerateData<TestType>();  
  tiv0() = c;

  (tov0 = s / tiv0).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), s / tiv0()));

  MATX_EXIT_HANDLER();  
}

TYPED_TEST(OperatorTestsNumericAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};    
  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0;

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  (tov0 = tiv0 + tiv0).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c + c));

  (tov0 = tiv0 - tiv0).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c - c));

  (tov0 = tiv0 * tiv0).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c * c));

  (tov0 = tiv0 / tiv0).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c / c));

  (tov0 = -tiv0).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), -c));

  IF(tiv0 == tiv0, tov0 = c).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c));

  TestType p = 2.0f;
  (tov0 = as_type<TestType>(pow(tiv0, p))).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_pow(c, p)));

  TestType three = 3.0f;

  (tov0 = tiv0 * tiv0 * (tiv0 + tiv0) / tiv0 + three).run();
  cudaStreamSynchronize(0);

  TestType res;
  res = c * c * (c + c) / c + three;
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), res, 0.07));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsIntegralAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};    
  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0;

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  TestType mod = 2;

  (tov0 = tiv0 % mod).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c % mod));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsBooleanAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0;

  TestType c = GenerateData<TestType>();
  TestType d = false;
  tiv0() = c;

  (tov0 = tiv0 && d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c && d));

  (tov0 = tiv0 || d).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c || d));

  (tov0 = !tiv0).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), !c));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplexTypesAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   

  tensor_t<TestType, 0> tiv0;
  tensor_t<TestType, 0> tov0;

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  (tov0 = exp(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_exp(c)));

  (tov0 = conj(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_conj(c)));

  // abs and norm take a complex and output a floating point value
  tensor_t<typename TestType::value_type, 0> tdd0;
  (tdd0 = norm(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tdd0(), detail::_internal_norm(c)));

  (tdd0 = abs(tiv0)).run(exec);
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tdd0(), detail::_internal_abs(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAllExecs, Flatten)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};  

  auto t2 = make_tensor<TestType>({10, 2});
  auto val = GenerateData<TestType>();

  for (index_t i = 0; i < t2.Size(0); i++) {
    for (index_t j = 0; j < t2.Size(1); j++) {
      t2(i,j) = val;
    }
  }

  auto t1 = make_tensor<TestType>({t2.Size(0)*t2.Size(1)});
  (t1 = flatten(t2)).run(exec);
  cudaStreamSynchronize(0);
  
  for (index_t i = 0; i < t2.Size(0)*t2.Size(1); i++) {
    ASSERT_EQ(t1(i), val);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericNoHalfAllExecs, AdvancedOperators)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  index_t count = 100;

  tensor_t<TestType, 1> a({count});
  tensor_t<TestType, 1> b({count});
  tensor_t<TestType, 1> c({count});

  for (index_t i = 0; i < count; i++) {
    a(i) = static_cast<detail::value_promote_t<TestType>>(i);
    b(i) = static_cast<detail::value_promote_t<TestType>>(i + 100);
  }

  {
    (c = a + b).run(exec);

    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt + (tcnt + (TestType)100)));
    }
  }

  {
    (c = a * b).run(exec);

    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt * (tcnt + (TestType)100)));
    }
  }

  {
    (c = a * b + a).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TestType)100) + tcnt));
    }
  }

  {

    (c = a * b + a * (TestType)4.0f).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TestType)100.0f) + tcnt * (TestType)4));
    }
  }
  MATX_EXIT_HANDLER();
}



TYPED_TEST(OperatorTestsNumericNonComplexAllExecs, AdvancedOperators)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  index_t count = 10;

  tensor_t<TestType, 1> a({count});
  tensor_t<TestType, 1> b({count});
  tensor_t<TestType, 1> c({count});

  for (index_t i = 0; i < count; i++) {
    a(i) = (TestType)i;
    b(i) = (TestType)(i + 2);
  }

  {
    (c = a + b).run(exec);

    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = (TestType)i;
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(c(i), (TestType)((float)tcnt + ((float)tcnt + 2.0f))));
    }
  }

  {
    (c = a * b).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = (TestType)i;
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), (float)tcnt * ((float)tcnt + 2.0f)));
    }
  }

  {
    (c = a * b + a).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = (TestType)i;
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), (float)tcnt * ((float)tcnt + 2.0f) + (float)tcnt));
    }
  }

  {

    (c = a * b + a * (TestType)2.0f).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TestType tcnt = (TestType)i;
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), (float)tcnt * ((float)tcnt + 2.0f) + (float)tcnt * 2.0f));
    }
  }
  MATX_EXIT_HANDLER();
}


// Testing 4 basic arithmetic operations with complex numbers and non-complex
TYPED_TEST(OperatorTestsComplexTypesAllExecs, ComplexTypeCompatibility)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  index_t count = 10;

  tensor_t<float, 1> fview({count});
  tensor_t<TestType, 1> dview({count});

  using data_type =
      typename std::conditional_t<is_complex_half_v<TestType>, float,
                                  typename TestType::value_type>;

  // Multiply by scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }

  (dview = dview * fview).run(exec);
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              static_cast<detail::value_promote_t<TestType>>(i * i));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TestType>>(i * i));
  }

  // Divide by scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = i == 0 ? static_cast<float>(1) : static_cast<float>(i);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }

  (dview = dview / fview).run(exec);
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              i == 0 ? static_cast<detail::value_promote_t<TestType>>(0)
                     : static_cast<detail::value_promote_t<TestType>>(1));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              i == 0 ? static_cast<detail::value_promote_t<TestType>>(0)
                     : static_cast<detail::value_promote_t<TestType>>(1));
  }

  // Add scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }
  

  (dview = dview + fview).run(exec);
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              static_cast<detail::value_promote_t<TestType>>(i + i));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TestType>>(i));
  }

  // Subtract scalar from complex
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i + 1);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }

  (dview = dview - fview).run(exec);
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              static_cast<detail::value_promote_t<TestType>>(-1));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TestType>>(i));
  }

  // Subtract complex from scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i + 1);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }

  (dview = fview - dview).run(exec);
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              static_cast<detail::value_promote_t<TestType>>(1));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TestType>>(-i));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, SquareCopyTranspose)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  index_t count = 512;
  tensor_t<TestType, 2> t2({count, count});
  tensor_t<TestType, 2> t2t({count, count});

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i * count + j);
    }
  }

  t2.PrefetchDevice(0);
  t2t.PrefetchDevice(0);
  matx::copy(t2t, t2, exec);

  t2t.PrefetchHost(0);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2t(i, j),
                                             TestType(i * count + (double)j)));
    }
  }

  t2t.PrefetchDevice(0);
  transpose(t2t, t2, exec);

  t2t.PrefetchHost(0);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j),
                                             TestType(i * count + (double)j)));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2t(j, i), TestType(i * count + j)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, NonSquareTranspose)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  index_t count = 100;
  index_t count1 = 200, count2 = 100;
  tensor_t<TestType, 2> t2({count1, count2});
  tensor_t<TestType, 2> t2t({count2, count1});

  for (index_t i = 0; i < count1; i++) {
    for (index_t j = 0; j < count2; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i * count + j);
    }
  }

  transpose(t2t, t2, exec);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count1; i++) {
    for (index_t j = 0; j < count2; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j),
                                             TestType(i * count + (double)j)));
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2t(j, i),
                                             TestType(i * count + (double)j)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, Transpose3D)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

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

  transpose(t3t, t3, exec);
  cudaError_t error = cudaStreamSynchronize(0);
  ASSERT_EQ(error, cudaSuccess);

  for (index_t i = 0; i < num_rows; i++) {
    for (index_t j = 0; j < num_cols; j++) {
        EXPECT_EQ(t3(0, i, j), t3t(0, j, i));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, CloneAndAdd)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};  

  index_t numSamples = 8;
  index_t numPulses = 4;
  index_t numPairs = 2;
  index_t numBeams = 2;

  tensor_t<TestType, 4> beamwiseRangeDoppler(
      {numBeams, numPulses, numPairs, numSamples});
  tensor_t<TestType, 2> steeredMx({numBeams, numSamples});
  tensor_t<TestType, 3> velAccelHypoth({numPulses, numPairs, numSamples});

  for (index_t i = 0; i < numBeams; i++) {
    for (index_t j = 0; j < numSamples; j++) {
      steeredMx(i, j) = static_cast<TestType>((i + 1) * 10 + (j + 1));
    }
  }

  for (index_t i = 0; i < numPulses; i++) {
    for (index_t j = 0; j < numPairs; j++) {
      for (index_t k = 0; k < numSamples; k++) {
        velAccelHypoth(i, j, k) = static_cast<TestType>(
            (i + 1) * 10 + (j + 1) * 1 + (k + 1) * 1);
      }
    }
  }

  auto smx = 
     clone<4>(steeredMx, {matxKeepDim, numPulses, numPairs, matxKeepDim});
  auto vah = clone<4>(velAccelHypoth,
      {numBeams, matxKeepDim, matxKeepDim, matxKeepDim});

  (beamwiseRangeDoppler = smx + vah).run();

  cudaStreamSynchronize(0);
  for (index_t i = 0; i < numBeams; i++) {
    for (index_t j = 0; j < numPulses; j++) {
      for (index_t k = 0; k < numPairs; k++) {
        for (index_t l = 0; l < numSamples; l++) {
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(
              beamwiseRangeDoppler(i, j, k, l),
              steeredMx(i, l) + velAccelHypoth(j, k, l)));
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(
              beamwiseRangeDoppler(i, j, k, l),
              ((i + 1) * 10 + (l + 1)) // steeredMx
                  + ((j + 1) * 10 + (k + 1) * 1 +
                     (l + 1) * 1) // velAccelHypoth
              ));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, Reshape)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

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


TYPED_TEST(OperatorTestsNumericAllExecs, Broadcast)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  {
    tensor_t<TestType, 0> t0;
    tensor_t<TestType, 4> t4i({10, 20, 30, 40});
    tensor_t<TestType, 4> t4o({10, 20, 30, 40});

    t0() = (TestType)2.0f;
    for (index_t i = 0; i < t4i.Size(0); i++) {
      for (index_t j = 0; j < t4i.Size(1); j++) {
        for (index_t k = 0; k < t4i.Size(2); k++) {
          for (index_t l = 0; l < t4i.Size(3); l++) {
            t4i(i, j, k, l) =
                static_cast<detail::value_promote_t<TestType>>(i + j + k + l);
          }
        }
      }
    }

    (t4o = t4i * t0).run(exec);
    cudaStreamSynchronize(0);
  
    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TestType>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (TestType)t4i(i, j, k, l) * (TestType)t0());
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t0());
            }
          }
        }
      }
    }
    (t4o = t0 * t4i).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TestType>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (TestType)t0() * (TestType)t4i(i, j, k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t0() * t4i(i, j, k, l));
            }
          }
        }
      }
    }
  // }
  // {
  //   tensor_t<TestType, 1> t1({4});
  //   tensor_t<TestType, 4> t4i({1, 2, 3, 4});
  //   tensor_t<TestType, 4> t4o({1, 2, 3, 4});

  //   for (index_t i = 0; i < t1.Size(0); i++) {
  //     t1(i) = static_cast<detail::value_promote_t<TestType>>(i);
  //   }

  //   for (index_t i = 0; i < t4i.Size(0); i++) {
  //     for (index_t j = 0; j < t4i.Size(1); j++) {
  //       for (index_t k = 0; k < t4i.Size(2); k++) {
  //         for (index_t l = 0; l < t4i.Size(3); l++) {
  //           t4i(i, j, k, l) =
  //               static_cast<detail::value_promote_t<TestType>>(i + j + k + l);
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t4i * t1).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TestType>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t4i(i, j, k, l) * (double)t1(l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t1(l));
  //           }
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t1 * t4i).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TestType>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t1(l) * (double)t4i(i, j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t1(l) * t4i(i, j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // {
  //   tensor_t<TestType, 2> t2({3, 4});
  //   tensor_t<TestType, 4> t4i({1, 2, 3, 4});
  //   tensor_t<TestType, 4> t4o({1, 2, 3, 4});

  //   for (index_t i = 0; i < t2.Size(0); i++) {
  //     for (index_t j = 0; j < t2.Size(1); j++) {
  //       t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i + j);
  //     }
  //   }

  //   for (index_t i = 0; i < t4i.Size(0); i++) {
  //     for (index_t j = 0; j < t4i.Size(1); j++) {
  //       for (index_t k = 0; k < t4i.Size(2); k++) {
  //         for (index_t l = 0; l < t4i.Size(3); l++) {
  //           t4i(i, j, k, l) =
  //               static_cast<detail::value_promote_t<TestType>>(i + j + k + l);
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t4i * t2).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TestType>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t4i(i, j, k, l) * (double)t2(k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t2(k, l));
  //           }
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t2 * t4i).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TestType>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t2(k, l) * (double)t4i(i, j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t2(k, l) * t4i(i, j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // {
  //   tensor_t<TestType, 3> t3({2, 3, 4});
  //   tensor_t<TestType, 4> t4i({1, 2, 3, 4});
  //   tensor_t<TestType, 4> t4o({1, 2, 3, 4});

  //   for (index_t i = 0; i < t3.Size(0); i++) {
  //     for (index_t j = 0; j < t3.Size(1); j++) {
  //       for (index_t k = 0; k < t3.Size(2); k++) {
  //         t3(i, j, k) = static_cast<detail::value_promote_t<TestType>>(i + j + k);
  //       }
  //     }
  //   }

  //   for (index_t i = 0; i < t4i.Size(0); i++) {
  //     for (index_t j = 0; j < t4i.Size(1); j++) {
  //       for (index_t k = 0; k < t4i.Size(2); k++) {
  //         for (index_t l = 0; l < t4i.Size(3); l++) {
  //           t4i(i, j, k, l) =
  //               static_cast<detail::value_promote_t<TestType>>(i + j + k + l);
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t4i * t3).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TestType>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t4i(i, j, k, l) * (double)t3(j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t3(j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t3 * t4i).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TestType>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t3(j, k, l) * (double)t4i(i, j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t3(j, k, l) * t4i(i, j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // {
  //   tensor_t<TestType, 0> t0;
  //   tensor_t<TestType, 1> t1({4});
  //   tensor_t<TestType, 2> t2({3, 4});
  //   tensor_t<TestType, 3> t3({2, 3, 4});
  //   tensor_t<TestType, 4> t4i({1, 2, 3, 4});
  //   tensor_t<TestType, 4> t4o({1, 2, 3, 4});

  //   t0() = (TestType)200.0f;

  //   for (index_t i = 0; i < t2.Size(0); i++) {
  //     t1(i) = static_cast<detail::value_promote_t<TestType>>(i);
  //   }

  //   for (index_t i = 0; i < t2.Size(0); i++) {
  //     for (index_t j = 0; j < t2.Size(1); j++) {
  //       t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i + j);
  //     }
  //   }

  //   for (index_t i = 0; i < t3.Size(0); i++) {
  //     for (index_t j = 0; j < t3.Size(1); j++) {
  //       for (index_t k = 0; k < t3.Size(2); k++) {
  //         t3(i, j, k) = static_cast<detail::value_promote_t<TestType>>(i + j + k);
  //       }
  //     }
  //   }

  //   for (index_t i = 0; i < t4i.Size(0); i++) {
  //     for (index_t j = 0; j < t4i.Size(1); j++) {
  //       for (index_t k = 0; k < t4i.Size(2); k++) {
  //         for (index_t l = 0; l < t4i.Size(3); l++) {
  //           t4i(i, j, k, l) =
  //               static_cast<detail::value_promote_t<TestType>>(i + j + k + l);
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t4i + t3 + t2 + t1 + t0).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TestType>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t4i(i, j, k, l) + (double)t3(j, k, l) +
  //                                (double)t2(k, l) + (double)t1(l) +
  //                                (double)(double)t0());
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) + t3(j, k, l) +
  //                                                 t2(k, l) + t1(l) + t0());
  //           }
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t0 + t1 + t2 + t3 + t4i).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TestType>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t0() + (double)t1(l) + (double)t2(k, l) +
  //                                (double)t3(j, k, l) + (double)t4i(i, j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t0() + t1(l) + t2(k, l) +
  //                                                 t3(j, k, l) +
  //                                                 t4i(i, j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Concatenate)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 
  index_t i, j;

  auto t11 = make_tensor<TestType>({10});
  auto t12 = make_tensor<TestType>({5});
  auto t1o = make_tensor<TestType>({15});
  auto t1o1 = make_tensor<TestType>({30});

  t11.SetVals({0,1,2,3,4,5,6,7,8,9});
  t12.SetVals({0,1,2,3,4});

  (t1o = concat(0, t11, t12)).run(exec);
  cudaStreamSynchronize(0);

  for (i = 0; i < t11.Size(0) + t12.Size(0); i++) {
    if (i < t11.Size(0)) {
      ASSERT_EQ(t11(i), t1o(i));
    }
    else {
      ASSERT_EQ(t12(i - t11.Size(0)), t1o(i));
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
  cudaStreamSynchronize(0);

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
  cudaStreamSynchronize(0);
  
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

  // Concatenating 3 tensors
  (t1o1 = concat(0, t11, t11, t11)).run(exec);
  cudaStreamSynchronize(0);

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

    cudaStreamSynchronize(0);
    for (int cnt = 0; cnt < result.Size(0); cnt++) {
      ASSERT_EQ(result(cnt), cnt + 1);
    }    
  }  
  
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, Stack)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  auto t1a = make_tensor<TestType>({5});
  auto t1b = make_tensor<TestType>({5});
  auto t1c = make_tensor<TestType>({5});
 
  auto cop = concat(0, t1a, t1b, t1c);
  
  (cop = (TestType)2).run(exec);
  cudaDeviceSynchronize();

  {
    auto op = stack(0, t1a, t1b, t1c);
   
    for(int i = 0; i < t1a.Size(0); i++) {
      ASSERT_EQ(op(0,i), t1a(i));
      ASSERT_EQ(op(1,i), t1b(i));
      ASSERT_EQ(op(2,i), t1c(i));
    }
  }  
 
  {
    auto op = stack(1, t1a, t1b, t1c);
    
    for(int i = 0; i < t1a.Size(0); i++) {
      ASSERT_EQ(op(i,0), t1a(i));
      ASSERT_EQ(op(i,1), t1b(i));
      ASSERT_EQ(op(i,2), t1c(i));
    }
  }  
  
  MATX_EXIT_HANDLER();
}


TYPED_TEST(OperatorTestsComplexTypesAllExecs, HermitianTranspose)
{
  MATX_ENTER_HANDLER();

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

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

  (t2s = hermitianT(t2)).run(exec);
  cudaStreamSynchronize(0);

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

TYPED_TEST(OperatorTestsComplexTypesAllExecs, PlanarTransform)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{};   
  index_t m = 10;
  index_t k = 20;
  tensor_t<TestType, 2> t2({m, k});
  tensor_t<typename TestType::value_type, 2> t2p({m * 2, k});
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      TestType tmp = {(float)i, (float)-j};
      t2(i, j) = tmp;
    }
  }

  (t2p = planar(t2)).run(exec);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j).real(), t2p(i, j)));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2(i, j).imag(), t2p(i + t2.Size(0), j))) << i << " " << j << "\n";
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplexTypesAllExecs, InterleavedTransform)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  index_t m = 10;
  index_t k = 20;
  tensor_t<TestType, 2> t2({m, k});
  tensor_t<typename TestType::value_type, 2> t2p({m * 2, k});
  for (index_t i = 0; i < 2 * m; i++) {
    for (index_t j = 0; j < k; j++) {
      if (i >= m) {
        t2p(i, j) = 2.0f;
      }
      else {
        t2p(i, j) = -1.0f;
      }
    }
  }

  (t2 = interleaved(t2p)).run(exec);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j).real(), t2p(i, j)));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2(i, j).imag(), t2p(i + t2.Size(0), j)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAllExecs, RepMat)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  index_t count0 = 4;
  index_t count1 = 4;
  index_t same_reps = 10;
  tensor_t<TestType, 2> t2({count0, count1});
  tensor_t<TestType, 2> t2s({count0 * same_reps, count1 * same_reps});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i);
    }
  }

  auto repop = repmat(t2, same_reps);
  ASSERT_TRUE(repop.Size(0) == same_reps * t2.Size(0));
  ASSERT_TRUE(repop.Size(1) == same_reps * t2.Size(1));

  (t2s = repop).run(exec);
  cudaStreamSynchronize(0);

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
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count0 * same_reps; i++) {
    for (index_t j = 0; j < count1 * same_reps * 2; j++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2r(i, j), t2(i % count0, j % count1)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Sphere2Cart)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  int n = 5;

  auto xi = range<0>({n},(TestType)1,(TestType)1);
  auto yi = range<0>({n},(TestType)1,(TestType)1);
  auto zi = range<0>({n},(TestType)1,(TestType)1);

  auto [theta, phi, r] = cart2sph(xi, yi, zi);
  auto [x, y, z] = sph2cart(theta, phi, r);

  for(int i=0; i<n; i++) {
    ASSERT_NEAR(xi(i), x(i), .01);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, ShiftOp)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  index_t count0 = 100;
  index_t count1 = 201;
  tensor_t<TestType, 2> t2({count0, count1});
  tensor_t<TestType, 2> t2s({count0, count1});
  tensor_t<TestType, 2> t2s2({count0, count1});
  tensor_t<int, 0> t0;
  t0() = -5;

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i * count1 + j);
    }
  }

  {
    (t2s = shift<0>(t2, -5)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2s(i, j), t2((i + 5) % count0, j)));
      }
    }
  }
  
  {
    (t2s = shift<0>(t2, t0)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2s(i, j), t2((i + 5) % count0, j)));
      }
    }
  }

  {
    (t2s = shift<1>(t2, -5)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, (j + 5) % count1)));
      }
    }
  }

  {
    (t2s = shift<1,0>(t2, -5, -6)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j), t2((i + 6) % count0, (j + 5) % count1)));
      }
    }
  }

  {
    (t2s = fftshift2D(t2)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j), t2((i + (count0 + 1) / 2) % count0,
                          (j + (count1 + 1) / 2) % count1)));
      }
    }
  }

  {
    (t2s = ifftshift2D(t2)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j),
            t2((i + (count0) / 2) % count0, (j + (count1) / 2) % count1)));
      }
    }
  }

  // Right shifts
  {
    (t2s = shift<0>(t2, 5)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        index_t idim = i < 5 ? (t2.Size(0) - 5 + i) : (i - 5);
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(idim, j)));
      }
    }
  }

  {
    (t2s = shift<1>(t2, 5)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        index_t jdim = j < 5 ? (t2.Size(1) - 5 + j) : (j - 5);
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, jdim)));
      }
    }
  }

  // Large shifts
  {
    (t2s = shift<0>(t2, -t2.Size(0) * 4)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, j)));
      }
    }
  }

  {
    // Shift 4 times the size back, minus one. This should be equivalent to
    // simply shifting by -1
    (t2s = shift<0>(t2, -t2.Size(0) * 4 - 1)).run(exec);
    (t2s2 = shift<0>(t2, -1)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2s2(i, j)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericAllExecs, Reverse)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  index_t count0 = 100;
  index_t count1 = 200;
  tensor_t<TestType, 2> t2({count0, count1});
  tensor_t<TestType, 2> t2r({count0, count1});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i * count1 + j);
    }
  }

  {
    (t2r = reverse<0>(t2)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(count0 - i - 1, j)));
      }
    }
  }

  {
    (t2r = reverse<1>(t2)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(i, count1 - j - 1)));
      }
    }
  }

  {
    (t2r = reverse<0,1>(t2)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2r(i, j), t2(count0 - i - 1, count1 - j - 1)));
      }
    }
  }

  // Flip versions
  {
    (t2r = flipud(t2)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(count0 - i - 1, j)));
      }
    }
  }

  {
    (t2r = fliplr(t2)).run(exec);
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(i, count1 - j - 1)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, Cast)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 4;
  auto t = make_tensor<int8_t>({count0});
  auto t2 = make_tensor<int8_t>({count0});
  auto to = make_tensor<float>({count0});

  t.SetVals({126, 126, 126, 126});
  t2.SetVals({126, 126, 126, 126});
  
  (to = as_type<int8_t>(t + t2)).run();
  cudaStreamSynchronize(0);

  for (int i = 0; i < t.Size(0); i++) {
    ASSERT_EQ(to(i), -4); // -4 from 126 + 126 wrap-around
  }

  (to = as_int8(t + t2)).run();
  cudaStreamSynchronize(0);
  
  for (int i = 0; i < t.Size(0); i++) {
    ASSERT_EQ(to(i), -4); // -4 from 126 + 126 wrap-around
  }  

  MATX_EXIT_HANDLER();
}

template<class TypeParam>
TypeParam legendre_check(int n, int m, TypeParam x) {
	if (m > n ) return 0;

	TypeParam a = detail::_internal_sqrt(TypeParam(1)-x*x);
	// first we will move move along diagonal

	// initialize registers
	TypeParam d1 = 1, d0;

	for(int i=0; i < m; i++) {
		// advance diagonal (shift)
		d0 = d1;
		// compute next term using recurrence relationship
		d1 = -TypeParam(2*i+1)*a*d0;
	}

	// next we will move to the right till we get to the correct entry

	// initialize registers
	TypeParam p0, p1 = 0, p2 = d1;

	for(int l=m; l<n; l++) {
		// advance one step (shift)
		p0 = p1;
		p1 = p2;

		// Compute next term using recurrence relationship
		p2 = (TypeParam(2*l+1) * x * p1 - TypeParam(l+m)*p0)/(TypeParam(l-m+1));
	}

	return p2;

}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Legendre)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  ExecType exec{}; 

  index_t size = 11;
  int order = 5;
  
  { // vector for n and m
    auto n = range<0, 1, int>({order}, 0, 1);
    auto m = range<0, 1, int>({order}, 0, 1);
    auto x = as_type<TestType>(linspace<0>({size}, TestType(0), TestType(1)));

    auto out = make_tensor<TestType>({order, order, size});

    (out = legendre(n, m, x)).run(exec);

    cudaStreamSynchronize(0);

    for(int j = 0; j < order; j++) {
      for(int p = 0; p < order; p++) {
        for(int i = 0 ; i < size; i++) {
          if constexpr (is_matx_half_v<TestType>) {
            ASSERT_NEAR(out(p,j,i), legendre_check(p, j, x(i)),50.0);
          }
          else {
            ASSERT_NEAR(out(p,j,i), legendre_check(p, j, x(i)),.0001);
          }
        }
      }
    }
  }
 
  { // constant for n
    auto m = range<0, 1, int>({order}, 0, 1);
    auto x = as_type<TestType>(linspace<0>({size}, TestType(0), TestType(1)));

    auto out = make_tensor<TestType>({order, size});

    (out = lcollapse<2>(legendre(order, m, x))).run(exec);

    cudaStreamSynchronize(0);

    for(int i = 0 ; i < size; i++) {
      for(int p = 0; p < order; p++) {
        if constexpr (is_matx_half_v<TestType>) {
          ASSERT_NEAR(out(p,i), legendre_check(order, p, x(i)),50.0);
        }
        else {
          ASSERT_NEAR(out(p,i), legendre_check(order, p, x(i)),.0001);
        }        
      }
    }
  }

  { // taking a constant for m and n;
    auto x = as_type<TestType>(linspace<0>({size}, TestType(0), TestType(1)));

    auto out = make_tensor<TestType>({size});

    (out = lcollapse<3>(legendre(order, order,  x))).run(exec);

    cudaStreamSynchronize(0);

    for(int i = 0 ; i < size; i++) {
      if constexpr (is_matx_half_v<TestType>) {
        ASSERT_NEAR(out(i), legendre_check(order, order, x(i)),50.0);
      }
      else {
        ASSERT_NEAR(out(i), legendre_check(order, order, x(i)),.0001);
      }        
    }
  }
  
  { // taking a rank0 tensor for m and constant for n
    auto x = as_type<TestType>(linspace<0>({size}, TestType(0), TestType(1)));
    auto m = make_tensor<int>();
    auto out = make_tensor<TestType>({size});
    m() = order;

    (out = lcollapse<3>(legendre(order, m,  x))).run(exec);

    cudaStreamSynchronize(0);

    for(int i = 0 ; i < size; i++) {
      if constexpr (is_matx_half_v<TestType>) {
        ASSERT_NEAR(out(i), legendre_check(order, order, x(i)),50.0);
      }
      else {
        ASSERT_NEAR(out(i), legendre_check(order, order, x(i)),.0001);
      }        
    }
  }
  MATX_EXIT_HANDLER();
}

TEST(OperatorTestsAdvanced, AdvancedRemapOp)
{
  typedef cuda::std::complex<float> complex;
  MATX_ENTER_HANDLER();

  int I = 4;
  int J = 4;
  int K = 14;
  int L = 133;

  int F = 4096;
  int P = 288;

  int M = 2;

  auto idx = matx::make_tensor<int, 1>({M});

  idx(0) = 1;
  idx(1) = 3;

  auto A = matx::make_tensor<complex, 4>({I, J, K, L});
  //collapsed tensor
  auto B = matx::make_tensor<complex, 2>({I * M * K, L});

  auto index = [&] (int i, int j, int k, int l) {
    return i * J * K * L +
      j * K * L +
      k * L +
      l;
  };
  for (int i = 0; i < I ; i++) {
    for (int j = 0; j < J ; j++) {
      for (int k = 0; k < K ; k++) {
        for (int l = 0; l < L ; l++) {
          float val = (float)index(i,j,k,l);
          A(i,j,k,l) = complex(val, val/100);
        }
      }
    }
  }

  (B = 0).run();

  auto rop = remap<1>(A, idx);
  auto lop = lcollapse<3>(rop);

  ASSERT_EQ(lop.Rank() , 2);
  ASSERT_EQ(lop.Size(1) , A.Size(3));
  ASSERT_EQ(lop.Size(0) , I * M * K);

  (B = lop).run();

  cudaDeviceSynchronize();  

  for (int i = 0; i < I; i++) {
    for (int m = 0; m < M; m++) {
      for (int k = 0; k < K; k++) {
        for (int l = 0; l < L; l++) {
          int j = idx(m);
          int fidx = i * M * K + m * K  + k;
          float val = (float)index(i,j,k,l);
          complex expected_val = complex(val,val/100);
          complex a_val = A(i,j,k,l);
          complex b_val = B(fidx, l);	  
          complex lop_val = lop(fidx, l);
          complex rop_val = rop(i, m, k, l);

          ASSERT_EQ(a_val, expected_val);
          ASSERT_EQ(rop_val, expected_val);
          ASSERT_EQ(lop_val, expected_val);
          ASSERT_EQ(b_val, expected_val);

          ASSERT_EQ(B(fidx, l) , lop(fidx, l));
        }
      }
    }
  }


  // convolution test
  auto O1 = matx::make_tensor<complex, 4>({I, J, K, F + P + L - 1});
  auto O2 = matx::make_tensor<complex, 4>({I, J, K, F + P + L - 1});
  auto O3 = matx::make_tensor<complex, 4>({I, J, K, F + P + L - 1});
  auto O4 = matx::make_tensor<complex, 4>({I, J, K, F + P + L - 1});

  auto C = matx::make_tensor<complex, 3>({I, K, F + P});
  //collapsed tensor
  auto D = matx::make_tensor<complex, 2>({I * M * K, F + P});
  
  auto indexc = [&] (int i, int j, int k) {
    return i * C.Size(1) * C.Size(2) +
      j * C.Size(2) +
      k;
  };
  
  for (int i = 0; i < I ; i++) {
    for (int j = 0; j < J ; j++) {
      for (int k = 0; k < K ; k++) {
        float val = (float) indexc(i,j,k);
        C(i,j,k) = complex(val, val/100);
      }
    }
  }
  
  A.PrefetchDevice(0);
  B.PrefetchDevice(0);
  C.PrefetchDevice(0);
  D.PrefetchDevice(0);
  O1.PrefetchDevice(0);
  O2.PrefetchDevice(0);
  O3.PrefetchDevice(0);
  O4.PrefetchDevice(0);

  cudaDeviceSynchronize();

  auto o1op = lcollapse<3>(remap<1>(O1, idx));
  auto o2op = lcollapse<3>(remap<1>(O2, idx));
  auto o3op = lcollapse<3>(remap<1>(O3, idx));
  auto o4op = lcollapse<3>(remap<1>(O4, idx));

  auto cop = C.Clone<4>({matxKeepDim, M, matxKeepDim, matxKeepDim});
  auto rcop = lcollapse<3>(remap<1>(cop, idx));

  (O1 = 1).run();
  (O2 = 2).run();
  (O3 = 3).run();
  (O4 = 4).run();
  
  (B = lop).run();
  (D = rcop).run();

  // two operators as input
  matx::conv1d(o1op, lop, rcop, matx::matxConvCorrMode_t::MATX_C_MODE_FULL, 0);

  // one tensor and one operators as input
  matx::conv1d(o2op, B, rcop, matx::matxConvCorrMode_t::MATX_C_MODE_FULL, 0);
  
  // one tensor and one operators as input
  matx::conv1d(o3op, lop, D, matx::matxConvCorrMode_t::MATX_C_MODE_FULL, 0);
  
  //two tensors as input
  matx::conv1d(o4op, B, D, matx::matxConvCorrMode_t::MATX_C_MODE_FULL, 0);

  cudaDeviceSynchronize();

  for (int i = 0; i < o1op.Size(0); i++) {
    for (int l = 0; l < o1op.Size(1); l++) {
      ASSERT_EQ(o1op(i,l), o2op(i,l));
      ASSERT_EQ(o2op(i,l), o3op(i,l));
      ASSERT_EQ(o3op(i,l), o4op(i,l));
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(OperatorTestsFloatAllExecs, Print)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({3});
  auto r = ones<TestType>(t.Shape());

  print(r);
  MATX_EXIT_HANDLER();
}  
