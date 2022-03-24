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

using namespace matx;

template <typename TensorType> struct TensorCreationTestsData {
  tensor_t<TensorType, 0> t0{};
  tensor_t<TensorType, 1> t1{{10}};
  tensor_t<TensorType, 2> t2{{20, 10}};
  tensor_t<TensorType, 3> t3{{30, 20, 10}};
  tensor_t<TensorType, 4> t4{{40, 30, 20, 10}};

  tensor_t<TensorType, 2> t2s = t2.Permute({1, 0});
  tensor_t<TensorType, 3> t3s = t3.Permute({2, 1, 0});
  tensor_t<TensorType, 4> t4s = t4.Permute({3, 2, 1, 0});
};

template <typename TensorType>
class TensorCreationTestsComplex : public ::testing::Test,
                                public TensorCreationTestsData<TensorType> {
};
template <typename TensorType>
class TensorCreationTestsFloat : public ::testing::Test,
                              public TensorCreationTestsData<TensorType> {
};
template <typename TensorType>
class TensorCreationTestsFloatNonComplex
    : public ::testing::Test,
      public TensorCreationTestsData<TensorType> {
};
template <typename TensorType>
class TensorCreationTestsNumeric : public ::testing::Test,
                                public TensorCreationTestsData<TensorType> {
};
template <typename TensorType>
class TensorCreationTestsNumericNonComplex
    : public ::testing::Test,
      public TensorCreationTestsData<TensorType> {
};
template <typename TensorType>
class TensorCreationTestsIntegral : public ::testing::Test,
                                 public TensorCreationTestsData<TensorType> {
};
template <typename TensorType>
class TensorCreationTestsBoolean : public ::testing::Test,
                                public TensorCreationTestsData<TensorType> {
};
template <typename TensorType>
class TensorCreationTestsAll : public ::testing::Test,
                            public TensorCreationTestsData<TensorType> {
};

TYPED_TEST_SUITE(TensorCreationTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(TensorCreationTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(TensorCreationTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(TensorCreationTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(TensorCreationTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(TensorCreationTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(TensorCreationTestsNumericNonComplex, MatXNumericNonComplexTypes);
TYPED_TEST_SUITE(TensorCreationTestsBoolean, MatXBoolTypes);

TYPED_TEST(TensorCreationTestsAll, MakeShape)
{
  auto mt2 = make_tensor<TypeParam>({2, 2});
  ASSERT_EQ(mt2.Size(0), 2);
  ASSERT_EQ(mt2.Size(1), 2);

  auto mt0 = make_tensor<TypeParam>();
  auto mt1 = make_tensor<TypeParam>({10});
  auto mt3 = make_tensor<TypeParam>({10, 5, 4});
  auto mt4 = make_tensor<TypeParam>({10, 5, 4, 3});

  ASSERT_EQ(mt1.Size(0), 10);
  ASSERT_EQ(mt3.Size(0), 10);
  ASSERT_EQ(mt3.Size(1), 5);
  ASSERT_EQ(mt3.Size(2), 4);
  ASSERT_EQ(mt4.Size(0), 10);
  ASSERT_EQ(mt4.Size(1), 5);
  ASSERT_EQ(mt4.Size(2), 4);
  ASSERT_EQ(mt4.Size(3), 3);  
}

// TYPED_TEST(TensorCreationTestsAll, MakeStaticShape)
// {
//   auto mt1 = make_static_tensor<TypeParam, 10>();
//   ASSERT_EQ(mt1.Size(0), 10);

//   auto mt2 = make_static_tensor<TypeParam, 10, 40>();
//   ASSERT_EQ(mt2.Size(0), 10);
//   ASSERT_EQ(mt2.Size(1), 40);

//   auto mt3 = make_static_tensor<TypeParam, 10, 40, 30>();
//   ASSERT_EQ(mt3.Size(0), 10);
//   ASSERT_EQ(mt3.Size(1), 40);  
//   ASSERT_EQ(mt3.Size(2), 30);  

//   auto mt4 = make_static_tensor<TypeParam, 10, 40, 30, 6>();
//   ASSERT_EQ(mt4.Size(0), 10);
//   ASSERT_EQ(mt4.Size(1), 40);  
//   ASSERT_EQ(mt4.Size(2), 30);    
//   ASSERT_EQ(mt4.Size(3), 6);    
// }
