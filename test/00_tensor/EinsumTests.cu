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
#include "matx_pybind.h"
#include "matx_einsum.h"

using namespace matx;

template <typename T> class EinsumTest : public ::testing::Test {

protected:
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.01f;
};


template <typename TensorType>
class EinsumTestsComplex : public EinsumTest<TensorType> {
};
template <typename TensorType>
class EinsumTestsFloat : public EinsumTest<TensorType> {
};
template <typename TensorType>
class EinsumTestsFloatNonComplex
    : public EinsumTest<TensorType> {
};
template <typename TensorType>
class EinsumTestsNumeric : public EinsumTest<TensorType> {
};
template <typename TensorType>
class EinsumTestsNumericNonComplex
    : public EinsumTest<TensorType> {
};

template <typename TensorType>
class EinsumTestsFloatNonComplexNonHalfTypes
    : public EinsumTest<TensorType> {
};

template <typename TensorType>
class EinsumTestsIntegral : public EinsumTest<TensorType> {
};
template <typename TensorType>
class EinsumTestsBoolean : public EinsumTest<TensorType> {
};
template <typename TensorType>
class EinsumTestsAll : public EinsumTest<TensorType> {
};

TYPED_TEST_SUITE(EinsumTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(EinsumTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(EinsumTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(EinsumTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(EinsumTestsFloatNonComplexNonHalfTypes, MatXFloatNonComplexNonHalfTypes);
TYPED_TEST_SUITE(EinsumTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(EinsumTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(EinsumTestsNumericNonComplex, MatXNumericNonComplexTypes);
TYPED_TEST_SUITE(EinsumTestsBoolean, MatXBoolTypes);

#if ENABLE_CUTENSOR
TYPED_TEST(EinsumTestsFloatNonComplexNonHalfTypes, Contraction3D)
{
  MATX_ENTER_HANDLER();

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_operators", "contraction", "run", {});  

  auto a1 = make_tensor<TypeParam>({60});
  auto b1 = make_tensor<TypeParam>({24});
  auto c2 = make_tensor<TypeParam>({5,2});

  (a1 = linspace<0>(a1.Shape(), (TypeParam)0, static_cast<TypeParam>(a1.Size(0) - 1))).run();
  (b1 = linspace<0>(b1.Shape(), (TypeParam)0, static_cast<TypeParam>(b1.Size(0) - 1))).run();
  auto a = a1.View({3,4,5});
  auto b = b1.View({4,3,2});

  cutensor::einsum(c2, "ijk,jil->kl", 0, a, b);
  cudaStreamSynchronize(0);
  MATX_TEST_ASSERT_COMPARE(this->pb, c2, "c_float3d", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EinsumTestsFloatNonComplexNonHalfTypes, Dot)
{
  MATX_ENTER_HANDLER();

  auto a1 = make_tensor<TypeParam>({60});
  auto b1 = make_tensor<TypeParam>({60});
  auto c0 = make_tensor<TypeParam>();
  (a1 = ones(a1.Shape()) * 2).run();
  (b1 = ones(b1.Shape()) * 2).run(); 
  cutensor::einsum(c0, "i,i->", 0, a1, b1);   
  cudaStreamSynchronize(0);
  MATX_ASSERT_EQ(c0(), 4 * a1.Size(0));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EinsumTestsFloatNonComplexNonHalfTypes, GEMM)
{
  MATX_ENTER_HANDLER();

  auto a2 = make_tensor<TypeParam>({10,20});
  auto b2 = make_tensor<TypeParam>({20,10});
  auto c2 = make_tensor<TypeParam>({10,10});    
  auto c22 = make_tensor<TypeParam>({10,10});   
  (a2 = ones(a2.Shape())).run();
  (b2 = ones(b2.Shape())).run(); 

  cutensor::einsum(c2, "mk,kn->mn", 0, a2, b2);
  matmul(c22, a2, b2);
  cudaStreamSynchronize(0);

  for (auto i = 0; i < c2.Size(0); i++) {
    for (auto j = 0; j < c2.Size(1); j++) {
      MATX_ASSERT_EQ(c2(i,j), c22(i,j));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EinsumTestsFloatNonComplexNonHalfTypes, GEMMTranspose)
{
    auto a2 = make_tensor<TypeParam>({5,20});
    auto b2 = make_tensor<TypeParam>({20,10});
    auto c2 = make_tensor<TypeParam>({10,5});    
    auto c22 = make_tensor<TypeParam>({5,10});   
    (a2 = ones(a2.Shape())).run();
    (b2 = ones(b2.Shape())).run(); 

    cutensor::einsum(c2, "mk,kn->nm", 0, a2, b2);
    matmul(c22, a2, b2);
    cudaStreamSynchronize(0);

    auto c22t = c22.Permute({1,0}); // Permute to match cutensor

    for (auto i = 0; i < c2.Size(0); i++) {
      for (auto j = 0; j < c2.Size(1); j++) {
        MATX_ASSERT_EQ(c2(i,j), c22t(i,j));
      }
    }
}

TYPED_TEST(EinsumTestsFloatNonComplexNonHalfTypes, Permute)
{
  auto a = make_tensor<TypeParam>({5,20,4,3});
  auto b = make_tensor<TypeParam>({20,3,4,5});  
  auto b2 = make_tensor<TypeParam>({20,3,4,5});  
  (a = ones(a.Shape())).run();
  (b = ones(b.Shape())).run(); 

  cutensor::einsum(b, "ijkl->jlki", 0, a);
  (b2 = a.Permute({1,3,2,0})).run();
  cudaStreamSynchronize(0);

  for (auto i = 0; i < b.Size(0); i++) {
    for (auto j = 0; j < b.Size(1); j++) {
      for (auto k = 0; k < b.Size(2); k++) {
        for (auto l = 0; l < b.Size(3); l++) {
          MATX_ASSERT_EQ(b(i,j,k,l), b2(i,j,k,l));
        }
      }
    }
  }
}


#endif
