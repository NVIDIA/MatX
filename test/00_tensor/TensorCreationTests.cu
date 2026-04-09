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

template <typename T> struct TensorCreationTestsData {
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;     
  tensor_t<GTestType, 0> t0{{}};
  tensor_t<GTestType, 1> t1{{10}};
  tensor_t<GTestType, 2> t2{{20, 10}};
  tensor_t<GTestType, 3> t3{{30, 20, 10}};
  tensor_t<GTestType, 4> t4{{40, 30, 20, 10}};

  tensor_t<GTestType, 2> t2s = t2.Permute({1, 0});
  tensor_t<GTestType, 3> t3s = t3.Permute({2, 1, 0});
  tensor_t<GTestType, 4> t4s = t4.Permute({3, 2, 1, 0});
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



TYPED_TEST_SUITE(TensorCreationTestsAll, MatXAllTypesAllExecs);
TYPED_TEST_SUITE(TensorCreationTestsComplex, MatXComplexTypesAllExecs);

TYPED_TEST(TensorCreationTestsAll, MakeShape)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};  
  auto mt2 = make_tensor<TestType>({2, 2});
  ASSERT_EQ(mt2.Size(0), 2);
  ASSERT_EQ(mt2.Size(1), 2);

  auto mt0 = make_tensor<float>({});
  auto mt1 = make_tensor<cuda::std::complex<float>>({10});
  auto mt3 = make_tensor<TestType>({10, 5, 4});
  auto mt4 = make_tensor<TestType>({10, 5, 4, 3});

  ASSERT_EQ(mt1.Size(0), 10);
  ASSERT_EQ(mt3.Size(0), 10);
  ASSERT_EQ(mt3.Size(1), 5);
  ASSERT_EQ(mt3.Size(2), 4);
  ASSERT_EQ(mt4.Size(0), 10);
  ASSERT_EQ(mt4.Size(1), 5);
  ASSERT_EQ(mt4.Size(2), 4);
  ASSERT_EQ(mt4.Size(3), 3);  
}

TYPED_TEST(TensorCreationTestsAll, MakeStaticShape)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto mt1 = make_static_tensor<TestType, 10>();
  ASSERT_EQ(mt1.Size(0), 10);

  auto mt2 = make_static_tensor<float, 10, 40>();
  ASSERT_EQ(mt2.Size(0), 10);
  ASSERT_EQ(mt2.Size(1), 40);

  auto mt3 = make_static_tensor<TestType, 10, 40, 30>();
  ASSERT_EQ(mt3.Size(0), 10);
  ASSERT_EQ(mt3.Size(1), 40);
  ASSERT_EQ(mt3.Size(2), 30);

  auto mt4 = make_static_tensor<TestType, 10, 40, 30, 6>();
  ASSERT_EQ(mt4.Size(0), 10);
  ASSERT_EQ(mt4.Size(1), 40);
  ASSERT_EQ(mt4.Size(2), 30);
  ASSERT_EQ(mt4.Size(3), 6);
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorRankAndStrides)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto mt1 = make_static_tensor<TestType, 8>();
  ASSERT_EQ(mt1.Rank(), 1);
  ASSERT_EQ(mt1.Stride(0), 1);
  ASSERT_TRUE(mt1.IsContiguous());

  auto mt2 = make_static_tensor<TestType, 4, 5>();
  ASSERT_EQ(mt2.Rank(), 2);
  ASSERT_EQ(mt2.Stride(0), 5);
  ASSERT_EQ(mt2.Stride(1), 1);
  ASSERT_TRUE(mt2.IsContiguous());

  auto mt3 = make_static_tensor<TestType, 3, 4, 5>();
  ASSERT_EQ(mt3.Rank(), 3);
  ASSERT_EQ(mt3.Stride(0), 20);
  ASSERT_EQ(mt3.Stride(1), 5);
  ASSERT_EQ(mt3.Stride(2), 1);
  ASSERT_TRUE(mt3.IsContiguous());

  auto mt4 = make_static_tensor<TestType, 2, 3, 4, 5>();
  ASSERT_EQ(mt4.Rank(), 4);
  ASSERT_EQ(mt4.Stride(0), 60);
  ASSERT_EQ(mt4.Stride(1), 20);
  ASSERT_EQ(mt4.Stride(2), 5);
  ASSERT_EQ(mt4.Stride(3), 1);
  ASSERT_TRUE(mt4.IsContiguous());
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorTotalSize)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto mt1 = make_static_tensor<TestType, 10>();
  ASSERT_EQ(mt1.TotalSize(), 10);

  auto mt2 = make_static_tensor<TestType, 4, 5>();
  ASSERT_EQ(mt2.TotalSize(), 20);

  auto mt3 = make_static_tensor<TestType, 3, 4, 5>();
  ASSERT_EQ(mt3.TotalSize(), 60);

  auto mt4 = make_static_tensor<TestType, 2, 3, 4, 5>();
  ASSERT_EQ(mt4.TotalSize(), 120);
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorDataPointer)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto mt1 = make_static_tensor<TestType, 10>();
  ASSERT_NE(mt1.Data(), nullptr);

  auto mt2 = make_static_tensor<TestType, 4, 5>();
  ASSERT_NE(mt2.Data(), nullptr);
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorAssignOnes)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  auto mt1 = make_static_tensor<TestType, 4>();
  (mt1 = ones<TestType>()).run(exec);
  exec.sync();
  for (index_t i = 0; i < 4; i++) {
    ASSERT_EQ(mt1(i), TestType(1));
  }

  auto mt2 = make_static_tensor<TestType, 3, 4>();
  (mt2 = ones<TestType>()).run(exec);
  exec.sync();
  for (index_t i = 0; i < 3; i++) {
    for (index_t j = 0; j < 4; j++) {
      ASSERT_EQ(mt2(i, j), TestType(1));
    }
  }
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorAssignZeros)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  auto mt1 = make_static_tensor<TestType, 5>();
  (mt1 = zeros<TestType>()).run(exec);
  exec.sync();
  for (index_t i = 0; i < 5; i++) {
    ASSERT_EQ(mt1(i), TestType(0));
  }

  auto mt2 = make_static_tensor<TestType, 2, 3>();
  (mt2 = zeros<TestType>()).run(exec);
  exec.sync();
  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 3; j++) {
      ASSERT_EQ(mt2(i, j), TestType(0));
    }
  }
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorCopy)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  auto src = make_static_tensor<TestType, 4, 3>();
  (src = ones<TestType>()).run(exec);
  exec.sync();

  // Copy constructor
  auto dst = src;
  ASSERT_EQ(dst.Size(0), 4);
  ASSERT_EQ(dst.Size(1), 3);
  ASSERT_EQ(dst.Data(), src.Data()); // shallow copy

  // Assign into another static tensor
  auto dst2 = make_static_tensor<TestType, 4, 3>();
  (dst2 = src).run(exec);
  exec.sync();
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 3; j++) {
      ASSERT_EQ(dst2(i, j), TestType(1));
    }
  }
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorArithmetic)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  auto a = make_static_tensor<TestType, 4>();
  auto b = make_static_tensor<TestType, 4>();
  auto c = make_static_tensor<TestType, 4>();

  (a = ones<TestType>()).run(exec);
  (b = ones<TestType>()).run(exec);
  (c = a + b).run(exec);
  exec.sync();

  for (index_t i = 0; i < 4; i++) {
    ASSERT_EQ(c(i), TestType(2));
  }
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorArithmetic2D)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  auto a = make_static_tensor<TestType, 3, 4>();
  auto b = make_static_tensor<TestType, 3, 4>();
  auto c = make_static_tensor<TestType, 3, 4>();

  (a = ones<TestType>()).run(exec);
  (b = ones<TestType>()).run(exec);
  (c = a + b).run(exec);
  exec.sync();

  for (index_t i = 0; i < 3; i++) {
    for (index_t j = 0; j < 4; j++) {
      ASSERT_EQ(c(i, j), TestType(2));
    }
  }
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorArithmetic3D)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  auto a = make_static_tensor<TestType, 2, 3, 4>();
  auto b = make_static_tensor<TestType, 2, 3, 4>();
  auto c = make_static_tensor<TestType, 2, 3, 4>();

  (a = ones<TestType>()).run(exec);
  (b = ones<TestType>()).run(exec);
  (c = a + b).run(exec);
  exec.sync();

  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 3; j++) {
      for (index_t k = 0; k < 4; k++) {
        ASSERT_EQ(c(i, j, k), TestType(2));
      }
    }
  }
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorArithmetic4D)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  auto a = make_static_tensor<TestType, 2, 3, 4, 5>();
  auto b = make_static_tensor<TestType, 2, 3, 4, 5>();
  auto c = make_static_tensor<TestType, 2, 3, 4, 5>();

  (a = ones<TestType>()).run(exec);
  (b = ones<TestType>()).run(exec);
  (c = a + b).run(exec);
  exec.sync();

  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 3; j++) {
      for (index_t k = 0; k < 4; k++) {
        for (index_t l = 0; l < 5; l++) {
          ASSERT_EQ(c(i, j, k, l), TestType(2));
        }
      }
    }
  }
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorShape)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto mt = make_static_tensor<TestType, 3, 4, 5>();
  auto shape = mt.Shape();
  ASSERT_EQ(shape[0], 3);
  ASSERT_EQ(shape[1], 4);
  ASSERT_EQ(shape[2], 5);
}

TYPED_TEST(TensorCreationTestsAll, StaticTensorDescriptor)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto mt = make_static_tensor<TestType, 3, 4, 5>();
  auto desc = mt.Descriptor();
  ASSERT_EQ(desc.Rank(), 3);
  ASSERT_EQ(desc.Size(0), 3);
  ASSERT_EQ(desc.Size(1), 4);
  ASSERT_EQ(desc.Size(2), 5);
  ASSERT_TRUE(desc.IsContiguous());
}

// Tests for static tensors with complex types (issue #759)
TYPED_TEST(TensorCreationTestsComplex, StaticTensorRealView)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using InnerType = typename TestType::value_type;
  ExecType exec{};

  constexpr index_t N = 8;
  auto t = make_static_tensor<TestType, N>();
  (t = ones<TestType>()).run(exec);
  exec.sync();

  auto real = t.RealView();
  ASSERT_EQ(real.Size(0), N);
  ASSERT_EQ(real.Rank(), 1);
  for (index_t i = 0; i < N; i++) {
    ASSERT_EQ(real(i), InnerType(1));
  }
}

TYPED_TEST(TensorCreationTestsComplex, StaticTensorImagView)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using InnerType = typename TestType::value_type;
  ExecType exec{};

  constexpr index_t N = 8;
  auto t = make_static_tensor<TestType, N>();
  (t = ones<TestType>()).run(exec);
  exec.sync();

  auto imag = t.ImagView();
  ASSERT_EQ(imag.Size(0), N);
  ASSERT_EQ(imag.Rank(), 1);
  for (index_t i = 0; i < N; i++) {
    ASSERT_EQ(imag(i), InnerType(0));
  }
}

TYPED_TEST(TensorCreationTestsComplex, StaticTensorRealView2D)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using InnerType = typename TestType::value_type;
  ExecType exec{};

  auto t = make_static_tensor<TestType, 4, 3>();
  (t = ones<TestType>()).run(exec);
  exec.sync();

  auto real = t.RealView();
  ASSERT_EQ(real.Size(0), 4);
  ASSERT_EQ(real.Size(1), 3);
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 3; j++) {
      ASSERT_EQ(real(i, j), InnerType(1));
    }
  }

  auto imag = t.ImagView();
  ASSERT_EQ(imag.Size(0), 4);
  ASSERT_EQ(imag.Size(1), 3);
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 3; j++) {
      ASSERT_EQ(imag(i, j), InnerType(0));
    }
  }
}
