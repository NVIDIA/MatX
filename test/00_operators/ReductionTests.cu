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
#include <random>

using namespace matx;

template <typename TensorType>
class ReductionTestsComplex : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsFloat : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsNumeric : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsNumericNonComplex : public ::testing::Test {
};

template <typename TensorType>
class ReductionTestsFloatNonComplex : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsFloatNonComplexNonHalf : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsIntegral : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsBoolean : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsFloatHalf : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsNumericNoHalf : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsAll : public ::testing::Test {
};
template <typename TensorType>
class ReductionTestsComplexNonHalfTypes : public ::testing::Test {
};

template <typename TensorType>
class ReductionTestsFloatNonComplexNonHalfAllExecs : public ::testing::Test {
};

template <typename TensorType>
class ReductionTestsNumericNoHalfAllExecs : public ::testing::Test {
};

template <typename TensorType>
class ReductionTestsComplexNonHalfTypesAllExecs : public ::testing::Test {
};

template <typename TensorType>
class ReductionTestsNumericNonComplexAllExecs : public ::testing::Test {
};

TYPED_TEST_SUITE(ReductionTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(ReductionTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(ReductionTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(ReductionTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(ReductionTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(ReductionTestsNumericNonComplex,
                 MatXNumericNonComplexTypes);               
TYPED_TEST_SUITE(ReductionTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(ReductionTestsFloatNonComplexNonHalf,
                 MatXFloatNonComplexNonHalfTypes);
TYPED_TEST_SUITE(ReductionTestsBoolean, MatXBoolTypes);
TYPED_TEST_SUITE(ReductionTestsFloatHalf, MatXFloatHalfTypes);
TYPED_TEST_SUITE(ReductionTestsNumericNoHalf, MatXNumericNoHalfTypes);
TYPED_TEST_SUITE(ReductionTestsComplexNonHalfTypes, MatXComplexNonHalfTypes);


TYPED_TEST_SUITE(ReductionTestsNumericNonComplexAllExecs,
                 MatXNumericNonComplexTypesAllExecs);  
TYPED_TEST_SUITE(ReductionTestsFloatNonComplexNonHalfAllExecs,
                 MatXFloatNonComplexNonHalfTypesAllExecs);  
TYPED_TEST_SUITE(ReductionTestsNumericNoHalfAllExecs, MatXNumericNoHalfTypesAllExecs);          
TYPED_TEST_SUITE(ReductionTestsComplexNonHalfTypesAllExecs, MatXComplexNonHalfTypesAllExecs);


TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, VarianceStd)
{
  MATX_ENTER_HANDLER();

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};  

  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr index_t size = 100;
  pb->InitAndRunTVGenerator<TestType>("00_operators", "stats", "run", {size});

  tensor_t<TestType, 0> t0;
  tensor_t<TestType, 1> t1({size});
  pb->NumpyToTensorView(t1, "x");

  var(t0, t1, exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var", 0.01);

  stdd(t0, t1, exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsComplexNonHalfTypesAllExecs, VarianceStdComplex)
{
  MATX_ENTER_HANDLER();

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr index_t size = 100;
  pb->InitAndRunTVGenerator<TestType>("00_operators", "stats", "run", {size});

  tensor_t<typename TestType::value_type, 0> t0;
  tensor_t<TestType, 1> t1({size});
  pb->NumpyToTensorView(t1, "x");

  var(t0, t1, exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var", 0.01);

  stdd(t0, t1, exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNoHalfAllExecs, Sum)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  {
    int x = 273;
    int y = 2;
    int z = 4;
    auto a = matx::make_tensor<TestType, 3>(
        {x, y, z});
    auto b = matx::make_tensor<TestType, 2>(
        {x, y});
    
    (a = TestType(1)).run(exec);

    sum(b, a, {2}, exec);
    cudaStreamSynchronize(0);
    for(int i = 0 ; i < x ; i++) {
      for(int j = 0; j < y ; j++) {
        ASSERT_TRUE( MatXUtils::MatXTypeCompare(b(i,j), (TestType)z));
      }
    }
  }

  {
    tensor_t<TestType, 0> t0;

    auto t4 = ones<TestType>({3, 4, 5, 6});
    auto t3 = ones<TestType>({3, 4, 5});
    auto t2 = ones<TestType>({3, 4});
    auto t1 = ones<TestType>({3});

    sum(t0, t4, exec);
    cudaStreamSynchronize(0);
    ASSERT_TRUE(MatXUtils::MatXTypeCompare(
        t0(), (TestType)(t4.Size(0) * t4.Size(1) * t4.Size(2) * t4.Size(3))));

     sum(t0, t3, exec);
     cudaStreamSynchronize(0);
     ASSERT_TRUE(MatXUtils::MatXTypeCompare(
         t0(), (TestType)(t3.Size(0) * t3.Size(1) * t3.Size(2))));

     sum(t0, t2, exec);
     cudaStreamSynchronize(0);
     ASSERT_TRUE(
         MatXUtils::MatXTypeCompare(t0(), (TestType)(t2.Size(0) * t2.Size(1))));

     sum(t0, t1, exec);
     cudaStreamSynchronize(0);
     ASSERT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(t1.Size(0))));
  }
  {
    tensor_t<TestType, 1> t1({3});

    auto t4 = ones<TestType>({3, 4, 5, 6});
    auto t3 = ones<TestType>({3, 4, 5});
    auto t2 = ones<TestType>({3, 4});

    sum(t1, t4, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i), (TestType)(t4.Size(1) * t4.Size(2) * t4.Size(3))));
    }

    sum(t1, t3, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i), (TestType)(t3.Size(1) * t3.Size(2))));
    }

    sum(t1, t2, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(t2.Size(1))));
    }

    // Test tensor input too
    auto t2t = make_tensor<TestType>({3, 4});
    (t2t = ones<TestType>({3, 4})).run();
    sum(t1, t2t);
    for (index_t i = 0; i < t1.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(t2t.Size(1))));
    }    
  }
  {
    tensor_t<TestType, 2> t2({3, 4});

    auto t4 = ones<TestType>({3, 4, 5, 6});
    auto t3 = ones<TestType>({3, 4, 5});

    sum(t2, t4, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2(i, j), (TestType)(t4.Size(2) * t4.Size(3))));
      }
    }

    sum(t2, t3, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(t3.Size(2))));
      }
    }
  }
  
  {
    tensor_t<TestType, 2> t2a({3, 4});
    tensor_t<TestType, 2> t2b({3, 4});
    
    auto t4 = ones<TestType>({3, 4, 5, 6});

    sum(t2a, t4, exec);
    sum(t2b, t4, {2,3}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }
  
  MATX_EXIT_HANDLER();
}


// This works with half precision, but we need the proper test infrastructure to prevent compiling
// half types for CCs that don't support it. Disable half on this test for now
TYPED_TEST(ReductionTestsFloatNonComplex, Softmax)
{
  MATX_ENTER_HANDLER();

  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr index_t size = 300;
  pb->InitAndRunTVGenerator<TypeParam>("00_reductions", "softmax", "run", {80, size, size});

  tensor_t<TypeParam, 1> t1({size});
  tensor_t<TypeParam, 1> t1_out({size});
  pb->NumpyToTensorView(t1, "t1");

  softmax(t1_out, t1);

  MATX_TEST_ASSERT_COMPARE(pb, t1_out, "t1_sm", 0.01);

  auto t3    = make_tensor<TypeParam>({80,size,size});
  auto t3_out = make_tensor<TypeParam>({80,size,size});
  pb->NumpyToTensorView(t3, "t3");
  softmax(t3_out, t3, {2});
  
  MATX_TEST_ASSERT_COMPARE(pb, t3_out, "t3_sm_axis2", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, PermutedReduce)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  tensor_t<TestType, 2> t2a({50, 60});
  tensor_t<TestType, 2> t2b({50, 60});
  tensor_t<index_t, 2> t2ai({50, 60});
  tensor_t<index_t, 2> t2bi({50, 60});
  tensor_t<TestType, 4> t4({30, 40, 50, 60});

  auto t4r = make_tensor<float>({30, 40, 50, 60});
  auto t4rv = t4r.View({TotalSize(t4)});
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_real_distribution<float> ud{0,1};
  for (index_t i = 0; i < t4rv.Size(0); i++) {
    t4rv(i) = ud(e1);
  }

  (t4 = as_type<TestType>(t4r)).run(exec);
  {
    sum(t2a, permute(t4,{2,3,0,1}), exec);
    sum(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j),.1));
      }
    }
  }

  {
    mean(t2a, permute(t4,{2,3,0,1}), exec);
    mean(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

#if 0  // not currently supported
  {
    median(t2a, permute(t4,{2,3,0,1}), exec);
    median(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }
#endif

  if constexpr (!is_complex_v<TestType>)
  {
    prod(t2a, permute(t4,{2,3,0,1}), exec);
    prod(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    rmax(t2a, permute(t4,{2,3,0,1}), exec);
    rmax(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    rmin(t2a, permute(t4,{2,3,0,1}), exec);
    rmin(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    argmax(t2a, t2ai, permute(t4,{2,3,0,1}), exec);
    argmax(t2b, t2bi, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2ai(i, j), t2bi(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    argmin(t2a, t2ai, permute(t4,{2,3,0,1}), exec);
    argmin(t2b, t2bi, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2ai(i, j), t2bi(i,j)));
      }
    }
  }

  if constexpr (std::is_same_v<TestType, bool>)
  {
    any(t2a, permute(t4,{2,3,0,1}), exec);
    any(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (std::is_same_v<TestType, bool>)
  {
    all(t2a, permute(t4,{2,3,0,1}), exec);
    all(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    var(t2a, permute(t4,{2,3,0,1}), exec);
    var(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    stdd(t2a, permute(t4,{2,3,0,1}), exec);
    stdd(t2b, t4, {0,1}, exec);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNonComplexAllExecs, Any)
{
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  MATX_ENTER_HANDLER();
  {
    tensor_t<TestType, 0> t0;

    tensor_t<TestType, 1> t1({30});
    tensor_t<TestType, 2> t2({30, 40});
    tensor_t<TestType, 3> t3({30, 40, 50});
    tensor_t<TestType, 4> t4({30, 40, 50, 60});

    (t1 = zeros<TestType>(t1.Shape())).run(exec);
    (t2 = zeros<TestType>(t2.Shape())).run(exec);
    (t3 = zeros<TestType>(t3.Shape())).run(exec);
    (t4 = zeros<TestType>(t4.Shape())).run(exec);
    cudaStreamSynchronize(0);

    t1(5) = 5;
    t3(1, 1, 1) = 6;

    any(t0, t4, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(0)));

    any(t0, t3, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    any(t0, t2, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(0)));

    any(t0, t1, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNonComplexAllExecs, All)
{
  MATX_ENTER_HANDLER();

  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  {
    tensor_t<TestType, 0> t0;

    tensor_t<TestType, 1> t1({30});
    tensor_t<TestType, 2> t2({30, 40});
    tensor_t<TestType, 3> t3({30, 40, 50});
    tensor_t<TestType, 4> t4({30, 40, 50, 60});

    (t1 = ones<TestType>(t1.Shape())).run();
    (t2 = ones<TestType>(t2.Shape())).run();
    (t3 = ones<TestType>(t3.Shape())).run();
    (t4 = ones<TestType>(t4.Shape())).run();
    cudaStreamSynchronize(0);

    t1(5) = 0;
    t3(1, 1, 1) = 0;

    all(t0, t4, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    all(t0, t3, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(0)));

    all(t0, t2, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    all(t0, t1, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(0)));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Median)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  {
    tensor_t<TestType, 0> t0{};
    tensor_t<TestType, 1> t1e{{10}};
    tensor_t<TestType, 1> t1o{{11}};
    tensor_t<TestType, 2> t2e{{2, 4}};
    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1out{{2}};

    t1e.SetVals({1, 3, 8, 2, 9, 6, 7, 4, 5, 0});
    t1o.SetVals({1, 3, 8, 2, 9, 6, 7, 4, 5, 0, 10});
    t2e.SetVals({{2, 4, 1, 3}, {3, 1, 2, 4}});
    t2o.SetVals({{2, 4, 1, 3, 5}, {3, 1, 5, 2, 4}});

    median(t0, t1e, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(4.5f)));

    median(t0, t1o, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(5)));

    median(t1out, t2e, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(0), (TestType)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(1), (TestType)(2.5f)));

    median(t1out, t2o, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(0), (TestType)(3.0f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(1), (TestType)(3.0f)));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, MinMaxNegative)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;  
  {
    auto t = matx::make_tensor<TestType, 1>({3});
    t.SetVals({-3, -1, -7});

    matx::tensor_t<TestType, 0> max_val{};
    matx::tensor_t<matx::index_t, 0> max_idx{};
    matx::argmax(max_val, max_idx, t, ExecType{});
    cudaStreamSynchronize(0);
    ASSERT_EQ(max_val(), -1);
    ASSERT_EQ(max_idx(), 1);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Max)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    tensor_t<TestType, 0> t0{};
    tensor_t<index_t, 0> t0i{};    
    tensor_t<TestType, 1> t1o{{11}};
    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1o_small{{2}};    
    tensor_t<index_t, 1> t1i_small{{2}};

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});
    t2o.SetVals({{(T)2, (T)4, (T)1, (T)3, (T)5}, {(T)3, (T)1, (T)5, (T)2, (T)4}});

    rmax(t0, t1o, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(11)));    
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Min)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    tensor_t<TestType, 0> t0{};
    tensor_t<index_t, 0> t0i{};    
    tensor_t<TestType, 1> t1o{{11}};
    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1o_small{{2}};    
    tensor_t<index_t, 1> t1i_small{{2}};

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});
    t2o.SetVals({{(T)2, (T)4, (T)1, (T)3, (T)5}, {(T)3, (T)1, (T)5, (T)2, (T)4}});

    rmin(t0, t1o, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1))); 
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, ArgMax)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    tensor_t<TestType, 0> t0{};
    tensor_t<index_t, 0> t0i{};    
    tensor_t<TestType, 1> t1o{{11}};
    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1o_small{{2}};    
    tensor_t<index_t, 1> t1i_small{{2}};

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});
    t2o.SetVals({{(T)2, (T)4, (T)1, (T)3, (T)5}, {(T)3, (T)1, (T)5, (T)2, (T)4}});

    argmax(t0, t0i, t1o, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(11)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0i(), (TestType)(10)));    

    argmax(t1o_small, t1i_small, t2o, exec);
    cudaStreamSynchronize(0);

    auto rel = GetIdxFromAbs(t2o, t1i_small(0));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(5)));
    rel = GetIdxFromAbs(t2o, t1i_small(1));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(5)));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, ArgMin)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    tensor_t<TestType, 0> t0{};
    tensor_t<index_t, 0> t0i{};    
    tensor_t<TestType, 1> t1o{{11}};
    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1o_small{{2}};    
    tensor_t<index_t, 1> t1i_small{{2}};

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});
    t2o.SetVals({{(T)2, (T)4, (T)1, (T)3, (T)5}, {(T)3, (T)1, (T)5, (T)2, (T)4}});

    argmin(t0, t0i, t1o, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0i(), (TestType)(0)));    

    argmin(t1o_small, t1i_small, t2o, exec);
    cudaStreamSynchronize(0);
    
    auto rel = GetIdxFromAbs(t2o, t1i_small(0));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(1)));
    rel = GetIdxFromAbs(t2o, t1i_small(1));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(1)));  
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Mean)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  {
    tensor_t<TestType, 0> t0;

    auto t4 = ones<TestType>({30, 40, 50, 60});
    auto t3 = ones<TestType>({30, 40, 50});
    auto t2 = ones<TestType>({30, 40});
    auto t1 = ones<TestType>({30});

    mean(t0, t4, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    mean(t0, t3, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    mean(t0, t2, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    mean(t0, t1, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));
  }
  {
    tensor_t<TestType, 1> t1({30});

    auto t4 = ones<TestType>({30, 40, 50, 60});
    auto t3 = ones<TestType>({30, 40, 50});
    auto t2 = ones<TestType>({30, 40});

    mean(t1, t4, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(1)));
    }

    mean(t1, t3, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(1)));
    }

    mean(t1, t2, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(1)));
    }
  }

  {
    tensor_t<TestType, 2> t2({30, 40});

    auto t4 = ones<TestType>({30, 40, 50, 60});
    auto t3 = ones<TestType>({30, 40, 50});

    mean(t2, t4, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(1)));
      }
    }

    mean(t2, t3, exec);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(1)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNonComplexAllExecs, Prod)
{
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec;

  MATX_ENTER_HANDLER();
  {
    tensor_t<TestType, 0> t0;

    std::array<index_t, 2> s2{3, 4};
    std::array<index_t, 1> s1{3};
    auto t1 = make_tensor<TestType>(s1);
    auto t2 = make_tensor<TestType>(s2);

    TestType t1p = (TestType)1;
    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TestType>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
      t1p *= t1(i);
    }

    TestType t2p = (TestType)1;
    for (int i = 0; i < t2.Size(0); i++) {
      for (int j = 0; j < t2.Size(1); j++) {
        t2(i, j) = static_cast<detail::value_promote_t<TestType>>(
            (float)rand() / (float)INT_MAX * 2.0f);
        t2p *= t2(i, j);
      }
    }

    prod(t0, t2, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), t2p));

    prod(t0, t1, exec);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), t1p));
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Find)
{
  MATX_ENTER_HANDLER();
  {
    using TestType = std::tuple_element_t<0, TypeParam>;
    using ExecType = std::tuple_element_t<1, TypeParam>;

    tensor_t<int, 0> num_found{};
    tensor_t<TestType, 1> t1{{100}};
    tensor_t<TestType, 1> t1o{{100}};
    TestType thresh = (TestType)0.5;


    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TestType>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // Find values greater than 0.5
    find(t1o, num_found, t1, GT{thresh}, ExecType{});
    cudaStreamSynchronize(0);
    
    int output_found = 0;
    for (int i = 0; i < t1.Size(0); i++) {
      if (t1(i) > thresh) {
        ASSERT_NEAR(t1o(output_found), t1(i), 0.01);
        output_found++;
      }
    }
    ASSERT_EQ(output_found, num_found());

  }

  MATX_EXIT_HANDLER();
}



TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, FindIdx)
{
  MATX_ENTER_HANDLER();
  {
    using TestType = std::tuple_element_t<0, TypeParam>;
    using ExecType = std::tuple_element_t<1, TypeParam>;

    tensor_t<int, 0> num_found{};
    tensor_t<TestType, 1> t1{{100}};
    tensor_t<int, 1> t1o{{100}};
    TestType thresh = (TestType)0.5;


    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TestType>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // Find values greater than 0.5
    find_idx(t1o, num_found, t1, GT{thresh}, ExecType{});
    cudaStreamSynchronize(0);
    
    int output_found = 0;
    for (int i = 0; i < t1.Size(0); i++) {
      if (t1(i) > thresh) {
        ASSERT_EQ(t1o(output_found), i);
        output_found++;
      }
    }
    ASSERT_EQ(output_found, num_found());

  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, FindIdxAndSelect)
{
  MATX_ENTER_HANDLER();
  {
    using TestType = std::tuple_element_t<0, TypeParam>;
    using ExecType = std::tuple_element_t<1, TypeParam>;

    tensor_t<int, 0> num_found{}, num_found2{};
    tensor_t<TestType, 1> t1{{100}};
    tensor_t<int, 1> t1o_idx{{100}};
    tensor_t<TestType, 1> t1o{{100}};
    tensor_t<TestType, 1> t1o_2{{100}};
    TestType thresh = (TestType)0.5;

    auto executor = ExecType{};

    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TestType>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // Find indices with values greater than 0.5
    find_idx(t1o_idx, num_found, t1, GT{thresh}, executor);
    cudaStreamSynchronize(0);

    auto t1o_slice = t1o.Slice({0}, {num_found()});
    auto t1o_idx_slice = t1o_idx.Slice({0}, {num_found()});
    (t1o_slice = select(t1o_slice, t1o_idx_slice)).run(executor);

    // Compare to simply finding the values
    find(t1o_2, num_found2, t1, GT{thresh}, executor);
    cudaStreamSynchronize(0);

    ASSERT_EQ(num_found(), num_found2());
    
    for (int i = 0; i < t1o_slice.Size(0); i++) {
      ASSERT_EQ(t1o(i), t1o_slice(i));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Unique)
{
  MATX_ENTER_HANDLER();
  {
    using TestType = std::tuple_element_t<0, TypeParam>;
    using ExecType = std::tuple_element_t<1, TypeParam>;

    tensor_t<int, 0> num_found{};
    tensor_t<TestType, 1> t1{{100}};
    tensor_t<TestType, 1> t1o{{100}};

    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = (TestType)(i % 10);
    }

    // Find values greater than 0
    unique(t1o, num_found, t1, ExecType{});
    cudaStreamSynchronize(0);

    for (int i = 0; i < 10; i++) {
      ASSERT_NEAR(t1o(i), i, 0.01);
    }

    ASSERT_EQ(10, num_found());

  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Trace)
{
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;

  ExecType exec;
    
  MATX_ENTER_HANDLER();
  index_t count = 10;
  TestType c = GenerateData<TestType>();

  tensor_t<TestType, 2> t2({count, count});
  auto t0 = make_tensor<TestType>();

  (t2 = ones<TestType>(t2.Shape())).run(exec);
  trace(t0, t2, exec);

  cudaDeviceSynchronize();

  ASSERT_EQ(t0(), count);
  MATX_EXIT_HANDLER();
}


