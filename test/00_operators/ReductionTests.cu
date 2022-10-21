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


TYPED_TEST(ReductionTestsFloatNonComplexNonHalf, VarianceStd)
{
  MATX_ENTER_HANDLER();

  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr index_t size = 100;
  pb->InitAndRunTVGenerator<TypeParam>("00_operators", "stats", "run", {size});

  tensor_t<TypeParam, 0> t0;
  tensor_t<TypeParam, 1> t1({size});
  pb->NumpyToTensorView(t1, "x");

  var(t0, t1, 0);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var", 0.01);

  stdd(t0, t1, 0);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsComplexNonHalfTypes, VarianceStdComplex)
{
  MATX_ENTER_HANDLER();

  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr index_t size = 100;
  pb->InitAndRunTVGenerator<TypeParam>("00_operators", "stats", "run", {size});

  tensor_t<typename TypeParam::value_type, 0> t0;
  tensor_t<TypeParam, 1> t1({size});
  pb->NumpyToTensorView(t1, "x");

  var(t0, t1, 0);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var", 0.01);

  stdd(t0, t1, 0);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNoHalf, Sum)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<TypeParam, 0> t0;

    auto t4 = ones<TypeParam>({30, 40, 50, 60});
    auto t3 = ones<TypeParam>({30, 40, 50});
    auto t2 = ones<TypeParam>({30, 40});
    auto t1 = ones<TypeParam>({30});

    sum(t0, t4, 0);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(
        t0(), (TypeParam)(t4.Size(0) * t4.Size(1) * t4.Size(2) * t4.Size(3))));

     sum(t0, t3, 0);
     cudaStreamSynchronize(0);
     EXPECT_TRUE(MatXUtils::MatXTypeCompare(
         t0(), (TypeParam)(t3.Size(0) * t3.Size(1) * t3.Size(2))));

     sum(t0, t2, 0);
     cudaStreamSynchronize(0);
     EXPECT_TRUE(
         MatXUtils::MatXTypeCompare(t0(), (TypeParam)(t2.Size(0) * t2.Size(1))));

     sum(t0, t1, 0);
     cudaStreamSynchronize(0);
     EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(t1.Size(0))));
  }
  {
    tensor_t<TypeParam, 1> t1({30});

    auto t4 = ones<TypeParam>({30, 40, 50, 60});
    auto t3 = ones<TypeParam>({30, 40, 50});
    auto t2 = ones<TypeParam>({30, 40});

    sum(t1, t4, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i), (TypeParam)(t4.Size(1) * t4.Size(2) * t4.Size(3))));
    }

    sum(t1, t3, 0);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i), (TypeParam)(t3.Size(1) * t3.Size(2))));
    }

    sum(t1, t2, 0);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(t2.Size(1))));
    }

    // Test tensor input too
    auto t2t = make_tensor<TypeParam>({30, 40});
    (t2t = ones<TypeParam>({30, 40})).run();
    sum(t1, t2t);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(t2t.Size(1))));
    }    
  }

  {
    tensor_t<TypeParam, 2> t2({30, 40});

    auto t4 = ones<TypeParam>({30, 40, 50, 60});
    auto t3 = ones<TypeParam>({30, 40, 50});

    sum(t2, t4, 0);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2(i, j), (TypeParam)(t4.Size(2) * t4.Size(3))));
      }
    }

    sum(t2, t3, 0);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2(i, j), (TypeParam)(t3.Size(2))));
      }
    }
  }
  
  {
    tensor_t<TypeParam, 2> t2a({30, 40});
    tensor_t<TypeParam, 2> t2b({30, 40});
    
    auto t4 = ones<TypeParam>({30, 40, 50, 60});

    sum(t2a, t4, 0);
    sum(t2b, t4, {2,3}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  
  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNoHalf, PermutedReduce)
{
  MATX_ENTER_HANDLER();


  tensor_t<TypeParam, 2> t2a({50, 60});
  tensor_t<TypeParam, 2> t2b({50, 60});
  
  tensor_t<int, 2> t2ai({50, 60});
  tensor_t<int, 2> t2bi({50, 60});

  tensor_t<TypeParam, 4> t4({30, 40, 50, 60});
  
  randomGenerator_t<float> random(30*40*50*60, 0);
  auto t4r = random.GetTensorView<4>({30,40,50,60}, UNIFORM, 100);

  (t4 = as_type<TypeParam>(t4r)).run();
  
  {
    sum(t2a, permute(t4,{2,3,0,1}), 0);
    sum(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j),.1));
      }
    }
  }
 
  {
    mean(t2a, permute(t4,{2,3,0,1}), 0);
    mean(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

#if 0  // Rank4 not supported at this time
  {
    median(t2a, permute(t4,{2,3,0,1}), 0);
    median(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }
#endif

  if constexpr (!is_complex_v<TypeParam>)
  {
    prod(t2a, permute(t4,{2,3,0,1}), 0);
    prod(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TypeParam>)
  {
    rmax(t2a, permute(t4,{2,3,0,1}), 0);
    rmax(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }
  
  if constexpr (!is_complex_v<TypeParam>)
  {
    rmin(t2a, permute(t4,{2,3,0,1}), 0);
    rmin(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }
  
  if constexpr (!is_complex_v<TypeParam>)
  {
    argmax(t2a, t2ai, permute(t4,{2,3,0,1}), 0);
    argmax(t2b, t2bi, t4, {0,1}, 0);

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
  
  if constexpr (!is_complex_v<TypeParam>)
  {
    argmin(t2a, t2ai, permute(t4,{2,3,0,1}), 0);
    argmin(t2b, t2bi, t4, {0,1}, 0);

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
  
  if constexpr (std::is_same_v<TypeParam, bool>)
  {
    any(t2a, permute(t4,{2,3,0,1}), 0);
    any(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }
  
  if constexpr (std::is_same_v<TypeParam, bool>)
  {
    all(t2a, permute(t4,{2,3,0,1}), 0);
    all(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }
 
  if constexpr (!is_complex_v<TypeParam>)
  {
    var(t2a, permute(t4,{2,3,0,1}), 0);
    var(t2b, t4, {0,1}, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }
  
  if constexpr (!is_complex_v<TypeParam>)
  {
    stdd(t2a, permute(t4,{2,3,0,1}), 0);
    stdd(t2b, t4, {0,1}, 0);

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

TEST(ReductionTests, Any)
{
  MATX_ENTER_HANDLER();
  using TypeParam = int;
  {
    tensor_t<TypeParam, 0> t0;

    tensor_t<int, 1> t1({30});
    tensor_t<int, 2> t2({30, 40});
    tensor_t<int, 3> t3({30, 40, 50});
    tensor_t<int, 4> t4({30, 40, 50, 60});

    (t1 = zeros<int>(t1.Shape())).run();
    (t2 = zeros<int>(t2.Shape())).run();
    (t3 = zeros<int>(t3.Shape())).run();
    (t4 = zeros<int>(t4.Shape())).run();
    cudaStreamSynchronize(0);

    t1(5) = 5;
    t3(1, 1, 1) = 6;

    any(t0, t4);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(0)));

    any(t0, t3);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    any(t0, t2);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(0)));

    any(t0, t1);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));
  }

  MATX_EXIT_HANDLER();
}

TEST(ReductionTests, All)
{
  MATX_ENTER_HANDLER();

  using TypeParam = int;
  {
    tensor_t<int, 0> t0;

    tensor_t<int, 1> t1({30});
    tensor_t<int, 2> t2({30, 40});
    tensor_t<int, 3> t3({30, 40, 50});
    tensor_t<int, 4> t4({30, 40, 50, 60});

    (t1 = ones<int>(t1.Shape())).run();
    (t2 = ones<int>(t2.Shape())).run();
    (t3 = ones<int>(t3.Shape())).run();
    (t4 = ones<int>(t4.Shape())).run();
    cudaStreamSynchronize(0);

    t1(5) = 0;
    t3(1, 1, 1) = 0;

    all(t0, t4);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    all(t0, t3);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(0)));

    all(t0, t2);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    all(t0, t1);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(0)));
  }

  MATX_EXIT_HANDLER();
}

TEST(ReductionTests, Median)
{
  MATX_ENTER_HANDLER();
  using TypeParam = float;
  {
    tensor_t<TypeParam, 0> t0{};
    tensor_t<TypeParam, 1> t1e{{10}};
    tensor_t<TypeParam, 1> t1o{{11}};
    tensor_t<TypeParam, 2> t2e{{2, 4}};
    tensor_t<TypeParam, 2> t2o{{2, 5}};
    tensor_t<TypeParam, 1> t1out{{2}};

    t1e.SetVals({1, 3, 8, 2, 9, 6, 7, 4, 5, 0});
    t1o.SetVals({1, 3, 8, 2, 9, 6, 7, 4, 5, 0, 10});
    t2e.SetVals({{2, 4, 1, 3}, {3, 1, 2, 4}});
    t2o.SetVals({{2, 4, 1, 3, 5}, {3, 1, 5, 2, 4}});

    median(t0, t1e);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(4.5f)));

    median(t0, t1o);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(5)));

    median(t1out, t2e);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(0), (TypeParam)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(1), (TypeParam)(2.5f)));

    median(t1out, t2o);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(0), (TypeParam)(3.0f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(1), (TypeParam)(3.0f)));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalf, MinMaxNegative)
{
  MATX_ENTER_HANDLER();
  {
    auto t = matx::make_tensor<TypeParam, 1>({3});
    t.SetVals({-3, -1, -7});

    matx::tensor_t<float, 0> max_val{};
    matx::tensor_t<matx::index_t, 0> max_idx{};
    matx::argmax(max_val, max_idx, t);
    cudaStreamSynchronize(0);
    ASSERT_EQ(max_val(), -1);
    ASSERT_EQ(max_idx(), 1);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNonComplex, MinMax)
{
  MATX_ENTER_HANDLER();
  {
    using T = TypeParam;
    tensor_t<TypeParam, 0> t0{};
    tensor_t<index_t, 0> t0i{};    
    tensor_t<TypeParam, 1> t1o{{11}};
    tensor_t<TypeParam, 2> t2o{{2, 5}};
    tensor_t<TypeParam, 1> t1o_small{{2}};    
    tensor_t<index_t, 1> t1i_small{{2}};

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});
    t2o.SetVals({{(T)2, (T)4, (T)1, (T)3, (T)5}, {(T)3, (T)1, (T)5, (T)2, (T)4}});

    rmin(t0, t1o);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    rmax(t0, t1o);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(11)));    

    argmax(t0, t0i, t1o);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(11)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0i(), (TypeParam)(10)));

    argmin(t0, t0i, t1o);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0i(), (TypeParam)(0)));    

    argmax(t1o_small, t1i_small, t2o);
    cudaStreamSynchronize(0);

    // We need to convert the absolute index into relative before comparing
    auto rel = GetIdxFromAbs(t2o, t1i_small(0));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TypeParam)(5)));
    rel = GetIdxFromAbs(t2o, t1i_small(1));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TypeParam)(5)));

    argmin(t1o_small, t1i_small, t2o);
    cudaStreamSynchronize(0);
    
    rel = GetIdxFromAbs(t2o, t1i_small(0));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TypeParam)(1)));
    rel = GetIdxFromAbs(t2o, t1i_small(1));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TypeParam)(1)));  
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalf, Mean)
{
  MATX_ENTER_HANDLER();
  using T = TypeParam;
  {
    tensor_t<TypeParam, 0> t0;

    auto t4 = ones<T>({30, 40, 50, 60});
    auto t3 = ones<T>({30, 40, 50});
    auto t2 = ones<T>({30, 40});
    auto t1 = ones<T>({30});

    mean(t0, t4);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    mean(t0, t3);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    mean(t0, t2);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    mean(t0, t1);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));
  }
  {
    tensor_t<TypeParam, 1> t1({30});

    auto t4 = ones<T>({30, 40, 50, 60});
    auto t3 = ones<T>({30, 40, 50});
    auto t2 = ones<T>({30, 40});

    mean(t1, t4);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(1)));
    }

    mean(t1, t3);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(1)));
    }

    mean(t1, t2);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(1)));
    }
  }

  {
    tensor_t<TypeParam, 2> t2({30, 40});

    auto t4 = ones<T>({30, 40, 50, 60});
    auto t3 = ones<T>({30, 40, 50});

    mean(t2, t4);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TypeParam)(1)));
      }
    }

    mean(t2, t3);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TypeParam)(1)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNonComplex, Prod)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<TypeParam, 0> t0;

    std::array<index_t, 2> s2{3, 4};
    std::array<index_t, 1> s1{3};
    auto t1 = make_tensor<TypeParam>(s1);
    auto t2 = make_tensor<TypeParam>(s2);

    TypeParam t1p = (TypeParam)1;
    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TypeParam>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
      t1p *= t1(i);
    }

    TypeParam t2p = (TypeParam)1;
    for (int i = 0; i < t2.Size(0); i++) {
      for (int j = 0; j < t2.Size(1); j++) {
        t2(i, j) = static_cast<detail::value_promote_t<TypeParam>>(
            (float)rand() / (float)INT_MAX * 2.0f);
        t2p *= t2(i, j);
      }
    }

    prod(t0, t2);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), t2p));

    prod(t0, t1);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), t1p));
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(ReductionTestsNumericNonComplex, Find)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<int, 0> num_found{};
    tensor_t<TypeParam, 1> t1{{100}};
    tensor_t<TypeParam, 1> t1o{{100}};
    TypeParam thresh = (TypeParam)0.5;


    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TypeParam>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // Find values greater than 0.5
    find(t1o, num_found, t1, GT{thresh});
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

TYPED_TEST(ReductionTestsNumericNonComplex, FindIdx)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<int, 0> num_found{};
    tensor_t<TypeParam, 1> t1{{100}};
    tensor_t<int, 1> t1o{{100}};
    TypeParam thresh = (TypeParam)0.5;


    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TypeParam>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // Find values greater than 0.5
    find_idx(t1o, num_found, t1, GT{thresh});
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

TYPED_TEST(ReductionTestsNumericNonComplex, FindIdxAndSelect)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<int, 0> num_found{}, num_found2{};
    tensor_t<TypeParam, 1> t1{{100}};
    tensor_t<int, 1> t1o_idx{{100}};
    tensor_t<TypeParam, 1> t1o{{100}};
    tensor_t<TypeParam, 1> t1o_2{{100}};
    TypeParam thresh = (TypeParam)0.5;


    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TypeParam>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // Find indices with values greater than 0.5
    find_idx(t1o_idx, num_found, t1, GT{thresh});
    cudaStreamSynchronize(0);

    auto t1o_slice = t1o.Slice({0}, {num_found()});
    auto t1o_idx_slice = t1o_idx.Slice({0}, {num_found()});
    (t1o_slice = select(t1o_slice, t1o_idx_slice)).run();

    // Compare to simply finding the values
    find(t1o_2, num_found2, t1, GT{thresh});
    cudaStreamSynchronize(0);

    ASSERT_EQ(num_found(), num_found2());
    
    for (int i = 0; i < t1o_slice.Size(0); i++) {
      ASSERT_EQ(t1o(i), t1o_slice(i));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNonComplex, Unique)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<int, 0> num_found{};
    tensor_t<TypeParam, 1> t1{{100}};
    tensor_t<TypeParam, 1> t1o{{100}};

    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = (TypeParam)(i % 10);
    }

    // Find values greater than 0
    unique(t1o, num_found, t1);
    cudaStreamSynchronize(0);

    for (int i = 0; i < 10; i++) {
      ASSERT_NEAR(t1o(i), i, 0.01);
    }

    ASSERT_EQ(10, num_found());

  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalf, Trace)
{
  MATX_ENTER_HANDLER();
  index_t count = 10;
  TypeParam c = GenerateData<TypeParam>();

  tensor_t<TypeParam, 2> t2({count, count});
  auto t0 = make_tensor<TypeParam>();

  (t2 = ones(t2.Shape())).run();
  trace(t0, t2);

  cudaDeviceSynchronize();

  ASSERT_EQ(t0(), count);
  MATX_EXIT_HANDLER();
}


