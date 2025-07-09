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

TYPED_TEST_SUITE(ReductionTestsAll, MatXAllTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsComplex, MatXComplexTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsFloat, MatXFloatTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsNumeric, MatXNumericTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsIntegral, MatXAllIntegralTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsNumericNonComplex,
                 MatXNumericNonComplexTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsFloatNonComplex, MatXFloatNonComplexTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsFloatNonComplexNonHalf,
                 MatXFloatNonComplexNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsBoolean, MatXBoolTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsFloatHalf, MatXFloatHalfTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsNumericNoHalf, MatXNumericNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(ReductionTestsComplexNonHalfTypes, MatXComplexNonHalfTypesCUDAExec);


TYPED_TEST_SUITE(ReductionTestsNumericNonComplexAllExecs,
                 MatXNumericNonComplexTypesAllExecs);
TYPED_TEST_SUITE(ReductionTestsFloatNonComplexNonHalfAllExecs,
                 MatXFloatNonComplexNonHalfTypesAllExecs);
TYPED_TEST_SUITE(ReductionTestsNumericNoHalfAllExecs, MatXNumericNoHalfTypesAllExecs);
TYPED_TEST_SUITE(ReductionTestsComplexNonHalfTypesAllExecs, MatXComplexNonHalfTypesAllExecs);


TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, VarianceStd)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr index_t size = 100;
  pb->InitAndRunTVGenerator<TestType>("00_operators", "stats", "run", {size});

  auto t0 = make_tensor<TestType>({});
  tensor_t<TestType, 1> t1({size});
  pb->NumpyToTensorView(t1, "x");

  // example-begin var-test-1
  (t0 = var(t1)).run(exec);
  // example-end var-test-1
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var_ub", 0.01);

  (t0 = var(t1, 0)).run(exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var_ml", 0.01);

  // example-begin stdd-test-1
  (t0 = stdd(t1)).run(exec);
  // example-end stdd-test-1
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std_ub", 0.01);

  (t0 = stdd(t1, 0)).run(exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std_ml", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsComplexNonHalfTypesAllExecs, VarianceStdComplex)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr index_t size = 100;
  pb->InitAndRunTVGenerator<TestType>("00_operators", "stats", "run", {size});

  auto t0 = make_tensor<typename TestType::value_type>({});
  tensor_t<TestType, 1> t1({size});
  pb->NumpyToTensorView(t1, "x");

  (t0 = var(t1)).run(exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var_ub", 0.01);

  (t0 = var(t1, 0)).run(exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var_ml", 0.01);

  (t0 = stdd(t1)).run(exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std_ub", 0.01);

  (t0 = stdd(t1, 0)).run(exec);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std_ml", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNoHalfAllExecs, Sum)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

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

    // example-begin sum-test-2
    // Reduce a 3D tensor into a 2D by taking the sum of the last dimension
    (b = sum(a, {2})).run(exec);
    // example-end sum-test-2
    exec.sync();
    for(int i = 0 ; i < x ; i++) {
      for(int j = 0; j < y ; j++) {
        ASSERT_TRUE( MatXUtils::MatXTypeCompare(b(i,j), (TestType)z));
      }
    }
  }

  {
    auto t0 = make_tensor<TestType>({});

    auto t4 = ones<TestType>({3, 4, 5, 6});
    auto t3 = ones<TestType>({3, 4, 5});
    auto t2 = ones<TestType>({3, 4});
    auto t1 = ones<TestType>({3});

    // example-begin sum-test-1
    // Reduce a 4D tensor into a 0D by taking the sum of all elements
    (t0 = sum(t4)).run(exec);
    // example-end sum-test-1
    exec.sync();
    ASSERT_TRUE(MatXUtils::MatXTypeCompare(
        t0(), (TestType)(t4.Size(0) * t4.Size(1) * t4.Size(2) * t4.Size(3))));

     (t0 = sum(t3)).run(exec);
     exec.sync();
     ASSERT_TRUE(MatXUtils::MatXTypeCompare(
         t0(), (TestType)(t3.Size(0) * t3.Size(1) * t3.Size(2))));

     (t0 = sum(t2)).run(exec);
     exec.sync();
     ASSERT_TRUE(
         MatXUtils::MatXTypeCompare(t0(), (TestType)(t2.Size(0) * t2.Size(1))));

     (t0 = sum(t1)).run(exec);
     exec.sync();
     ASSERT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(t1.Size(0))));
  }
  {
    tensor_t<TestType, 1> t1({3});

    auto t4 = ones<TestType>({3, 4, 5, 6});
    auto t3 = ones<TestType>({3, 4, 5});
    auto t2 = ones<TestType>({3, 4});

    (t1 = sum(t4, {1, 2, 3})).run(exec);

    exec.sync();
    for (index_t i = 0; i < t1.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i), (TestType)(t4.Size(1) * t4.Size(2) * t4.Size(3))));
    }

    (t1 = sum(t3, {1, 2})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t1.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i), (TestType)(t3.Size(1) * t3.Size(2))));
    }

    (t1 = sum(t2, {1})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t1.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(t2.Size(1))));
    }

    // Test tensor input too
    auto t2t = make_tensor<TestType>({3, 4});
    (t2t = ones<TestType>({3, 4})).run(exec);
    (t1 = sum(t2t, {1})).run(exec);
    exec.sync();

    for (index_t i = 0; i < t1.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(t2t.Size(1))));
    }
  }
  {
    tensor_t<TestType, 2> t2({3, 4});

    auto t4 = ones<TestType>({3, 4, 5, 6});
    auto t3 = ones<TestType>({3, 4, 5});

    (t2 = sum(t4, {2, 3})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2(i, j), (TestType)(t4.Size(2) * t4.Size(3))));
      }
    }

    (t2 = sum(t3, {2})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(t3.Size(2))));
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
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr index_t size = 300;
  pb->InitAndRunTVGenerator<TestType>("00_reductions", "softmax", "run", {80, size, size});

  tensor_t<TestType, 1> t1({size});
  tensor_t<TestType, 1> t1_out({size});
  pb->NumpyToTensorView(t1, "t1");

  // example-begin softmax-test-1
  (t1_out = softmax(t1)).run(exec);
  // example-end softmax-test-1

  MATX_TEST_ASSERT_COMPARE(pb, t1_out, "t1_sm", 0.01);

  auto t3    = make_tensor<TestType>({80,size,size});
  auto t3_out = make_tensor<TestType>({80,size,size});
  pb->NumpyToTensorView(t3, "t3");
  // example-begin softmax-test-2
  (t3_out = softmax(t3, {2})).run(exec);
  // example-end softmax-test-2

  MATX_TEST_ASSERT_COMPARE(pb, t3_out, "t3_sm_axis2", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, PermutedReduce)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

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
    (t2a = sum(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = sum(t4, {0,1})).run(exec);

    exec.sync();
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j),.1));
      }
    }
  }

  {
    (t2a = mean(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = mean(t4, {0,1})).run(exec);

    exec.sync();
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

    exec.sync();
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
    (t2a = prod(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = prod(t4, {0,1})).run(exec);

    exec.sync();
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    // example-begin max-test-2
    // Reduce a 4D tensor into a 2D tensor by collapsing the inner two dimensions. Both
    // examples permute the dimensions before the reduction
    (t2a = max(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = max(t4, {0,1})).run(exec);
    // example-end max-test-2

    exec.sync();
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    // example-begin min-test-2
    // Reduce a 4D tensor into a 2D tensor by collapsing the inner two dimensions. Both
    // examples permute the dimensions before the reduction
    (t2a = min(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = min(t4, {0,1})).run(exec);
    // example-end min-test-2

    exec.sync();
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    // example-begin argmax-test-2
    // Reduce a 4D tensor into a 2D tensor by collapsing the inner two dimensions. Both
    // examples permute the dimensions before the reduction
    (mtie(t2a, t2ai) = argmax(permute(t4,{2,3,0,1}))).run(exec);
    (mtie(t2b, t2bi) = argmax(t4, {0,1})).run(exec);
    // example-end argmax-test-2

    exec.sync();
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
    // example-begin argmin-test-2
    // Reduce a 4D tensor into a 2D tensor by collapsing the inner two dimensions. Both
    // examples permute the dimensions before the reduction
    (mtie(t2a, t2ai) = argmin(permute(t4,{2,3,0,1}))).run(exec);
    (mtie(t2b, t2bi) = argmin(t4, {0,1})).run(exec);
    // example-end argmin-test-2

    exec.sync();
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
    // example-begin any-test-2
    // Reduce a 4D tensor into a 2D tensor by collapsing the inner two dimensions. Both
    // examples permute the dimensions before the reduction
    (t2a = any(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = any(t4, {0,1})).run(exec);
    // example-end any-test-2

    exec.sync();
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (std::is_same_v<TestType, bool>)
  {
    // example-begin all-test-2
    // Reduce a 4D tensor into a 2D tensor by collapsing the inner two dimensions. Both
    // examples permute the dimensions before the reduction
    (t2a = all(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = all(t4, {0,1})).run(exec);
    // example-end all-test-2

    exec.sync();
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    (t2a = var(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = var(t4, {0,1})).run(exec);

    exec.sync();
    for (index_t i = 0; i < t2a.Size(0); i++) {
      for (index_t j = 0; j < t2a.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2a(i, j), t2b(i,j)));
      }
    }
  }

  if constexpr (!is_complex_v<TestType>)
  {
    (t2a = stdd(permute(t4,{2,3,0,1}))).run(exec);
    (t2b = stdd(t4, {0,1})).run(exec);

    exec.sync();
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
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  MATX_ENTER_HANDLER();
  {
    auto t0 = make_tensor<TestType>({});

    tensor_t<TestType, 1> t1({30});
    tensor_t<TestType, 2> t2({30, 40});
    tensor_t<TestType, 3> t3({30, 40, 50});
    tensor_t<TestType, 4> t4({30, 40, 50, 60});

    (t1 = zeros<TestType>(t1.Shape())).run(exec);
    (t2 = zeros<TestType>(t2.Shape())).run(exec);
    (t3 = zeros<TestType>(t3.Shape())).run(exec);
    (t4 = zeros<TestType>(t4.Shape())).run(exec);
    exec.sync();

    t1(5) = 5;
    t3(1, 1, 1) = 6;

    // example-begin any-test-1
    // Reduce a 4D tensor into a single output (0D) tensor indicating whether any values were
    // convertible to "true"
    (t0 = any(t4)).run(exec);
    // example-end any-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(0)));

    (t0 = any(t3)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    (t0 = any(t2)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(0)));

    (t0 = any(t1)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    // test partial reduction
    (t2 = any(t4, {2, 3})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(0)));
      }
    }

    (t2 = any(t3, {2})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(i == 1 && j == 1)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, AllClose)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  // example-begin allclose-test-1
  auto A = make_tensor<TestType>({5, 5, 5});
  auto B = make_tensor<TestType>({5, 5, 5});
  auto C = make_tensor<int>({});

  (A = ones<TestType>(A.Shape())).run(exec);
  (B = ones<TestType>(B.Shape())).run(exec);
  allclose(C, A, B, 1e-5, 1e-8, exec);
  // example-end allclose-test-1
  exec.sync();

  ASSERT_EQ(C(), 1);

  B(1,1,1) = 2;
  allclose(C, A, B, 1e-5, 1e-8, exec);
  exec.sync();

  ASSERT_EQ(C(), 0);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsNumericNonComplexAllExecs, All)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  {
    auto t0 = make_tensor<TestType>({});

    tensor_t<TestType, 1> t1({30});
    tensor_t<TestType, 2> t2({30, 40});
    tensor_t<TestType, 3> t3({30, 40, 50});
    tensor_t<TestType, 4> t4({30, 40, 50, 60});

    (t1 = ones<TestType>(t1.Shape())).run(exec);
    (t2 = ones<TestType>(t2.Shape())).run(exec);
    (t3 = ones<TestType>(t3.Shape())).run(exec);
    (t4 = ones<TestType>(t4.Shape())).run(exec);
    exec.sync();

    t1(5) = 0;
    t3(1, 1, 1) = 0;

    // example-begin all-test-1
    // Reduce a 4D tensor into a 0D tensor where the 0D is "true" if all values in "t4"
    // convert to "true", or "false" otherwise
    (t0 = all(t4)).run(exec);
    // example-end all-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    (t0 = all(t3)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(0)));

    (t0 = all(t2)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    (t0 = all(t1)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(0)));

    // test partial reduction
    (t2 = all(t4, {2, 3})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(1)));
      }
    }

    (t2 = all(t3, {2})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(i != 1 || j != 1)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Percentile)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  auto pb = std::make_unique<detail::MatXPybind>();
  const index_t dsize = 6;
  pb->InitAndRunTVGenerator<TestType>("00_reductions", "percentile", "run", {dsize});

  ExecType exec{};

  MATX_ENTER_HANDLER();
  {
    auto t1e = make_tensor<TestType>({dsize});
    auto t1o = make_tensor<TestType>({dsize+1});
    auto t0 = make_tensor<TestType>({});
    pb->NumpyToTensorView(t1e, "t1e");
    pb->NumpyToTensorView(t1o, "t1o");

    // example-begin percentile-test-1
    // Find the 50th percentile value in `t1e` using linear interpolation between midpoints
    (t0 = percentile(t1e, 50, PercentileMethod::LINEAR)).run(exec);
    // example-end percentile-test-1
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1e_linear50", 0.01);

    (t0 = percentile(t1e, 80, PercentileMethod::LINEAR)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1e_linear80", 0.01);

    (t0 = percentile(t1e, 50, PercentileMethod::LOWER)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1e_lower50", 0.01);

    (t0 = percentile(t1e, 80, PercentileMethod::LOWER)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1e_lower80", 0.01);

    (t0 = percentile(t1e, 50, PercentileMethod::HIGHER)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1e_higher50", 0.01);

    (t0 = percentile(t1e, 80, PercentileMethod::HIGHER)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1e_higher80", 0.01);

    (t0 = percentile(t1o, 50, PercentileMethod::LINEAR)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1o_linear50", 0.01);

    (t0 = percentile(t1o, 80, PercentileMethod::LINEAR)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1o_linear80", 0.01);

    (t0 = percentile(t1o, 50, PercentileMethod::LOWER)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1o_lower50", 0.01);

    (t0 = percentile(t1o, 80, PercentileMethod::LOWER)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1o_lower80", 0.01);

    (t0 = percentile(t1o, 50, PercentileMethod::HIGHER)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1o_higher50", 0.01);

    (t0 = percentile(t1o, 80, PercentileMethod::HIGHER)).run(exec);
    exec.sync();
    MATX_TEST_ASSERT_COMPARE(pb, t0, "t1o_higher80", 0.01);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Median)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  {
    tensor_t<TestType, 0> t0{{}};
    tensor_t<TestType, 1> t1e{{10}};
    tensor_t<TestType, 1> t1o{{11}};
    tensor_t<TestType, 2> t2e{{2, 4}};
    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1out{{2}};
    tensor_t<TestType, 1> t4out{{4}};
    tensor_t<TestType, 1> t5out{{5}};

    t1e.SetVals({1, 3, 8, 2, 9, 6, 7, 4, 5, 0});
    t1o.SetVals({1, 3, 8, 2, 9, 6, 7, 4, 5, 0, 10});
    t2e.SetVals({{2, 4, 1, 3}, {3, 1, 2, 4}});
    t2o.SetVals({{2, 4, 1, 3, 5}, {3, 1, 5, 2, 4}});

    // example-begin median-test-1
    // Compute media over all elements in "t1e" and store result in "t0"
    (t0 = median(t1e)).run(exec);
    // example-end median-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(4.5f)));

    (t0 = median(t1o)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(5)));

    (t1out = median(t2e, {1})).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(0), (TestType)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(1), (TestType)(2.5f)));

    (t4out = median(t2e, {0})).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t4out(0), (TestType)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t4out(1), (TestType)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t4out(2), (TestType)(1.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t4out(3), (TestType)(3.5f)));

    (t1out = median(t2o, {1})).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(0), (TestType)(3.0f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(1), (TestType)(3.0f)));

    (t5out = median(t2o, {0})).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t5out(0), (TestType)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t5out(1), (TestType)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t5out(2), (TestType)(3.0f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t5out(3), (TestType)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t5out(4), (TestType)(4.5f)));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, MinMaxNegative)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  {
    auto t = matx::make_tensor<TestType, 1>({3});
    t.SetVals({-3, -1, -7});

    ExecType exec{};

    matx::tensor_t<TestType, 0> max_val{{}};
    matx::tensor_t<matx::index_t, 0> max_idx{{}};
    (mtie(max_val, max_idx) = matx::argmax(t)).run(exec);
    exec.sync();
    ASSERT_EQ(max_val(), -1);
    ASSERT_EQ(max_idx(), 1);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Max)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    // example-begin max-test-1
    auto t0 = make_tensor<TestType>({});
    auto t1o = make_tensor<TestType>({11});

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});

    // Reduce all inputs in "t1o" into "t0" by the maximum of all elements
    (t0 = max(t1o)).run(exec);
    // example-end max-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(11)));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Min)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    // example-begin min-test-1
    auto t0 = make_tensor<TestType>({});
    auto t1o = make_tensor<TestType>({11});

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});

    // Reduce all inputs in "t1o" into "t0" by the minimum of all elements
    (t0 = min(t1o)).run(exec);
    // example-end min-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, ArgMax)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    // example-begin argmax-test-1
    auto t0 = make_tensor<TestType>({});
    auto t0i = make_tensor<index_t>({});
    auto t1o = make_tensor<TestType>({11});

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});

    (mtie(t0, t0i) = argmax(t1o)).run(exec);
    // example-end argmax-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(11)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0i(), (TestType)(10)));

    // Test with a non-tensor input
    (mtie(t0, t0i) = argmax(t1o+static_cast<TestType>(0))).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(11)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0i(), (TestType)(10)));

    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1o_small{{2}};
    tensor_t<index_t, 1> t1i_small{{2}};
    t2o.SetVals({{(T)2, (T)4, (T)1, (T)3, (T)5}, {(T)3, (T)1, (T)5, (T)2, (T)4}});

    (mtie(t1o_small, t1i_small) = argmax(t2o, {1})).run(exec);
    exec.sync();

    auto rel = GetIdxFromAbs(t2o, t1i_small(0));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(5)));
    rel = GetIdxFromAbs(t2o, t1i_small(1));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(5)));
  }

  {
    ExecType exec{};
    const int BATCHES = 6;
    const int ROWS = 33;
    const int COLUMNS = 33;
    const int BATCH_STRIDE = ROWS*COLUMNS;
    auto t_a = matx::make_tensor<TestType>({BATCHES,ROWS,COLUMNS});
    auto t_bi = matx::make_tensor<matx::index_t>({BATCHES});
    auto t_b = matx::make_tensor<TestType>({BATCHES});

    (t_a = static_cast<TestType>(0)).run(exec);
    exec.sync();

    matx::index_t expected_abs[6] {31*33+22, 32*33+24, 19*33+12, 21*33+17, 17*33+7, 1*33+24};
    for (int n=0; n<BATCHES; n++)
    {
      matx::index_t max_row = expected_abs[n] / COLUMNS;
      matx::index_t max_col = expected_abs[n] - max_row*COLUMNS;
      t_a(n,max_row,max_col) = static_cast<TestType>(1);

      expected_abs[n] += n*BATCH_STRIDE;
    }

    (matx::mtie(t_b, t_bi) = matx::argmax(t_a, {1,2})).run(exec);
    exec.sync();

    for (int n=0; n<BATCHES; n++)
    {
      EXPECT_TRUE(t_bi(n) == expected_abs[n]);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, ArgMin)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    // example-begin argmin-test-1
    auto t0 = make_tensor<TestType>({});
    auto t0i = make_tensor<index_t>({});
    auto t1o = make_tensor<TestType>({11});

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});

    (mtie(t0, t0i) = argmin(t1o)).run(exec);
    // example-end argmin-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0i(), (TestType)(0)));

    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1o_small{{2}};
    tensor_t<index_t, 1> t1i_small{{2}};
    t2o.SetVals({{(T)2, (T)4, (T)1, (T)3, (T)5}, {(T)3, (T)1, (T)5, (T)2, (T)4}});

    (mtie(t1o_small, t1i_small) = argmin(t2o, {1})).run(exec);
    exec.sync();

    auto rel = GetIdxFromAbs(t2o, t1i_small(0));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(1)));
    rel = GetIdxFromAbs(t2o, t1i_small(1));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(1)));
  }

  {
    ExecType exec{};
    const int BATCHES = 6;
    const int ROWS = 33;
    const int COLUMNS = 33;
    const int BATCH_STRIDE = ROWS*COLUMNS;
    auto t_a = matx::make_tensor<TestType>({BATCHES,ROWS,COLUMNS});
    auto t_bi = matx::make_tensor<matx::index_t>({BATCHES});
    auto t_b = matx::make_tensor<TestType>({BATCHES});

    (t_a = static_cast<TestType>(0)).run(exec);
    exec.sync();

    matx::index_t expected_abs[6] {31*33+22, 32*33+24, 19*33+12, 21*33+17, 17*33+7, 1*33+24};
    for (int n=0; n<BATCHES; n++)
    {
      matx::index_t max_row = expected_abs[n] / COLUMNS;
      matx::index_t max_col = expected_abs[n] - max_row*COLUMNS;
      t_a(n,max_row,max_col) = static_cast<TestType>(-1);

      expected_abs[n] += n*BATCH_STRIDE;
    }

    (matx::mtie(t_b, t_bi) = matx::argmin(t_a, {1,2})).run(exec);
    exec.sync();

    for (int n=0; n<BATCHES; n++)
    {
      EXPECT_TRUE(t_bi(n) == expected_abs[n]);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, ArgMinMax)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  {
    ExecType exec{};
    using T = TestType;
    // example-begin argminmax-test-1
    auto t0min = make_tensor<TestType>({});
    auto t0mini = make_tensor<index_t>({});
    auto t0max = make_tensor<TestType>({});
    auto t0maxi = make_tensor<index_t>({});
    auto t1o = make_tensor<TestType>({11});

    t1o.SetVals({(T)1, (T)3, (T)8, (T)2, (T)9, (T)10, (T)6, (T)7, (T)4, (T)5, (T)11});

    (mtie(t0min, t0mini, t0max, t0maxi) = argminmax(t1o)).run(exec);
    // example-end argminmax-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0min(), (TestType)(1)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0mini(), (TestType)(0)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0max(), (TestType)(11)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0maxi(), (TestType)(10)));

    tensor_t<TestType, 2> t2o{{2, 5}};
    tensor_t<TestType, 1> t1o_min_small{{2}};
    tensor_t<index_t, 1> t1i_min_small{{2}};
    tensor_t<TestType, 1> t1o_max_small{{2}};
    tensor_t<index_t, 1> t1i_max_small{{2}};
    t2o.SetVals({{(T)2, (T)4, (T)1, (T)3, (T)5}, {(T)3, (T)1, (T)5, (T)2, (T)4}});

    (mtie(t1o_min_small, t1i_min_small, t1o_max_small, t1i_max_small) = argminmax(t2o, {1})).run(exec);
    exec.sync();

    auto rel = GetIdxFromAbs(t2o, t1i_min_small(0));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(1)));
    rel = GetIdxFromAbs(t2o, t1i_min_small(1));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(1)));
    rel = GetIdxFromAbs(t2o, t1i_max_small(0));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(5)));
    rel = GetIdxFromAbs(t2o, t1i_max_small(1));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2o(rel), (TestType)(5)));
  }

  {
    ExecType exec{};
    const int BATCHES = 6;
    const int ROWS = 33;
    const int COLUMNS = 33;
    const int BATCH_STRIDE = ROWS*COLUMNS;
    auto t_a = matx::make_tensor<TestType>({BATCHES,ROWS,COLUMNS});
    auto t_bi = matx::make_tensor<matx::index_t>({BATCHES});
    auto t_b = matx::make_tensor<TestType>({BATCHES});
    auto t_ci = matx::make_tensor<matx::index_t>({BATCHES});
    auto t_c = matx::make_tensor<TestType>({BATCHES});

    (t_a = static_cast<TestType>(0)).run(exec);
    exec.sync();

    matx::index_t expected_max_abs[6] {31*33+22, 32*33+24, 19*33+12, 21*33+17, 17*33+7, 1*33+24};
    matx::index_t expected_min_abs[6] {1*33+2, 2*33+4, 4*33+6, 9*33+12, 11*33+7, 13*33+24};
    for (int n=0; n<BATCHES; n++)
    {
      matx::index_t min_row = expected_min_abs[n] / COLUMNS;
      matx::index_t min_col = expected_min_abs[n] - min_row*COLUMNS;
      t_a(n,min_row,min_col) = static_cast<TestType>(-1);

      matx::index_t max_row = expected_max_abs[n] / COLUMNS;
      matx::index_t max_col = expected_max_abs[n] - max_row*COLUMNS;
      t_a(n,max_row,max_col) = static_cast<TestType>(1);

      expected_min_abs[n] += n*BATCH_STRIDE;
      expected_max_abs[n] += n*BATCH_STRIDE;
    }

    // example-begin argminmax-test-2
    (matx::mtie(t_c, t_ci, t_b, t_bi) = matx::argminmax(t_a, {1,2})).run(exec);
    // example-end argminmax-test-2
    exec.sync();

    for (int n=0; n<BATCHES; n++)
    {
      EXPECT_TRUE(t_ci(n) == expected_min_abs[n]);
      EXPECT_TRUE(t_bi(n) == expected_max_abs[n]);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Mean)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  {
    auto t3 = ones<TestType>({30, 40, 50});
    auto t2 = ones<TestType>({30, 40});
    auto t1 = ones<TestType>({30});

    // example-begin mean-test-1
    auto t0 = make_tensor<TestType>({});
    auto t4 = ones<TestType>({30, 40, 50, 60});
    // Compute the mean over all dimensions in "t4" and store the result in "t0"
    (t0 = mean(t4)).run(exec);
    // example-end mean-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    (t0 = mean(t3)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    (t0 = mean(t2)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));

    (t0 = mean(t1)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TestType)(1)));
  }
  {
    tensor_t<TestType, 1> t1({30});

    auto t4 = ones<TestType>({30, 40, 50, 60});
    auto t3 = ones<TestType>({30, 40, 50});
    auto t2 = ones<TestType>({30, 40});

    (t1 = mean(t4, {1, 2, 3})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(1)));
    }

    (t1 = mean(t3, {1, 2})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(1)));
    }

    (t1 = mean(t2, {1})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)(1)));
    }
  }

  {
    tensor_t<TestType, 2> t2({30, 40});

    auto t4 = ones<TestType>({30, 40, 50, 60});
    auto t3 = ones<TestType>({30, 40, 50});

    (t2 = mean(t4, {2, 3})).run(exec);
    exec.sync();
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TestType)(1)));
      }
    }

    (t2 = mean(t3, {2})).run(exec);
    exec.sync();
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
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec;

  MATX_ENTER_HANDLER();
  {
    auto t0 = make_tensor<TestType>({});

    cuda::std::array<index_t, 2> s2{3, 4};
    cuda::std::array<index_t, 1> s1{3};
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

    // example-begin prod-test-1
    // Compute the product of all elements in "t2" and store into "t0"
    (t0 = prod(t2)).run(exec);
    // example-end prod-test-1
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), t2p));

    (t0 = prod(t1)).run(exec);
    exec.sync();
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), t1p));
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Find)
{
  MATX_ENTER_HANDLER();
  {
    using TestType = cuda::std::tuple_element_t<0, TypeParam>;
    using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

    ExecType exec{};

    tensor_t<int, 0> num_found{{}};
    tensor_t<TestType, 1> t1{{100}};
    tensor_t<TestType, 1> t1o{{100}};

    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TestType>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // example-begin find-test-1
    // Find values greater than 0.5
    TestType thresh = (TestType)0.5;
    (mtie(t1o, num_found) = find(t1, GT{thresh})).run(exec);
    // example-end find-test-1
    exec.sync();

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
    using TestType = cuda::std::tuple_element_t<0, TypeParam>;
    using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

    ExecType exec{};

    tensor_t<int, 0> num_found{{}};
    tensor_t<TestType, 1> t1{{100}};
    tensor_t<int, 1> t1o{{100}};

    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TestType>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // example-begin find_idx-test-1
    // Find indices with values greater than 0.5
    TestType thresh = (TestType)0.5;
    (mtie(t1o, num_found) = find_idx(t1, GT{thresh})).run(exec);
    // example-end find_idx-test-1
    exec.sync();

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
    using TestType = cuda::std::tuple_element_t<0, TypeParam>;
    using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

    tensor_t<int, 0> num_found{{}}, num_found2{{}};
    tensor_t<TestType, 1> t1{{100}};
    tensor_t<int, 1> t1o_idx{{100}};
    tensor_t<TestType, 1> t1o{{100}};
    tensor_t<TestType, 1> t1o_2{{100}};
    TestType thresh = (TestType)0.5;

    ExecType exec{};

    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<detail::value_promote_t<TestType>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
    }

    // Find indices with values greater than 0.5
    // example-begin select-test-1
    (mtie(t1o_idx, num_found) = find_idx(t1, GT{thresh})).run(exec);

    // Since we use the output on the host in select() we need to synchronize first
    exec.sync();

    auto t1o_slice = slice(t1o, {0}, {num_found()});
    auto t1o_idx_slice = slice(t1o_idx, {0}, {num_found()});
    (t1o_slice = select(t1o_slice, t1o_idx_slice)).run(exec);

    // Compare to simply finding the values
    (mtie(t1o_2, num_found2) = find(t1, GT{thresh})).run(exec);
    // example-end select-test-1
    exec.sync();

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
    using TestType = cuda::std::tuple_element_t<0, TypeParam>;
    using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

    ExecType exec{};

    tensor_t<int, 0> num_found{{}};
    tensor_t<TestType, 1> t1{{100}};
    tensor_t<TestType, 1> t1o{{100}};

    for (int i = 0; i < t1.Size(0); i++) {
      t1(i) = (TestType)(i % 10);
    }

    // example-begin unique-test-1
    (mtie(t1o, num_found) = unique(t1)).run(exec);
    // example-end unique-test-1
    exec.sync();

    for (int i = 0; i < 10; i++) {
      ASSERT_NEAR(t1o(i), i, 0.01);
    }

    ASSERT_EQ(10, num_found());

  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ReductionTestsFloatNonComplexNonHalfAllExecs, Trace)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec;

  MATX_ENTER_HANDLER();
  index_t count = 10;

  // example-begin trace-test-1
  auto t2 = make_tensor<TestType>({count, count});
  auto t0 = make_tensor<TestType>({});

  (t2 = ones<TestType>(t2.Shape())).run(exec);
  (t0 = trace(t2)).run(exec);
  // example-end trace-test-1

  exec.sync();

  ASSERT_EQ(t0(), count);
  MATX_EXIT_HANDLER();
}


