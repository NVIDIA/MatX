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


template <typename T> struct CUBTestsData {
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;   
  GExecType exec{};   

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
class CUBTestsComplex : public ::testing::Test,
                                public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsFloat : public ::testing::Test,
                              public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsFloatNonComplex
    : public ::testing::Test,
      public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsNumeric : public ::testing::Test,
                                public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsNumericNonComplex
    : public ::testing::Test,
      public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsIntegral : public ::testing::Test,
                                 public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsBoolean : public ::testing::Test,
                                public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsAll : public ::testing::Test,
                            public CUBTestsData<TensorType> {
};

template <typename TensorType>
class CUBTestsNumericNonComplexAllExecs : public ::testing::Test, public CUBTestsData<TensorType> {
};


TYPED_TEST_SUITE(CUBTestsAll, MatXAllTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsComplex, MatXComplexTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsFloat, MatXFloatTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsFloatNonComplex, MatXFloatNonComplexTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsNumeric, MatXNumericTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsIntegral, MatXAllIntegralTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsNumericNonComplex, MatXNumericNonComplexTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsBoolean, MatXBoolTypesCUDAExec);

TYPED_TEST_SUITE(CUBTestsNumericNonComplexAllExecs,
                 MatXFloatNonComplexNonHalfTypesAllExecs);  

TEST(TensorStats, Hist)
{
  MATX_ENTER_HANDLER(); 

  constexpr int levels = 7;
  tensor_t<float, 1> inv({10});
  tensor_t<int, 1> outv({levels - 1});

  inv.SetVals({2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5});

  cudaExecutor exec{};

  // example-begin hist-test-1
  (outv = hist(inv, 0.0f, 12.0f)).run(exec);
  // example-end hist-test-1
  exec.sync();

  cuda::std::array<int, levels - 1> sol = {1, 5, 0, 3, 0, 0};
  for (index_t i = 0; i < outv.Lsize(); i++) {
    ASSERT_NEAR(outv(i), sol[i], 0.001);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CUBTestsNumericNonComplexAllExecs, CumSum)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  for (index_t i = 0; i < this->t1.Lsize(); i++) {
    this->t1(i) = static_cast<TestType>((2 * (i % 2) - 1) * i);
  }

  tensor_t<TestType, 1> tmpv({this->t1.Lsize()});

  // Ascending
  // example-begin cumsum-test-1
  // Compute the cumulative sum/exclusive scan across "t1"
  (tmpv = cumsum(this->t1)).run(this->exec);
  // example-end cumsum-test-1
  this->exec.sync();

  TestType ttl = 0;
  for (index_t i = 0; i < tmpv.Lsize(); i++) {
    ttl += this->t1(i);
    ASSERT_NEAR(tmpv(i), ttl, 0.001);
  }

  // 2D tests
  auto tmpv2 = make_tensor<TestType>(this->t2.Shape());

  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i, j) = static_cast<TestType>((2 * (j % 2) - 1) * j + i);
    }
  }

  (tmpv2 = cumsum(this->t2)).run(this->exec);
  this->exec.sync();
  for (index_t i = 0; i < tmpv2.Size(0); i++) {
    ttl = 0;
    for (index_t j = 0; j < tmpv2.Size(1); j++) {
      ttl += this->t2(i, j);
      ASSERT_NEAR(tmpv2(i, j), ttl, 0.001) << i << j;
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CUBTestsNumericNonComplexAllExecs, Sort)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
 

  for (index_t i = 0; i < this->t1.Lsize(); i++) {
    this->t1(i) = static_cast<TestType>((2 * (i % 2) - 1) * i);
  }

  auto tmpv = make_tensor<TestType>({this->t1.Lsize()});

  // example-begin sort-test-1
  // Ascending sort of 1D input
  (tmpv = matx::sort(this->t1, SORT_DIR_ASC)).run(this->exec);
  // example-end sort-test-1
  this->exec.sync();

  for (index_t i = 1; i < tmpv.Lsize(); i++) {
    ASSERT_TRUE(tmpv(i) > tmpv(i - 1));
  }

  // example-begin sort-test-2
  // Descending sort of 1D input
  (tmpv = matx::sort(this->t1, SORT_DIR_DESC)).run(this->exec);
  // example-end sort-test-2
  this->exec.sync();

  for (index_t i = 1; i < tmpv.Lsize(); i++) {
    ASSERT_TRUE(tmpv(i) < tmpv(i - 1));
  }

  // 2D tests
  auto tmpv2 = make_tensor<TestType>(this->t2.Shape());

  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i, j) = static_cast<TestType>((2 * (j % 2) - 1) * j + i);
    }
  }

  (tmpv2 = matx::sort(this->t2, SORT_DIR_ASC)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < tmpv2.Size(0); i++) {
    for (index_t j = 1; j < tmpv2.Size(1); j++) {
      ASSERT_TRUE(tmpv2(i, j) > tmpv2(i, j - 1));
    }
  }

  // Descending
  (tmpv2 = matx::sort(this->t2, SORT_DIR_DESC)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < tmpv2.Size(0); i++) {
    for (index_t j = 1; j < tmpv2.Size(1); j++) {
      ASSERT_TRUE(tmpv2(i, j) < tmpv2(i, j - 1));
    }
  }

  MATX_EXIT_HANDLER();
}
