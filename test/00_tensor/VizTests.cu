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
#if MATX_ENABLE_VIZ


#include "assert.h"
#include "matx.h"
#include "matx_viz.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;

template <typename TensorType> struct VizTestsData {
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
class VizTestsComplex : public ::testing::Test,
                        public VizTestsData<TensorType> {
};
template <typename TensorType>
class VizTestsFloat : public ::testing::Test, public VizTestsData<TensorType> {
};
template <typename TensorType>
class VizTestsFloatNonComplex : public ::testing::Test,
                                public VizTestsData<TensorType> {
};
template <typename TensorType>
class VizTestsNumeric : public ::testing::Test,
                        public VizTestsData<TensorType> {
};
template <typename TensorType>
class VizTestsNumericNonComplex : public ::testing::Test,
                                  public VizTestsData<TensorType> {
};
template <typename TensorType>
class VizTestsIntegral : public ::testing::Test,
                         public VizTestsData<TensorType> {
};
template <typename TensorType>
class VizTestsBoolean : public ::testing::Test,
                        public VizTestsData<TensorType> {
};
template <typename TensorType>
class VizTestsAll : public ::testing::Test, public VizTestsData<TensorType> {
};

TYPED_TEST_SUITE(VizTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(VizTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(VizTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(VizTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(VizTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(VizTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(VizTestsNumericNonComplex, MatXNumericNonComplexTypes);
TYPED_TEST_SUITE(VizTestsBoolean, MatXBoolTypes);

TYPED_TEST(VizTestsNumericNonComplex, Line)
{
  MATX_ENTER_HANDLER();

  this->t1.SetVals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  viz::line(this->t1, "My Line Plot", "Y label", "X label", "line.html");

  MATX_EXIT_HANDLER();
}

TYPED_TEST(VizTestsNumericNonComplex, Scatter)
{
  MATX_ENTER_HANDLER();

  this->t1.SetVals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  tensor_t<TypeParam, 1> t1y({10});

  t1y.SetVals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  viz::scatter(this->t1, t1y, "My Scatter Plot", "Y label", "X label",
               "scatter.html");

  MATX_EXIT_HANDLER();
}

TYPED_TEST(VizTestsNumericNonComplex, Bar)
{
  MATX_ENTER_HANDLER();

  this->t1.SetVals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  tensor_t<TypeParam, 1> t1y({10});

  t1y.SetVals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  viz::bar(this->t1, "My Bar Plot", "X label", "bar.html");
  viz::bar(this->t1, t1y, "My Bar Plot", "X label", "Y label", "bar2.html");

  MATX_EXIT_HANDLER();
}
#endif