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

constexpr index_t a_len = 16;

template <typename T>
class NormTest : public ::testing::Test {
  using GTestType = std::tuple_element_t<0, T>;
  using GExecType = std::tuple_element_t<1, T>;
protected:
  void SetUp() override
  {
    CheckTestTypeSupport<GTestType>();
    pb = std::make_unique<detail::MatXPybind>();

    // Half precision needs a bit more tolerance when compared to fp32
    if constexpr (is_complex_half_v<GTestType> || is_matx_half_v<GTestType>) {
      thresh = 0.5f;
    }
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};   
  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<GTestType, 1> in_v{{a_len}};
  tensor_t<GTestType, 2> in_m{{a_len, a_len}};
  tensor_t<GTestType, 0> out_v{{}};
  tensor_t<GTestType, 0> out_m{{}};
  float thresh = 0.01f;
};


template <typename TensorType>
class NormTestFloatTypes
    : public NormTest<TensorType> {
};



TYPED_TEST_SUITE(NormTestFloatTypes, MatXTypesFloatNonComplexAllExecs);


TYPED_TEST(NormTestFloatTypes, VectorL1)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "norm_operators", {a_len});
  this->pb->RunTVGenerator("vector_l1");
  this->pb->NumpyToTensorView(this->in_v, "in_v");
  this->pb->NumpyToTensorView(this->out_v, "out_v");

  auto redOp = vector_norm(this->in_v, NormOrder::L1);

  EXPECT_TRUE(redOp.Rank() == this->out_v.Rank());
  for(int i = 0; i < redOp.Rank(); i++) {
    EXPECT_TRUE(redOp.Size(i) == this->out_v.Size(i));
  }
  (this->out_v = redOp).run(this->exec);
  MATX_TEST_ASSERT_COMPARE(this->pb, this->out_v, "out_v", this->thresh);

  (this->out_v = TestType(0)).run(this->exec);

  // example-begin vector-norm-test-1
  (this->out_v = vector_norm(this->in_v, NormOrder::L1)).run(this->exec);
  // example-end vector-norm-test-1
  
  MATX_TEST_ASSERT_COMPARE(this->pb, this->out_v, "out_v", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(NormTestFloatTypes, VectorL2)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "norm_operators", {a_len});
  this->pb->RunTVGenerator("vector_l2");
  this->pb->NumpyToTensorView(this->in_v, "in_v");
  this->pb->NumpyToTensorView(this->out_v, "out_v");
  
  auto redOp = vector_norm(this->in_v, NormOrder::L2);
  
  EXPECT_TRUE(redOp.Rank() == this->out_v.Rank());
  for(int i = 0; i < redOp.Rank(); i++) {
    EXPECT_TRUE(redOp.Size(i) == this->out_v.Size(i));
  }
  (this->out_v = redOp).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->out_v, "out_v", this->thresh);
  
  (this->out_v = TestType(0)).run(this->exec);

  // example-begin vector-norm-test-2
  (this->out_v = vector_norm(this->in_v, NormOrder::L2)).run(this->exec);
  // example-end vector-norm-test-2

  MATX_TEST_ASSERT_COMPARE(this->pb, this->out_v, "out_v", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(NormTestFloatTypes, MatrixL1)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "norm_operators", {a_len, a_len});
  this->pb->RunTVGenerator("matrix_l1");
  this->pb->NumpyToTensorView(this->in_m, "in_m");
  this->pb->NumpyToTensorView(this->out_m, "out_m");
  
  auto redOp = matrix_norm(this->in_m, NormOrder::L1);
  
  EXPECT_TRUE(redOp.Rank() == this->out_v.Rank());
  for(int i = 0; i < redOp.Rank(); i++) {
    EXPECT_TRUE(redOp.Size(i) == this->out_v.Size(i));
  }
  
  (this->out_m = redOp).run(this->exec);
  
  MATX_TEST_ASSERT_COMPARE(this->pb, this->out_m, "out_m", this->thresh);
  
  (this->out_v = TestType(0)).run(this->exec);
  
  // example-begin matrix-norm-test-1
  (this->out_m = matrix_norm(this->in_m, NormOrder::L1)).run(this->exec);
  // example-end matrix-norm-test-1

  MATX_TEST_ASSERT_COMPARE(this->pb, this->out_m, "out_m", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(NormTestFloatTypes, MatrixL2)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "norm_operators", {a_len, a_len});
  this->pb->RunTVGenerator("matrix_frob");
  this->pb->NumpyToTensorView(this->in_m, "in_m");
  this->pb->NumpyToTensorView(this->out_m, "out_m");
  
  auto redOp = matrix_norm(this->in_m, NormOrder::FROB);
  
  EXPECT_TRUE(redOp.Rank() == this->out_v.Rank());
  for(int i = 0; i < redOp.Rank(); i++) {
    EXPECT_TRUE(redOp.Size(i) == this->out_v.Size(i));
  }
  (this->out_m = redOp).run(this->exec);
  
  MATX_TEST_ASSERT_COMPARE(this->pb, this->out_m, "out_m", this->thresh);
  
  (this->out_v = TestType(0)).run(this->exec);
  
  // example-begin matrix-norm-test-2
  (this->out_m = matrix_norm(this->in_m, NormOrder::FROB)).run(this->exec);
  // example-end matrix-norm-test-2

  MATX_TEST_ASSERT_COMPARE(this->pb, this->out_m, "out_m", this->thresh);

  MATX_EXIT_HANDLER();
}

