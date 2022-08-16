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

constexpr index_t a_len = 256;
constexpr index_t b_len_even = 16;
constexpr index_t b_len_odd = 15;
constexpr index_t c_len_full_even = a_len + b_len_even - 1;
constexpr index_t c_len_full_odd = a_len + b_len_odd - 1;
constexpr index_t c_len_valid_even = a_len - b_len_even + 1;
constexpr index_t c_len_valid_odd = a_len - b_len_odd + 1;
constexpr index_t c_len_same = a_len;

template <typename T>
class CorrelationConvolutionTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    CheckTestTypeSupport<T>();
    pb = std::make_unique<detail::MatXPybind>();

    // Half precision needs a bit more tolerance when compared to
    // fp32
    if constexpr (is_complex_half_v<T> || is_matx_half_v<T>) {
      thresh = 0.2f;
    }
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<T, 1> av{{a_len}};
  tensor_t<T, 1> bv_even{{b_len_even}};
  tensor_t<T, 1> bv_odd{{b_len_odd}};
  tensor_t<T, 1> cv_full_even{{c_len_full_even}};
  tensor_t<T, 1> cv_full_odd{{c_len_full_odd}};  
  tensor_t<T, 1> cv_valid_even{{c_len_valid_even}};
  tensor_t<T, 1> cv_valid_odd{{c_len_valid_odd}};
  tensor_t<T, 1> cv_same{{c_len_same}};
  float thresh = 0.01f;
};

template <typename TensorType>
class CorrelationConvolutionTestFloatTypes
    : public CorrelationConvolutionTest<TensorType> {
};

TYPED_TEST_SUITE(CorrelationConvolutionTestFloatTypes, MatXFloatTypes);

// Real/real direct 1D convolution
TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionFullEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv1d(this->cv_full_even, this->av, this->bv_even, MATX_C_MODE_FULL, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionRemap)
{
  MATX_ENTER_HANDLER();

  int N = 256;
  int B = 5;
  int F = 3;
  int R = 3;

  // This is a does it compile and run test
  // TODO add correctness checking
  auto in = make_tensor<int>({B, N});
  auto out = make_tensor<int>({B, N});
  auto filt = make_tensor<int>({B, F});
  auto idx = make_tensor<int>({R});
  for(int i = 0; i < idx.Size(0); i++) {
    idx(i) = i;
  }

  (in = 1).run();
  (filt = 1).run();

  conv1d(out, in, filt, MATX_C_MODE_SAME, 0);
  
  conv1d(remap<0>(out,idx), remap<0>(in,idx), filt, MATX_C_MODE_SAME, 0);

  cudaDeviceSynchronize();
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionSameEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv1d(this->cv_same, this->av, this->bv_even, MATX_C_MODE_SAME, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionValidEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv1d(this->cv_valid_even, this->av, this->bv_even, MATX_C_MODE_VALID, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_even, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionFullOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_odd});  
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv1d(this->cv_full_odd, this->av, this->bv_odd, MATX_C_MODE_FULL, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_odd, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionSameOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_odd});   
  this->pb->RunTVGenerator("conv");   
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv1d(this->cv_same, this->av, this->bv_odd, MATX_C_MODE_SAME, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionValidOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_odd});   
  this->pb->RunTVGenerator("conv");   
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv1d(this->cv_valid_odd, this->av, this->bv_odd, MATX_C_MODE_VALID, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_odd, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionSwap)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv1d(this->cv_full_even, this->bv_even, this->av, MATX_C_MODE_FULL, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DCorrelation)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_even});  
  this->pb->RunTVGenerator("corr");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  corr(this->cv_full_even, this->av, this->bv_even, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "corr", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DCorrelationSwap)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len_even});  
  this->pb->RunTVGenerator("corr_swap");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  corr(this->cv_full_even, this->bv_even, this->av, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "corr_swap", this->thresh);
  MATX_EXIT_HANDLER();
}

// // Complex/complex direct 1D convolution
// TEST_F(CorrelationConvolutionTest, Direct1DC2CConvolution)
// {
//   MATX_ENTER_HANDLER();
//   conv1d(ccv, acv, bcv, MATX_C_MODE_FULL, 0);
//   MATX_TEST_ASSERT_COMPARE(pb, ccv, "c_op_complex_conv", 0.01);
//   MATX_EXIT_HANDLER();
// }

// // Real/real direct 1D convolution with swapped parameters
// TEST_F(CorrelationConvolutionTest, Direct1DR2RConvolutionSwap)
// {
//   MATX_ENTER_HANDLER();
//   conv1d(crv, brv, arv, MATX_C_MODE_FULL, 0);
//   MATX_TEST_ASSERT_COMPARE(pb, crv, "c_op_real_conv", 0.01);
//   MATX_EXIT_HANDLER();
// }

// // Complex/complex direct 1D convolution with swapped parameters
// TEST_F(CorrelationConvolutionTest, Direct1DC2CConvolutionSwap)
// {
//   MATX_ENTER_HANDLER();
//   conv1d(ccv, bcv, acv, MATX_C_MODE_FULL, 0);
//   MATX_TEST_ASSERT_COMPARE(pb, ccv, "c_op_complex_conv", 0.01);
//   MATX_EXIT_HANDLER();
// }

// // Real/real direct 1D correlation
// TEST_F(CorrelationConvolutionTest, Direct1DR2RCorrelation)
// {
//   MATX_ENTER_HANDLER();
//   corr(crv, arv, brv, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT, 0);
//   MATX_TEST_ASSERT_COMPARE(pb, crv, "c_op_real_corr", 0.01);
//   MATX_EXIT_HANDLER();
// }

// // Complex/complex direct 1D correlation
// TEST_F(CorrelationConvolutionTest, Direct1DC2CCorrelation)
// {
//   MATX_ENTER_HANDLER();
//   corr(ccv, acv, bcv, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT, 0);
//   MATX_TEST_ASSERT_COMPARE(pb, ccv, "c_op_complex_corr", 0.01);
//   MATX_EXIT_HANDLER();
// }

// // Real/real direct 1D correlation with swapped parameters
// TEST_F(CorrelationConvolutionTest, Direct1DR2RCorrelationSwap)
// {
//   MATX_ENTER_HANDLER();
//   corr(crv, brv, arv, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT, 0);
//   MATX_TEST_ASSERT_COMPARE(pb, crv, "c_op_real_corr_swap", 0.01);
//   MATX_EXIT_HANDLER();
// }

// // Complex/complex direct 1D correlation with swapped parameters
// TEST_F(CorrelationConvolutionTest, Direct1DC2CCorrelationSwap)
// {
//   MATX_ENTER_HANDLER();
//   corr(ccv, bcv, acv, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT, 0);
//   MATX_TEST_ASSERT_COMPARE(pb, ccv, "c_op_complex_corr_swap", 0.01);
//   MATX_EXIT_HANDLER();
// }
