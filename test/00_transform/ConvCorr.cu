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

constexpr index_t a_len0 = 256;
constexpr index_t b_len0_even = 16;
constexpr index_t b_len0_odd = 15;
constexpr index_t c_len0_full_even = a_len0 + b_len0_even - 1;
constexpr index_t c_len0_full_odd = a_len0 + b_len0_odd - 1;
constexpr index_t c_len0_valid_even = a_len0 - b_len0_even + 1;
constexpr index_t c_len0_valid_odd = a_len0 - b_len0_odd + 1;
constexpr index_t c_len0_same = a_len0;

constexpr index_t a_len1 = 128;
constexpr index_t b_len1_even = 8;
constexpr index_t b_len1_odd = 7;
constexpr index_t c_len1_full_even = a_len1 + b_len1_even - 1;
constexpr index_t c_len1_full_odd = a_len1 + b_len1_odd - 1;
constexpr index_t c_len1_valid_even = a_len1 - b_len1_even + 1;
constexpr index_t c_len1_valid_odd = a_len1 - b_len1_odd + 1;
constexpr index_t c_len1_same = a_len1;

constexpr index_t a_len = 8 * 1228800 + 2 * 32768;
constexpr index_t b_len = 209;
constexpr index_t c_len = a_len + b_len - 1;

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
  tensor_t<T, 1> av{{a_len0}};
  tensor_t<T, 1> bv_even{{b_len0_even}};
  tensor_t<T, 1> bv_odd{{b_len0_odd}};
  tensor_t<T, 1> cv_full_even{{c_len0_full_even}};
  tensor_t<T, 1> cv_full_odd{{c_len0_full_odd}};  
  tensor_t<T, 1> cv_valid_even{{c_len0_valid_even}};
  tensor_t<T, 1> cv_valid_odd{{c_len0_valid_odd}};
  tensor_t<T, 1> cv_same{{c_len0_same}};
  float thresh = 0.01f;
};

template <typename T>
class CorrelationConvolution2DTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    CheckTestTypeSupport<T>();
    pb = std::make_unique<detail::MatXPybind>();

    // Half precision needs a bit more tolerance when compared to
    // fp32
    if constexpr (is_complex_half_v<T> || is_matx_half_v<T>) {
      thresh = .2f;
    }
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<T, 2> av{{a_len0,a_len1}};
  tensor_t<T, 2> bv_even{{b_len0_even,b_len1_even}};
  tensor_t<T, 2> bv_odd{{b_len0_odd,b_len1_odd}};
  tensor_t<T, 2> cv_full_even{{c_len0_full_even,c_len1_full_even}};
  tensor_t<T, 2> cv_full_odd{{c_len0_full_odd,c_len1_full_odd}};  
  tensor_t<T, 2> cv_valid_even{{c_len0_valid_even,c_len1_valid_even}};
  tensor_t<T, 2> cv_valid_odd{{c_len0_valid_odd,c_len1_valid_odd}};
  tensor_t<T, 2> cv_same{{c_len0_same,c_len1_same}};
  float thresh = 0.01f;
};

template <typename T>
class CorrelationConvolutionLargeTest : public ::testing::Test {
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
  tensor_t<T, 1> bv{{b_len}};
  tensor_t<T, 1> cv{{c_len}};
  float thresh = 0.01f;
};

template <typename TensorType>
class CorrelationConvolutionTestFloatTypes
    : public CorrelationConvolutionTest<TensorType> {
};

template <typename TensorType>
class CorrelationConvolutionLargeTestFloatTypes
    : public CorrelationConvolutionLargeTest<TensorType> {
};

template <typename TensorType>
class CorrelationConvolution2DTestFloatTypes
    : public CorrelationConvolution2DTest<TensorType> {
};

TYPED_TEST_SUITE(CorrelationConvolutionTestFloatTypes, MatXFloatTypes);
TYPED_TEST_SUITE(CorrelationConvolutionLargeTestFloatTypes, MatXFloatNonHalfTypes);
TYPED_TEST_SUITE(CorrelationConvolution2DTestFloatTypes, MatXFloatNonHalfTypes);

// Real/real direct 1D convolution Large
TYPED_TEST(CorrelationConvolutionLargeTestFloatTypes, Direct1DConvolutionLarge)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len, b_len});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv, "b_op");
  // example-begin conv1d-test-1
  // 1D convolution in FULL mode where every output is stored
  conv1d(this->cv, this->av, this->bv, MATX_C_MODE_FULL, 0);
  // example-end conv1d-test-1

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

// Real/real direct 1D convolution
TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionFullEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv1d(this->cv_full_even, this->av, this->bv_even, MATX_C_MODE_FULL, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

// Real/real direct 2D convolution
TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionFullEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_even, b_len1_even});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv2d(this->cv_full_even, this->av, this->bv_even, MATX_C_MODE_FULL, 0);

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

  // example-begin conv1d-test-2
  // 1D convolution in SAME mode where the output size matches the input
  conv1d(out, in, filt, MATX_C_MODE_SAME, 0);
  // example-end conv1d-test-2
  
  conv1d(remap<0>(out,idx), remap<0>(in,idx), filt, MATX_C_MODE_SAME, 0);

  cudaDeviceSynchronize();
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionSameEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv1d(this->cv_same, this->av, this->bv_even, MATX_C_MODE_SAME, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionSameEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_even, b_len1_even});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  // example-begin conv2d-test-1
  conv2d(this->cv_same, this->av, this->bv_even, MATX_C_MODE_SAME, 0);
  // example-end conv2d-test-1
  
  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionValidEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv1d(this->cv_valid_even, this->av, this->bv_even, MATX_C_MODE_VALID, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_even, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionValidEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_even, b_len1_even});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv2d(this->cv_valid_even, this->av, this->bv_even, MATX_C_MODE_VALID, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_even, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionFullOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_odd});  
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv1d(this->cv_full_odd, this->av, this->bv_odd, MATX_C_MODE_FULL, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_odd, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionFullOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_odd, b_len1_odd});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv2d(this->cv_full_odd, this->av, this->bv_odd, MATX_C_MODE_FULL, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_odd, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionSameOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_odd});   
  this->pb->RunTVGenerator("conv");   
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv1d(this->cv_same, this->av, this->bv_odd, MATX_C_MODE_SAME, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionSameOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_odd, b_len1_odd});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv2d(this->cv_same, this->av, this->bv_odd, MATX_C_MODE_SAME, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionValidOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_odd});   
  this->pb->RunTVGenerator("conv");   
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv1d(this->cv_valid_odd, this->av, this->bv_odd, MATX_C_MODE_VALID, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_odd, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionValidOdd)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_odd, b_len1_odd});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  conv2d(this->cv_valid_odd, this->av, this->bv_odd, MATX_C_MODE_VALID, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_odd, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionSwap)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv1d(this->cv_full_even, this->bv_even, this->av, MATX_C_MODE_FULL, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionSwap)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_even, b_len1_even});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  conv2d(this->cv_full_even, this->bv_even, this->av, MATX_C_MODE_FULL, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DCorrelation)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_even});  
  this->pb->RunTVGenerator("corr");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  // example-begin corr-test-1
  // Full correlation mode with direct correlation
  corr(this->cv_full_even, this->av, this->bv_even, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT, 0);
  // example-end corr-test-1

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "corr", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DCorrelationSwap)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_even});  
  this->pb->RunTVGenerator("corr_swap");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  corr(this->cv_full_even, this->bv_even, this->av, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "corr_swap", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Conv1Axis)
{
  MATX_ENTER_HANDLER();
  const int d1 = 8;
  const int d2 = 512;
  const int d3 = 1024;

  auto in1 = make_tensor<TypeParam>({d1, d2, d3});
  auto in2 = make_tensor<TypeParam>({d1, d2, d3});
  auto out1 = make_tensor<TypeParam>({d1, d2, d3});
  auto out2 = make_tensor<TypeParam>({d1, d2, d3});

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        in1(i,j,k) = static_cast<TypeParam>((float)(i+j+k));
        in2(i,j,k) = static_cast<TypeParam>((float)(1));
      }
    }
  }

  conv1d(out1, in1, in2, MATX_C_MODE_SAME);
  // example-begin conv1d-test-1
  conv1d(out2, in1, in2, {2}, MATX_C_MODE_SAME);
  // example-end conv1d-test-1

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  conv1d(out1.Permute({0,2,1}), in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME);
  // example-begin conv1d-test-3
  conv1d(out2, in1, in2, {1}, MATX_C_MODE_SAME);
  // example-end conv1d-test-3

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  conv1d(out1.Permute({1,2,0}), in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME);
  conv1d(out2, in1, in2, {0}, MATX_C_MODE_SAME);

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  corr(out1, in1, in2, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT);
  corr(out2, in1, in2, {2}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT);

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  corr(out1.Permute({0,2,1}), in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT);
  corr(out2, in1, in2, {1}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT);

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  corr(out1.Permute({1,2,0}), in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT);
  corr(out2, in1, in2, {0}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT);

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionTestFloatTypes, Conv2Axis)
{
  MATX_ENTER_HANDLER();
#if 1  // currently doesn't work because Conv2D requires rank2 filter.
  const int d1 = 8;
  const int d2 = 512;
  const int d3 = 1024;

  auto in1 = make_tensor<TypeParam>({d1, d2, d3});
  auto in2 = make_tensor<TypeParam>({d1, d2, d3});
  auto out1 = make_tensor<TypeParam>({d1, d2, d3});
  auto out2 = make_tensor<TypeParam>({d1, d2, d3});

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
			for(int k = 0; k < d3; k++) {
				in1(i,j,k) = static_cast<TypeParam>((float)(i+j+k));
				in2(i,j,k) = static_cast<TypeParam>((float)(1));
      }
    }
  }

  conv2d(out1, in1, in2, MATX_C_MODE_SAME);
  conv2d(out2, in1, in2, {1, 2}, MATX_C_MODE_SAME);

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }
 
  conv2d(out1.Permute({0,2,1}), in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME);
  conv2d(out2, in1, in2, {2, 1}, MATX_C_MODE_SAME);

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }
  
  conv2d(out1.Permute({1,2,0}), in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME);
  conv2d(out2, in1, in2, {2, 0}, MATX_C_MODE_SAME);

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }
#endif

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
