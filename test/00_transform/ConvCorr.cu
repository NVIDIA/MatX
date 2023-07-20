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
  (this->cv = conv1d(this->av, this->bv, MATX_C_MODE_FULL)).run();
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
  (this->cv_full_even = conv1d(this->av, this->bv_even, MATX_C_MODE_FULL)).run();

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
  (this->cv_full_even = conv2d(this->av, this->bv_even, MATX_C_MODE_FULL)).run();

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}



TYPED_TEST(CorrelationConvolutionTestFloatTypes, Direct1DConvolutionSameEven)
{
  MATX_ENTER_HANDLER();
  this->pb->template InitTVGenerator<TypeParam>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_same = conv1d(this->av, this->bv_even, MATX_C_MODE_SAME)).run();

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
  (this->cv_same = conv2d(this->av, this->bv_even, MATX_C_MODE_SAME)).run();
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
  (this->cv_valid_even = conv1d(this->av, this->bv_even, MATX_C_MODE_VALID)).run();

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
  (this->cv_valid_even = conv2d(this->av, this->bv_even, MATX_C_MODE_VALID)).run();

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
  (this->cv_full_odd = conv1d(this->av, this->bv_odd, MATX_C_MODE_FULL)).run();

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
  (this->cv_full_odd = conv2d(this->av, this->bv_odd, MATX_C_MODE_FULL)).run();

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
  (this->cv_same = conv1d(this->av, this->bv_odd, MATX_C_MODE_SAME)).run();

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
  (this->cv_same = conv2d(this->av, this->bv_odd, MATX_C_MODE_SAME)).run();

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
  (this->cv_valid_odd = conv1d(this->av, this->bv_odd, MATX_C_MODE_VALID)).run();

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
  (this->cv_valid_odd = conv2d(this->av, this->bv_odd, MATX_C_MODE_VALID)).run();

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
  (this->cv_full_even = conv1d(this->bv_even, this->av, MATX_C_MODE_FULL)).run();

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
  (this->cv_full_even = conv2d(this->bv_even, this->av, MATX_C_MODE_FULL)).run();

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
  (this->cv_full_even = corr(this->av, this->bv_even, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT)).run();
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
  (this->cv_full_even = corr(this->bv_even, this->av, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT)).run();

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

  (out1 = conv1d(in1, in2, MATX_C_MODE_SAME)).run();
  // example-begin conv1d-test-1
  (out2 = conv1d(in1, in2, {2}, MATX_C_MODE_SAME)).run();
  // example-end conv1d-test-1

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1.Permute({0,2,1}) = conv1d(in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME)).run();
  // example-begin conv1d-test-3
  (out2 = conv1d(in1, in2, {1}, MATX_C_MODE_SAME)).run();
  // example-end conv1d-test-3

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1.Permute({1,2,0}) = conv1d(in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME)).run();
  (out2 = conv1d(in1, in2, {0}, MATX_C_MODE_SAME)).run();

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1 = corr(in1, in2, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run();
  (out2 = corr(in1, in2, {2}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run();

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1.Permute({0,2,1}) = corr(in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run();
  (out2 = corr(in1, in2, {1}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run();

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1.Permute({1,2,0}) = corr(in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run();
  (out2 = corr(in1, in2, {0}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run();

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

  (out1 = conv2d(in1, in2, MATX_C_MODE_SAME)).run();
  (out2 = conv2d(in1, in2, {1, 2}, MATX_C_MODE_SAME)).run();

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }
 
  (out1.Permute({0,2,1}) = conv2d(in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME)).run();
  (out2 = conv2d(in1, in2, {2, 1}, MATX_C_MODE_SAME)).run();

  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }
  
  (out1.Permute({1,2,0}) = conv2d(in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME)).run();
  (out2 = conv2d(in1, in2, {2, 0}, MATX_C_MODE_SAME)).run();

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
