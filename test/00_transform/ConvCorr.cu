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

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000

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

constexpr index_t a_len = 8 * 122880 + 2 * 32768;
constexpr index_t b_len = 209;
constexpr index_t c_len = a_len + b_len - 1;

template <typename T, matxConvCorrMethod_t METHOD = MATX_C_METHOD_DIRECT>
class CorrelationConvolutionTest : public ::testing::Test {
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;
protected:
  void SetUp() override
  {
    CheckTestTypeSupport<GTestType>();
    CheckExecSupport();
    pb = std::make_unique<detail::MatXPybind>();

    // Half precision needs a bit more tolerance when compared to
    // fp32
    if constexpr (is_complex_half_v<GTestType> || is_matx_half_v<GTestType>) {
      thresh = 0.2f;
    }
  }

  void TearDown() override { pb.reset(); }

  constexpr void CheckExecSupport() {
    if constexpr (METHOD == MATX_C_METHOD_FFT) {
      if constexpr (!detail::CheckFFT1DConvSupport<GExecType, GTestType>()) {
        GTEST_SKIP();
      }
    } else {
      if constexpr (!detail::CheckDirect1DConvSupport<GExecType>()) {
        GTEST_SKIP();
      }
    }
  }

  GExecType exec{};   
  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<GTestType, 1> av{{a_len0}};
  tensor_t<GTestType, 1> bv_even{{b_len0_even}};
  tensor_t<GTestType, 1> bv_odd{{b_len0_odd}};
  tensor_t<GTestType, 1> cv_full_even{{c_len0_full_even}};
  tensor_t<GTestType, 1> cv_full_odd{{c_len0_full_odd}};  
  tensor_t<GTestType, 1> cv_valid_even{{c_len0_valid_even}};
  tensor_t<GTestType, 1> cv_valid_odd{{c_len0_valid_odd}};
  tensor_t<GTestType, 1> cv_same{{c_len0_same}};
  float thresh = 0.01f;
};

template <typename T>
class CorrelationConvolution2DTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;

  void SetUp() override
  {
    CheckTestTypeSupport<GTestType>();

    if constexpr (!detail::Check2DConvSupport<GExecType>()) {
      GTEST_SKIP();
    }

    pb = std::make_unique<detail::MatXPybind>();

    // Half precision needs a bit more tolerance when compared to
    // fp32
    if constexpr (is_complex_half_v<GTestType> || is_matx_half_v<GTestType>) {
      thresh = .2f;
    }
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};   
  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<GTestType, 2> av{{a_len0,a_len1}};
  tensor_t<GTestType, 2> bv_even{{b_len0_even,b_len1_even}};
  tensor_t<GTestType, 2> bv_odd{{b_len0_odd,b_len1_odd}};
  tensor_t<GTestType, 2> cv_full_even{{c_len0_full_even,c_len1_full_even}};
  tensor_t<GTestType, 2> cv_full_odd{{c_len0_full_odd,c_len1_full_odd}};  
  tensor_t<GTestType, 2> cv_valid_even{{c_len0_valid_even,c_len1_valid_even}};
  tensor_t<GTestType, 2> cv_valid_odd{{c_len0_valid_odd,c_len1_valid_odd}};
  tensor_t<GTestType, 2> cv_same{{c_len0_same,c_len1_same}};
  float thresh = 0.01f;
};

template <typename T, matxConvCorrMethod_t METHOD = MATX_C_METHOD_DIRECT>
class CorrelationConvolutionLargeTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;

  void SetUp() override
  {
    CheckTestTypeSupport<GTestType>();
    CheckExecSupport();
    pb = std::make_unique<detail::MatXPybind>();

    // Half precision needs a bit more tolerance when compared to
    // fp32
    if constexpr (is_complex_half_v<GTestType> || is_matx_half_v<GTestType>) {
      thresh = 0.2f;
    }
  }

  void TearDown() override { pb.reset(); }

  constexpr void CheckExecSupport() {
    if constexpr (METHOD == MATX_C_METHOD_FFT) {
      if constexpr (!detail::CheckFFT1DConvSupport<GExecType, GTestType>()) {
        GTEST_SKIP();
      }
    } else {
      if constexpr (!detail::CheckDirect1DConvSupport<GExecType>()) {
        GTEST_SKIP();
      }
    }
  }

  GExecType exec{};   
  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<GTestType, 1> av{{a_len}};
  tensor_t<GTestType, 1> bv{{b_len}};
  tensor_t<GTestType, 1> cv{{c_len}};
  float thresh = 0.01f;
};

template <typename TensorType>
class CorrelationConvolutionFFTTestFloatTypes
    : public CorrelationConvolutionTest<TensorType, MATX_C_METHOD_FFT> {
};

template <typename TensorType>
class CorrelationConvolutionDirectTestFloatTypes
    : public CorrelationConvolutionTest<TensorType, MATX_C_METHOD_DIRECT> {
};

template <typename TensorType>
class CorrelationConvolutionFFTTestNonHalfFloatTypes
    : public CorrelationConvolutionTest<TensorType, MATX_C_METHOD_FFT> {
};

template <typename TensorType>
class CorrelationConvolutionDirectTestNonHalfFloatTypes
    : public CorrelationConvolutionTest<TensorType, MATX_C_METHOD_DIRECT> {
};

template <typename TensorType>
class CorrelationConvolutionLargeFFTTestFloatTypes
    : public CorrelationConvolutionLargeTest<TensorType, MATX_C_METHOD_FFT> {
};

template <typename TensorType>
class CorrelationConvolutionLargeDirectTestFloatTypes
    : public CorrelationConvolutionLargeTest<TensorType, MATX_C_METHOD_DIRECT> {
};

template <typename TensorType>
class CorrelationConvolution2DTestFloatTypes
    : public CorrelationConvolution2DTest<TensorType> {
};

template <typename TensorType>
class CorrelationConvolutionComplexTypes
    : public CorrelationConvolutionTest<TensorType> {
};

using ExecutorTypesHostOnly = cuda::std::tuple<matx::SingleThreadedHostExecutor,
                                               matx::AllThreadsHostExecutor,
                                               matx::SelectThreadsHostExecutor>;
using MatXFloatNonComplexNonHalfTypesHostExecs =
    TupleToTypes<TypedCartesianProduct<MatXFloatNonComplexNonHalfTuple, ExecutorTypesHostOnly>::type>::type;

template <typename TensorType>
class CorrelationConvolutionHostDirectTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, TensorType>;
  using GExecType = cuda::std::tuple_element_t<1, TensorType>;

  GExecType exec{};
};

TYPED_TEST_SUITE(CorrelationConvolutionDirectTestFloatTypes, MatXFloatTypesCUDAExec);
TYPED_TEST_SUITE(CorrelationConvolutionFFTTestNonHalfFloatTypes, MatXFloatNonHalfTypesAllExecs);
TYPED_TEST_SUITE(CorrelationConvolutionLargeDirectTestFloatTypes, MatXFloatNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(CorrelationConvolutionLargeFFTTestFloatTypes, MatXFloatNonHalfTypesAllExecs);
TYPED_TEST_SUITE(CorrelationConvolution2DTestFloatTypes, MatXFloatNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(CorrelationConvolutionHostDirectTest, MatXFloatNonComplexNonHalfTypesHostExecs);

TYPED_TEST(CorrelationConvolutionHostDirectTest, Direct1DConvolutionModes)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto in = make_tensor<TestType>({4}, MATX_HOST_MALLOC_MEMORY);
  auto filter = make_tensor<TestType>({3}, MATX_HOST_MALLOC_MEMORY);
  auto out_full = make_tensor<TestType>({6}, MATX_HOST_MALLOC_MEMORY);
  auto out_same = make_tensor<TestType>({4}, MATX_HOST_MALLOC_MEMORY);
  auto out_valid = make_tensor<TestType>({2}, MATX_HOST_MALLOC_MEMORY);

  in(0) = static_cast<TestType>(1);
  in(1) = static_cast<TestType>(2);
  in(2) = static_cast<TestType>(3);
  in(3) = static_cast<TestType>(4);
  filter(0) = static_cast<TestType>(2);
  filter(1) = static_cast<TestType>(1);
  filter(2) = static_cast<TestType>(-1);

  (out_full = conv1d(in, filter, MATX_C_MODE_FULL)).run(this->exec);
  (out_same = conv1d(in, filter, MATX_C_MODE_SAME)).run(this->exec);
  (out_valid = conv1d(in, filter, MATX_C_MODE_VALID)).run(this->exec);

  const double expected_full[] = {2, 5, 7, 9, 1, -4};
  const double expected_same[] = {5, 7, 9, 1};
  const double expected_valid[] = {7, 9};

  for (int i = 0; i < 6; i++) {
    ASSERT_NEAR(static_cast<double>(out_full(i)), expected_full[i], 0.001);
  }
  for (int i = 0; i < 4; i++) {
    ASSERT_NEAR(static_cast<double>(out_same(i)), expected_same[i], 0.001);
  }
  for (int i = 0; i < 2; i++) {
    ASSERT_NEAR(static_cast<double>(out_valid(i)), expected_valid[i], 0.001);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionHostDirectTest, Direct1DConvolutionBatchedStridedModes)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr index_t batches = 3;
  constexpr index_t signal_len = 257;
  constexpr index_t filter_len = 33;
  constexpr index_t full_len = signal_len + filter_len - 1;
  constexpr index_t valid_len = signal_len - filter_len + 1;

  auto in_storage = make_tensor<TestType>({signal_len, batches}, MATX_HOST_MALLOC_MEMORY);
  auto filter_storage = make_tensor<TestType>({filter_len, batches}, MATX_HOST_MALLOC_MEMORY);
  auto in = in_storage.Permute({1, 0});
  auto filter = filter_storage.Permute({1, 0});
  auto out_full = make_tensor<TestType>({batches, full_len}, MATX_HOST_MALLOC_MEMORY);
  auto out_same = make_tensor<TestType>({batches, signal_len}, MATX_HOST_MALLOC_MEMORY);
  auto out_valid = make_tensor<TestType>({batches, valid_len}, MATX_HOST_MALLOC_MEMORY);
  auto corr_full = make_tensor<TestType>({batches, full_len}, MATX_HOST_MALLOC_MEMORY);
  auto corr_same = make_tensor<TestType>({batches, signal_len}, MATX_HOST_MALLOC_MEMORY);
  auto corr_valid = make_tensor<TestType>({batches, valid_len}, MATX_HOST_MALLOC_MEMORY);

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < signal_len; i++) {
      const double val = static_cast<double>((i % 17) - 8) * 0.05 + static_cast<double>(b + 1) * 0.125;
      in(b, i) = static_cast<TestType>(val);
    }

    for (index_t f = 0; f < filter_len; f++) {
      const double val = static_cast<double>((f % 9) - 4) * 0.025 + static_cast<double>(b + 1) * 0.01;
      filter(b, f) = static_cast<TestType>(val);
    }
  }

  (out_full = conv1d(in, filter, MATX_C_MODE_FULL)).run(this->exec);
  (out_same = conv1d(in, filter, MATX_C_MODE_SAME)).run(this->exec);
  (out_valid = conv1d(in, filter, MATX_C_MODE_VALID)).run(this->exec);
  (corr_full = corr(in, filter, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT)).run(this->exec);
  (corr_same = corr(in, filter, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run(this->exec);
  (corr_valid = corr(in, filter, MATX_C_MODE_VALID, MATX_C_METHOD_DIRECT)).run(this->exec);

  auto expected = [&](index_t b, index_t out_idx, matxConvCorrMode_t mode) {
    index_t start = 0;
    if (mode == MATX_C_MODE_SAME) {
      start = (filter_len - 1) / 2;
    }
    else if (mode == MATX_C_MODE_VALID) {
      start = filter_len - 1;
    }

    const index_t full_idx = out_idx + start;
    double sum = 0.0;
    for (index_t f = 0; f < filter_len; f++) {
      const index_t sig_idx = full_idx - f;
      if (sig_idx >= 0 && sig_idx < signal_len) {
        sum += static_cast<double>(in(b, sig_idx)) * static_cast<double>(filter(b, f));
      }
    }

    return sum;
  };

  auto expected_corr = [&](index_t b, index_t out_idx, matxConvCorrMode_t mode) {
    index_t start = 0;
    if (mode == MATX_C_MODE_SAME) {
      start = (filter_len - 1) / 2;
    }
    else if (mode == MATX_C_MODE_VALID) {
      start = filter_len - 1;
    }

    const index_t full_idx = out_idx + start;
    double sum = 0.0;
    for (index_t f = 0; f < filter_len; f++) {
      const index_t sig_idx = full_idx - f;
      if (sig_idx >= 0 && sig_idx < signal_len) {
        sum += static_cast<double>(in(b, sig_idx)) * static_cast<double>(filter(b, filter_len - 1 - f));
      }
    }

    return sum;
  };

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < full_len; i++) {
      ASSERT_NEAR(static_cast<double>(out_full(b, i)), expected(b, i, MATX_C_MODE_FULL), 0.001);
      ASSERT_NEAR(static_cast<double>(corr_full(b, i)), expected_corr(b, i, MATX_C_MODE_FULL), 0.001);
    }
    for (index_t i = 0; i < signal_len; i++) {
      ASSERT_NEAR(static_cast<double>(out_same(b, i)), expected(b, i, MATX_C_MODE_SAME), 0.001);
      ASSERT_NEAR(static_cast<double>(corr_same(b, i)), expected_corr(b, i, MATX_C_MODE_SAME), 0.001);
    }
    for (index_t i = 0; i < valid_len; i++) {
      ASSERT_NEAR(static_cast<double>(out_valid(b, i)), expected(b, i, MATX_C_MODE_VALID), 0.001);
      ASSERT_NEAR(static_cast<double>(corr_valid(b, i)), expected_corr(b, i, MATX_C_MODE_VALID), 0.001);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionHostDirectTest, Direct2DConvolutionModes)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto in = make_tensor<TestType>({3, 3}, MATX_HOST_MALLOC_MEMORY);
  auto filter = make_tensor<TestType>({2, 2}, MATX_HOST_MALLOC_MEMORY);
  auto out_full = make_tensor<TestType>({4, 4}, MATX_HOST_MALLOC_MEMORY);
  auto out_same = make_tensor<TestType>({3, 3}, MATX_HOST_MALLOC_MEMORY);
  auto out_valid = make_tensor<TestType>({2, 2}, MATX_HOST_MALLOC_MEMORY);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      in(i, j) = static_cast<TestType>(i * 3 + j + 1);
    }
  }

  filter(0, 0) = static_cast<TestType>(1);
  filter(0, 1) = static_cast<TestType>(2);
  filter(1, 0) = static_cast<TestType>(3);
  filter(1, 1) = static_cast<TestType>(4);

  (out_full = conv2d(in, filter, MATX_C_MODE_FULL)).run(this->exec);
  (out_same = conv2d(in, filter, MATX_C_MODE_SAME)).run(this->exec);
  (out_valid = conv2d(in, filter, MATX_C_MODE_VALID)).run(this->exec);

  const double expected_full[4][4] = {
      {1, 4, 7, 6},
      {7, 23, 33, 24},
      {19, 53, 63, 42},
      {21, 52, 59, 36}};
  const double expected_same[3][3] = {
      {1, 4, 7},
      {7, 23, 33},
      {19, 53, 63}};
  const double expected_valid[2][2] = {
      {23, 33},
      {53, 63}};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ASSERT_NEAR(static_cast<double>(out_full(i, j)), expected_full[i][j], 0.001);
    }
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      ASSERT_NEAR(static_cast<double>(out_same(i, j)), expected_same[i][j], 0.001);
    }
  }
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      ASSERT_NEAR(static_cast<double>(out_valid(i, j)), expected_valid[i][j], 0.001);
    }
  }

  MATX_EXIT_HANDLER();
}

// Real/real direct 1D convolution Large
TYPED_TEST(CorrelationConvolutionLargeDirectTestFloatTypes, Direct1DConvolutionLarge)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckDirect1DConvSupport<ExecType>()) {
    GTEST_SKIP();
  } else {
    this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len, b_len});
    this->pb->RunTVGenerator("conv");
    this->pb->NumpyToTensorView(this->av, "a_op");
    this->pb->NumpyToTensorView(this->bv, "b_op");
    // example-begin conv1d-test-1
    // 1D convolution in FULL mode where every output is stored
    (this->cv = conv1d(this->av, this->bv, MATX_C_MODE_FULL)).run(this->exec);
    // example-end conv1d-test-1

    MATX_TEST_ASSERT_COMPARE(this->pb, this->cv, "conv_full", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionLargeFFTTestFloatTypes, FFT1DConvolutionLarge)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len, b_len});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv, "b_op");

  // 1D convolution in FULL mode where every output is stored
  (this->cv = conv1d(this->av, this->bv, MATX_C_MODE_FULL, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv, "conv_full", this->thresh);
  
  MATX_EXIT_HANDLER();
}


// Real/real direct 1D convolution
TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DConvolutionFullEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_full_even = conv1d(this->av, this->bv_even, MATX_C_MODE_FULL)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionFFTTestNonHalfFloatTypes, FFT1DConvolutionFullEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_full_even = conv1d(this->av, this->bv_even, MATX_C_MODE_FULL, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}


// Real/real direct 2D convolution
TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionFullEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_even, b_len1_even});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_full_even = conv2d(this->av, this->bv_even, MATX_C_MODE_FULL)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}



TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DConvolutionSameEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_same = conv1d(this->av, this->bv_even, MATX_C_MODE_SAME)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionFFTTestNonHalfFloatTypes, FFT1DConvolutionSameEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_same = conv1d(this->av, this->bv_even, MATX_C_MODE_SAME, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionSameEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_even, b_len1_even});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  // example-begin conv2d-test-1
  (this->cv_same = conv2d(this->av, this->bv_even, MATX_C_MODE_SAME)).run(this->exec);
  // example-end conv2d-test-1
  
  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DConvolutionValidEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_valid_even = conv1d(this->av, this->bv_even, MATX_C_MODE_VALID)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_even, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionFFTTestNonHalfFloatTypes, FFT1DConvolutionValidEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_valid_even = conv1d(this->av, this->bv_even, MATX_C_MODE_VALID, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_even, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionValidEven)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_even, b_len1_even});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_valid_even = conv2d(this->av, this->bv_even, MATX_C_MODE_VALID)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_even, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DConvolutionFullOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_odd});  
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_full_odd = conv1d(this->av, this->bv_odd, MATX_C_MODE_FULL)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_odd, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionFFTTestNonHalfFloatTypes, FFT1DConvolutionFullOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_odd});  
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_full_odd = conv1d(this->av, this->bv_odd, MATX_C_MODE_FULL, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_odd, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}



TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionFullOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_odd, b_len1_odd});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_full_odd = conv2d(this->av, this->bv_odd, MATX_C_MODE_FULL)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_odd, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DConvolutionSameOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_odd});   
  this->pb->RunTVGenerator("conv");   
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_same = conv1d(this->av, this->bv_odd, MATX_C_MODE_SAME)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionFFTTestNonHalfFloatTypes, FFT1DConvolutionSameOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_odd});   
  this->pb->RunTVGenerator("conv");   
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_same = conv1d(this->av, this->bv_odd, MATX_C_MODE_SAME, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionSameOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_odd, b_len1_odd});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_same = conv2d(this->av, this->bv_odd, MATX_C_MODE_SAME)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_same, "conv_same", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DConvolutionValidOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_odd});   
  this->pb->RunTVGenerator("conv");   
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_valid_odd = conv1d(this->av, this->bv_odd, MATX_C_MODE_VALID)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_odd, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionFFTTestNonHalfFloatTypes, FFT1DConvolutionValidOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_odd});   
  this->pb->RunTVGenerator("conv");   
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_valid_odd = conv1d(this->av, this->bv_odd, MATX_C_MODE_VALID, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_odd, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionValidOdd)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_odd, b_len1_odd});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_odd, "b_op");
  (this->cv_valid_odd = conv2d(this->av, this->bv_odd, MATX_C_MODE_VALID)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_valid_odd, "conv_valid", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DConvolutionSwap)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_full_even = conv1d(this->bv_even, this->av, MATX_C_MODE_FULL)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionFFTTestNonHalfFloatTypes, FFT1DConvolutionSwap)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});
  this->pb->RunTVGenerator("conv");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_full_even = conv1d(this->bv_even, this->av, MATX_C_MODE_FULL, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Direct2DConvolutionSwap)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv2d_operators", {a_len0, a_len1, b_len0_even, b_len1_even});
  this->pb->RunTVGenerator("conv2d");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_full_even = conv2d(this->bv_even, this->av, MATX_C_MODE_FULL)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "conv_full", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DCorrelation)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});  
  this->pb->RunTVGenerator("corr");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  // example-begin corr-test-1
  // Full correlation mode with direct correlation
  (this->cv_full_even = corr(this->av, this->bv_even, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT)).run(this->exec);
  // example-end corr-test-1

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "corr", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionFFTTestNonHalfFloatTypes, FFT1DCorrelation)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});  
  this->pb->RunTVGenerator("corr");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  // Full correlation mode with direct correlation
  (this->cv_full_even = corr(this->av, this->bv_even, MATX_C_MODE_FULL, MATX_C_METHOD_FFT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "corr", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Direct1DCorrelationSwap)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->template InitTVGenerator<TestType>("00_transforms", "conv_operators", {a_len0, b_len0_even});  
  this->pb->RunTVGenerator("corr_swap");
  this->pb->NumpyToTensorView(this->av, "a_op");
  this->pb->NumpyToTensorView(this->bv_even, "b_op");
  (this->cv_full_even = corr(this->bv_even, this->av, MATX_C_MODE_FULL, MATX_C_METHOD_DIRECT)).run(this->exec);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv_full_even, "corr_swap", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolutionDirectTestFloatTypes, Conv1Axis)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  const int d1 = 8;
  const int d2 = 512;
  const int d3 = 1024;

  auto in1 = make_tensor<TestType>({d1, d2, d3});
  auto in2 = make_tensor<TestType>({d1, d2, d3});
  auto out1 = make_tensor<TestType>({d1, d2, d3});
  auto out2 = make_tensor<TestType>({d1, d2, d3});

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        in1(i,j,k) = static_cast<TestType>((float)(i+j+k));
        in2(i,j,k) = static_cast<TestType>((float)(1));
      }
    }
  }

  (out1 = conv1d(in1, in2, MATX_C_MODE_SAME)).run(this->exec);
  // example-begin conv1d-test-2
  (out2 = conv1d(in1, in2, {2}, MATX_C_MODE_SAME)).run(this->exec);
  // example-end conv1d-test-2

  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1.Permute({0,2,1}) = conv1d(in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME)).run(this->exec);
  // example-begin conv1d-test-3
  (out2 = conv1d(in1, in2, {1}, MATX_C_MODE_SAME)).run(this->exec);
  // example-end conv1d-test-3

  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1.Permute({1,2,0}) = conv1d(in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME)).run(this->exec);
  (out2 = conv1d(in1, in2, {0}, MATX_C_MODE_SAME)).run(this->exec);

  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1 = corr(in1, in2, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run(this->exec);
  (out2 = corr(in1, in2, {2}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run(this->exec);

  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1.Permute({0,2,1}) = corr(in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run(this->exec);
  (out2 = corr(in1, in2, {1}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run(this->exec);

  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  (out1.Permute({1,2,0}) = corr(in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run(this->exec);
  (out2 = corr(in1, in2, {0}, MATX_C_MODE_SAME, MATX_C_METHOD_DIRECT)).run(this->exec);

  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CorrelationConvolution2DTestFloatTypes, Conv2Axis)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
#if 1  // currently doesn't work because Conv2D requires rank2 filter.
  const int d1 = 8;
  const int d2 = 512;
  const int d3 = 1024;

  auto in1 = make_tensor<TestType>({d1, d2, d3});
  auto in2 = make_tensor<TestType>({d1, d2, d3});
  auto out1 = make_tensor<TestType>({d1, d2, d3});
  auto out2 = make_tensor<TestType>({d1, d2, d3});

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
			for(int k = 0; k < d3; k++) {
				in1(i,j,k) = static_cast<TestType>((float)(i+j+k));
				in2(i,j,k) = static_cast<TestType>((float)(1));
      }
    }
  }

  (out1 = conv2d(in1, in2, MATX_C_MODE_SAME)).run(this->exec);
  (out2 = conv2d(in1, in2, {1, 2}, MATX_C_MODE_SAME)).run(this->exec);

  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }
 
  (out1.Permute({0,2,1}) = conv2d(in1.Permute({0,2,1}), in2.Permute({0,2,1}), MATX_C_MODE_SAME)).run(this->exec);
  (out2 = conv2d(in1, in2, {2, 1}, MATX_C_MODE_SAME)).run(this->exec);

  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k));
      }
    }
  }
  
  (out1.Permute({1,2,0}) = conv2d(in1.Permute({1,2,0}), in2.Permute({1,2,0}), MATX_C_MODE_SAME)).run(this->exec);
  (out2 = conv2d(in1, in2, {2, 0}, MATX_C_MODE_SAME)).run(this->exec);

  this->exec.sync();

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

#endif
