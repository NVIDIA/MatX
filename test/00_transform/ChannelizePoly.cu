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
#include <cuda/std/complex>

using namespace matx;

template <typename T>
class ChannelizePolyTest : public ::testing::Test {
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;
protected:
  void SetUp() override
  {
    CheckTestTypeSupport<GTestType>();
    pb = std::make_unique<detail::MatXPybind>();

    if constexpr (is_complex_half_v<GTestType> || is_matx_half_v<GTestType>) {
      thresh = 1.0e-1;
    } else if constexpr (std::is_same_v<GTestType, double>) {
      thresh = 1.0e-12;
    } else {
      // Revisit this tolerance. We should likely use a relative tolerance
      // rather than absolute for larger values.
      thresh = 1.0e-3;
    }
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  double thresh;
};

template <typename TensorType>
class ChannelizePolyTestNonHalfFloatTypes
    : public ChannelizePolyTest<TensorType> {
};

template <typename TensorType>
class ChannelizePolyTestDoubleType
    : public ChannelizePolyTest<TensorType> {
};

template <typename TensorType>
class ChannelizePolyTestFloatTypes
    : public ChannelizePolyTest<TensorType> {
};

namespace test_types {
    template<typename T>
    struct inner_type {
      using type = T;
    };

    template<typename T>
    struct inner_type<std::complex<T>> {
        using type = typename std::complex<T>::value_type;
    };

    template<typename T>
    struct inner_type<cuda::std::complex<T>> {
        using type = typename cuda::std::complex<T>::value_type;
    };

    template<typename T>
    struct complex_type {
      using type = typename cuda::std::complex<T>;
    };

    template<typename T>
    struct complex_type<std::complex<T>> {
      using type = typename std::complex<T>;
    };

    template<typename T>
    struct complex_type<cuda::std::complex<T>> {
      using type = typename cuda::std::complex<T>;
    };
}

TYPED_TEST_SUITE(ChannelizePolyTestNonHalfFloatTypes, MatXFloatNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(ChannelizePolyTestDoubleType, MatXDoubleOnlyTypeCUDAExec);

// Simple tests use random input and filter values
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, Simple)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
  } test_cases[] = {
    { 2500, 170, 10 },
    { 2500, 187, 11 },
    { 1800, 120, 2 },
    { 1800, 120, 3 },
    { 35000, 5*180, 180 },
    { 35000, 5*181, 181 },
    { 37193, 41*8, 8 },
    { 37193, 41*9, 9 },
    { 35000, 5*180+33, 180 },
    { 35000, 5*181+17, 181 },
    { 37193, 41*8+4, 8 },
    { 37193, 41*9+8, 9 },
    { 271373, 31*13+3, 13 },
    { 271374, 31*14+4, 14 },
    { 27137, 301*13+3, 13 },
    { 27138, 301*14+4, 14 },
    { 1000000, 32*16, 32 },
    { 1000000, 40*16, 40 }
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = num_channels;
    [[maybe_unused]] const index_t b_len_per_channel = (a_len + num_channels - 1) / num_channels;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize", {a_len, f_len, num_channels, num_channels});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    // example-begin channelize_poly-test-1
    // Channelize "a" into "num_channels" channels using filter "f" and "decimation_factor" decimation
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);
    // example-end channelize_poly-test-1

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);    

    // Now test with a multiplicative operator on the input. The channelizer is linear,
    // so we can inverse-scale the output to compare against the golden outputs.
    (b = channelize_poly(static_cast<TestType>(4.0) * a, f, num_channels, decimation_factor)).run(this->exec);
    (b = b * static_cast<TestType>(0.25)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// num_channels == 1 is a degenerate case: no channelization and (with D==M==1)
// no decimation, so channelize_poly collapses to a plain FIR. The dispatcher
// used to throw here because the FusedChan switch started at N=2; now it
// falls through to Smem / SmemTiled / Generic. Verify numeric correctness
// against a hand-rolled FIR reference.
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, NumChannelsOne)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  const index_t a_len = 128;
  const index_t f_len = 9;
  const index_t num_channels = 1;
  const index_t decimation_factor = 1;
  const index_t b_len_per_channel = (a_len + num_channels - 1) / num_channels;

  auto a = make_tensor<TestType>({a_len});
  auto f = make_tensor<TestType>({f_len});
  auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

  // Initialize input and filter with simple deterministic values via direct
  // host operator() writes. Handles both real and complex TestType uniformly.
  for (index_t i = 0; i < a_len; i++) {
    if constexpr (is_complex_v<TestType>) {
      using scalar_t = typename test_types::inner_type<TestType>::type;
      a(i) = TestType{static_cast<scalar_t>(i + 1), static_cast<scalar_t>(-(i % 7))};
    } else {
      a(i) = static_cast<TestType>(i + 1);
    }
  }
  for (index_t k = 0; k < f_len; k++) {
    if constexpr (is_complex_v<TestType>) {
      using scalar_t = typename test_types::inner_type<TestType>::type;
      f(k) = TestType{static_cast<scalar_t>(k + 1), static_cast<scalar_t>(0)};
    } else {
      f(k) = static_cast<TestType>(k + 1);
    }
  }

  (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);
  this->exec.sync();

  // Reference FIR in double precision: b_ref[t] = sum_{k=0..F-1} f[k] * a[t-k]
  // with zero padding for t-k < 0.
  for (index_t t = 0; t < a_len; t++) {
    cuda::std::complex<double> expect{0.0, 0.0};
    for (index_t k = 0; k < f_len; k++) {
      if (t - k >= 0) {
        if constexpr (is_complex_v<TestType>) {
          const auto ak = a(t - k);
          const auto fk = f(k);
          expect += cuda::std::complex<double>{
              static_cast<double>(fk.real()) * static_cast<double>(ak.real())
                - static_cast<double>(fk.imag()) * static_cast<double>(ak.imag()),
              static_cast<double>(fk.real()) * static_cast<double>(ak.imag())
                + static_cast<double>(fk.imag()) * static_cast<double>(ak.real())};
        } else {
          const double fk = static_cast<double>(f(k));
          const double ak = static_cast<double>(a(t - k));
          expect += cuda::std::complex<double>{fk * ak, 0.0};
        }
      }
    }
    const auto got = b(t, 0);
    const double mag = std::abs(expect.real()) + std::abs(expect.imag());
    ASSERT_NEAR(static_cast<double>(got.real()), expect.real(), this->thresh * (1.0 + mag))
        << "real-part mismatch at t=" << t;
    ASSERT_NEAR(static_cast<double>(got.imag()), expect.imag(), this->thresh * (1.0 + mag))
        << "imag-part mismatch at t=" << t;
  }

  MATX_EXIT_HANDLER();
}

// Mixed type tests verify that mixing float and double for the input and filter behaves
// as expected (e.g., that the output is the complex version of the higher precision).
TYPED_TEST(ChannelizePolyTestDoubleType, MixedPrecision)
{
  MATX_ENTER_HANDLER();

  cudaStream_t stream = 0;

  const index_t a_len = 2500;
  const index_t f_len = 170;
  const index_t num_channels = 10;
  const index_t decimation_factor = num_channels;
  const index_t b_len_per_channel = (a_len + num_channels - 1) / num_channels;
  const double mixed_thresh = 1e-5;

  this->pb->template InitAndRunTVGenerator<double>(
    "00_transforms", "channelize_poly_operators", "channelize", {a_len, f_len, num_channels, num_channels});
  auto a64 = make_tensor<double>({a_len});
  auto f64 = make_tensor<double>({f_len});
  this->pb->NumpyToTensorView(a64, "a");
  this->pb->NumpyToTensorView(f64, "filter_random");

  // Double precision input, single precision filter
  {
    auto f32 = make_tensor<float>({f_len});
    (f32 = as_float(f64)).run(this->exec);
    auto b = make_tensor<cuda::std::complex<double>>({b_len_per_channel, num_channels});
    (b = channelize_poly(a64, f32, num_channels, decimation_factor)).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", mixed_thresh);
  }

  // Single precision input, double precision filter
  {
    auto a32 = make_tensor<float>({a_len});
    (a32 = as_float(a64)).run(this->exec);
    auto b = make_tensor<cuda::std::complex<double>>({b_len_per_channel, num_channels});
    (b = channelize_poly(a32, f64, num_channels, decimation_factor)).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", mixed_thresh);
  }

  this->pb->template InitAndRunTVGenerator<cuda::std::complex<double>>(
    "00_transforms", "channelize_poly_operators", "channelize", {a_len, f_len, num_channels, num_channels});
  auto ac64 = make_tensor<cuda::std::complex<double>>({a_len});
  this->pb->NumpyToTensorView(ac64, "a");
  this->pb->NumpyToTensorView(f64, "filter_random_real");

  // Double precision complex input, single precision filter
  {
    auto f32 = make_tensor<float>({f_len});
    (f32 = as_float(f64)).run(this->exec);
    auto b = make_tensor<cuda::std::complex<double>>({b_len_per_channel, num_channels});
    (b = channelize_poly(ac64, f32, num_channels, decimation_factor)).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random_hreal", mixed_thresh);
  }

  // Single precision complex input, double precision filter
  {
    auto ac32 = make_tensor<cuda::std::complex<float>>({a_len});
    (ac32 = as_complex_float(ac64)).run(this->exec);

    auto b = make_tensor<cuda::std::complex<double>>({b_len_per_channel, num_channels});
    (b = channelize_poly(ac32, f64, num_channels, decimation_factor)).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random_hreal", mixed_thresh);
  }

  MATX_EXIT_HANDLER();
}

// Tests that verify that the accumulator property is used to set the type of the accumulator.
// Also tests that the output property is used to set the type of the output.
TYPED_TEST(ChannelizePolyTestDoubleType, AccumProperty)
{
  MATX_ENTER_HANDLER();

  cudaStream_t stream = 0;

  const index_t a_len = 2500;
  const index_t f_len = 155;
  const index_t num_channels = 5;
  const index_t decimation_factor = num_channels;
  const index_t b_len_per_channel = (a_len + num_channels - 1) / num_channels;
  const double mixed_thresh = 1e-5;
  const double mixed_thresh_complex_input = 1e-4;

  this->pb->template InitAndRunTVGenerator<double>(
    "00_transforms", "channelize_poly_operators", "channelize", {a_len, f_len, num_channels, num_channels});
  auto a64 = make_tensor<double>({a_len});
  auto f64 = make_tensor<double>({f_len});
  auto gold_output_hreal = make_tensor<cuda::std::complex<double>>({b_len_per_channel, num_channels});
  this->pb->NumpyToTensorView(a64, "a");
  this->pb->NumpyToTensorView(f64, "filter_random");
  this->pb->NumpyToTensorView(gold_output_hreal, "b_random_hreal");

  auto a32 = make_tensor<float>({a_len});
  (a32 = as_float(a64)).run(this->exec);
  auto f32 = make_tensor<float>({f_len});
  (f32 = as_float(f64)).run(this->exec);
  auto b32 = make_tensor<cuda::std::complex<float>>({b_len_per_channel, num_channels});
  auto b64 = make_tensor<cuda::std::complex<double>>({b_len_per_channel, num_channels});

  // Single precision input, output, and filter, double precision accumulator
  {
    auto chan_poly = channelize_poly(a32, f32, num_channels, decimation_factor).props<PropAccum<double>>();
    (b32 = chan_poly).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b32, "b_random", mixed_thresh);
  }

  // Single precision input, double precision filter and accumulator
  {
    auto chan_poly = channelize_poly(a32, f64, num_channels, decimation_factor).props<PropAccum<double>>();
    (b64 = chan_poly).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b64, "b_random", mixed_thresh);
  }

  // Double precision input and accumulator, single precision filter
  {
    auto chan_poly = channelize_poly(a64, f32, num_channels, decimation_factor).props<PropAccum<double>>();
    (b64 = chan_poly).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b64, "b_random", mixed_thresh);
  }

  // Double precision input and filter, single precision accumulator. This would be an odd
  // configuration, but technically allowed.
  {
    auto chan_poly = channelize_poly(a64, f64, num_channels, decimation_factor).props<PropAccum<float>>();
    (b32 = chan_poly).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b32, "b_random", mixed_thresh);
  }

  this->pb->template InitAndRunTVGenerator<cuda::std::complex<double>>(
    "00_transforms", "channelize_poly_operators", "channelize", {a_len, f_len, num_channels, num_channels});
  auto ac64 = make_tensor<cuda::std::complex<double>>({a_len});
  this->pb->NumpyToTensorView(ac64, "a");
  this->pb->NumpyToTensorView(f64, "filter_random_real");
  this->pb->NumpyToTensorView(gold_output_hreal, "b_random_hreal");

  // The following cases are all for complex inputs and real filters

  auto ac32 = make_tensor<cuda::std::complex<float>>({a_len});
  (ac32 = as_complex_float(ac64)).run(this->exec);
  (f32 = as_float(f64)).run(this->exec);

  // Below, we test that using a double-precision accumulator by running with fp32 inputs with and
  // without a double-precision accumulator demonstrates higher accuracy when an fp64 accumulator.
  // Note that for the single-precision test, we need to write to a single-precision output or the
  // accumulator would by default use the output type (i.e,. cuda::std::complex<double> for b64).
  auto max_err = make_tensor<double>({});
  double max_err_fp32{};
  double max_err_fp32_in_fp64_accum{};

  // All single precision
  {
    auto chan_poly = channelize_poly(ac32, f32, num_channels, decimation_factor);
    (b32 = chan_poly).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b32, "b_random_hreal", mixed_thresh_complex_input);
    (max_err = matx::max(matx::abs(as_complex_double(b32) - gold_output_hreal), {0,1})).run(this->exec);
    cudaStreamSynchronize(stream);
    max_err_fp32 = max_err();
  }

  // Single precision complex input, output, and filter, double precision accumulator
  {
    auto chan_poly = channelize_poly(ac32, f32, num_channels, decimation_factor)
      .props<PropAccum<double>, PropOutput<cuda::std::complex<double>>>();
    // Add a dummy complex value to force a PreRun call on chan_poly, which
    // without the PropOutput would create a single-precision complex output type for
    // the intermediate result.
    (b64 = chan_poly + cuda::std::complex<double>(0.0, 0.0)).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b64, "b_random_hreal", mixed_thresh_complex_input);
    (max_err = matx::max(matx::abs(b64 - gold_output_hreal), {0,1})).run(this->exec);
    cudaStreamSynchronize(stream);
    max_err_fp32_in_fp64_accum = max_err();
  }

  // We expect a bit better than 8x improvement with the fp64 accumulator for this case
  ASSERT_GT(max_err_fp32, 8.0 * max_err_fp32_in_fp64_accum) << "expect > 8x improvement with fp64 accumulator";

  // Single precision complex input, double precision filter and accumulator
  {
    // example-begin channelize_poly-test-2
    // Single precision complex input (ac32) and double-precision filter (f64).
    // Properties force double-precision accumulator and output.
    auto chan_poly = channelize_poly(ac32, f64, num_channels, decimation_factor)
      .props<PropAccum<double>, PropOutput<cuda::std::complex<double>>>();
    (b64 = chan_poly).run(this->exec);
    // example-end channelize_poly-test-2
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b64, "b_random_hreal", mixed_thresh_complex_input);
  }

  // Double precision complex input and accumulator, single precision filter
  {
    auto chan_poly = channelize_poly(ac64, f32, num_channels, decimation_factor).props<PropAccum<double>>();
    (b64 = chan_poly).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b64, "b_random_hreal", mixed_thresh_complex_input);
  }

  // Double precision complex input and filter, single precision accumulator. This would be an odd
  // configuration, but technically allowed.
  {
    auto chan_poly = channelize_poly(ac64, f64, num_channels, decimation_factor).props<PropAccum<float>>();
    (b64 = chan_poly).run(this->exec);
    cudaStreamSynchronize(stream);
    MATX_TEST_ASSERT_COMPARE(this->pb, b64, "b_random_hreal", mixed_thresh_complex_input);
  }

  MATX_EXIT_HANDLER();
}

// Batched tests use random input and filter values with two batch dimensions.
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, Batched)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  constexpr int BATCH_DIMS = 2;
  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    cuda::std::array<index_t, BATCH_DIMS> batch_dims;
  } test_cases[] = {
    { 2500, 170, 10, { 2, 3 } },
    { 2500, 187, 11, { 11, 14 } },
    { 37193, 41*8+4, 8, { 3, 1 } },
    { 37205, 41*9+8, 9, { 4, 5 } },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = num_channels;
    const index_t b_len_per_channel = (a_len + num_channels - 1) / num_channels;
    std::vector<index_t> sizes = { a_len, f_len, num_channels, decimation_factor };
    sizes.insert(sizes.end(), test_cases[i].batch_dims.begin(), test_cases[i].batch_dims.end());
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize", sizes);
    const cuda::std::array<index_t, BATCH_DIMS+1> a_dims = {
      test_cases[i].batch_dims[0], test_cases[i].batch_dims[1], a_len
    };
    const cuda::std::array<index_t, BATCH_DIMS+2> b_dims = {
      test_cases[i].batch_dims[0], test_cases[i].batch_dims[1], b_len_per_channel, num_channels
    };
    auto a = make_tensor<TestType>(a_dims);
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>(b_dims);

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);    

    // Now test with a multiplicative operator on the input. The channelizer is linear,
    // so we can inverse-scale the output to compare against the golden outputs.
    (b = channelize_poly(static_cast<TestType>(4.0) * a, f, num_channels, decimation_factor)).run(this->exec);
    (b = b * static_cast<TestType>(0.25)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// The identity filter has a single tap of value 1.0 for each channel. Thus, the filtered
// values will be single values selected from the input signal and the output will be the
// DFT thereof in the channel dimension (up to the sign on the complex exponential in the
// DFT).
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, IdentityFilter)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using InnerType = typename test_types::inner_type<TestType>::type;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  const index_t a_len = 4;
  [[maybe_unused]] const index_t f_len = 2;
  // It would be simpler to test with 1 channel, but we currently do not support single channel
  // cases (a single channel channelizer is just a convolution).
  const index_t num_channels = 2;
  this->pb->template InitAndRunTVGenerator<TestType>(
    "00_transforms", "channelize_poly_operators", "channelize", { a_len, f_len, num_channels, num_channels });

  auto a = make_tensor<TestType>({a_len});
  auto f = make_tensor<TestType>({f_len});
  const index_t b_elem_per_channel = a_len / num_channels;
  auto b = make_tensor<ComplexType>({ b_elem_per_channel, num_channels });

  this->pb->NumpyToTensorView(a, "a");
  for (auto i = 0; i < num_channels; i++) { f(i) = 1; }

  this->exec.sync();

  const index_t decimation_factor = num_channels;
  (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

  this->exec.sync();

  for (auto k = 0; k < b_elem_per_channel; k++) {
    // Explicit DFT in the channel dimension. The complex exponential sign here is opposite
    // that typically used for the DFT. See the Harris 2003 paper where the factors applied
    // in the filter bank are exp(j...) instead of exp(-j...).
    for (auto i = 0; i < num_channels; i++) {
      ComplexType accum { 0 };
      for (auto j = 0; j < num_channels; j++) {
        const double arg = 2.0 * M_PI * j * i / num_channels;
        double sinx, cosx;
        sincos(arg, &sinx, &cosx);
        ComplexType rhs { static_cast<InnerType>(cosx), static_cast<InnerType>(sinx) };
        accum += a(k*num_channels+num_channels-1-j) * rhs;
      }
      ComplexType bk { b(k, i) };
      ASSERT_NEAR(accum.real(), bk.real(), this->thresh);
      ASSERT_NEAR(accum.imag(), bk.imag(), this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}

// Tests that involve non-trivial input and output operators.
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, Operators)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  const index_t a_len = 2500;
  [[maybe_unused]] const index_t f_len = 90;
  const index_t num_channels = 10;
  this->pb->template InitAndRunTVGenerator<TestType>(
    "00_transforms", "channelize_poly_operators", "channelize", { a_len, f_len, num_channels, num_channels });

  auto a = make_tensor<TestType>({a_len});
  auto f = make_tensor<TestType>({f_len});
  [[maybe_unused]] const index_t b_elem_per_channel = a_len / num_channels;
  auto bp = make_tensor<ComplexType>({ num_channels, b_elem_per_channel });

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(f, "filter_random");

  this->exec.sync();

  const index_t decimation_factor = num_channels;
  auto b = permute(bp, {1, 0});
  (b = channelize_poly(shift<0>(shift<0>(a, 8), -8), f, num_channels, decimation_factor)).run(this->exec);

  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);

  MATX_EXIT_HANDLER();
}

// Test case inspired by the 10 channel polyphase channelizer example in
// "Digital Receivers and Transmitters Using Polyphase Filter Banks for Wireless Communications",
// F. J. Harris, C. Dick, M. Rice, IEEE Transactions on Microwave Theory and Techniques,
// Vol. 51, No. 4, Apr. 2003.
TYPED_TEST(ChannelizePolyTestDoubleType, Harris2003)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  cudaStream_t stream = 0;

  const std::vector<double> filter = {
    -7.108172299812534e-05, -5.383190907453450e-05, -6.459797687779421e-05, -6.733363514333483e-05,
    -5.745336885182142e-05, -3.031106538335635e-05, 1.815599994873786e-05, 9.100030655126845e-05,
    1.889522626479438e-04, 3.104111057628484e-04, 4.506822359023572e-04, 6.016135746338058e-04,
    7.522206292194672e-04, 8.886408885505310e-04, 9.954660202586407e-04, 1.056790595419086e-03,
    1.058003578493307e-03, 9.873459132155706e-04, 8.379774454391282e-04, 6.092222109556477e-04,
    3.080197822202455e-04, -5.066307344633294e-05, -4.439099570669816e-04, -8.419055679231051e-04,
    -1.209786822776072e-03, -1.510414142280465e-03, -1.707653315939468e-03, -1.770125458485520e-03,
    -1.674998512691401e-03, -1.411346013002809e-03, -9.829186262971859e-04, -4.096733898534489e-04,
    2.720201919377083e-04, 1.010854494593115e-03, 1.743672815412935e-03, 2.400218565611791e-03,
    2.909046961309308e-03, 3.204385264814235e-03, 3.233169385867668e-03, 2.961687936811111e-03,
    2.381113534757274e-03, 1.511217740103499e-03, 4.017303217288107e-04, -8.690672824987330e-04,
    -2.198658200146505e-03, -3.467469072349120e-03, -4.548395876629596e-03, -5.318172235731722e-03,
    -5.669499219624308e-03, -5.523008346973195e-03, -4.837780903924761e-03, -3.619381955857180e-03,
    -1.924297157446905e-03, 1.398783531465044e-04, 2.419052852305444e-03, 4.721994421234152e-03,
    6.834078000511559e-03, 8.534575588665213e-03, 9.616239260189539e-03, 9.905652888254391e-03,
    9.282593063412357e-03, 7.696634771619944e-03, 5.179276629755422e-03, 1.850239376782610e-03,
    -2.083158126677665e-03, -6.333940841377637e-03, -1.055070295418721e-02, -1.433932169672528e-02,
    -1.728987333832456e-02, -1.900686037800291e-02, -1.914055875238895e-02, -1.741700339554898e-02,
    -1.366417081085800e-02, -7.832090674357747e-03, -4.979664675274029e-06, 9.595882106110693e-03,
    2.061921064939119e-02, 3.260052595387335e-02, 4.498784768298310e-02, 5.717457743613726e-02,
    6.853738242014783e-02, 7.847631484703010e-02, 8.645430840298432e-02, 9.203306090698263e-02,
    9.490265763728883e-02, 9.490265763728883e-02, 9.203306090698263e-02, 8.645430840298432e-02,
    7.847631484703010e-02, 6.853738242014783e-02, 5.717457743613726e-02, 4.498784768298310e-02,
    3.260052595387335e-02, 2.061921064939119e-02, 9.595882106110693e-03, -4.979664675274029e-06,
    -7.832090674357747e-03, -1.366417081085800e-02, -1.741700339554898e-02, -1.914055875238895e-02,
    -1.900686037800291e-02, -1.728987333832456e-02, -1.433932169672528e-02, -1.055070295418721e-02,
    -6.333940841377637e-03, -2.083158126677665e-03, 1.850239376782610e-03, 5.179276629755422e-03,
    7.696634771619944e-03, 9.282593063412357e-03, 9.905652888254391e-03, 9.616239260189539e-03,
    8.534575588665213e-03, 6.834078000511559e-03, 4.721994421234152e-03, 2.419052852305444e-03,
    1.398783531465044e-04, -1.924297157446905e-03, -3.619381955857180e-03, -4.837780903924761e-03,
    -5.523008346973195e-03, -5.669499219624308e-03, -5.318172235731722e-03, -4.548395876629596e-03,
    -3.467469072349120e-03, -2.198658200146505e-03, -8.690672824987330e-04, 4.017303217288107e-04,
    1.511217740103499e-03, 2.381113534757274e-03, 2.961687936811111e-03, 3.233169385867668e-03,
    3.204385264814235e-03, 2.909046961309308e-03, 2.400218565611791e-03, 1.743672815412935e-03,
    1.010854494593115e-03, 2.720201919377083e-04, -4.096733898534489e-04, -9.829186262971859e-04,
    -1.411346013002809e-03, -1.674998512691401e-03, -1.770125458485520e-03, -1.707653315939468e-03,
    -1.510414142280465e-03, -1.209786822776072e-03, -8.419055679231051e-04, -4.439099570669816e-04,
    -5.066307344633294e-05, 3.080197822202455e-04, 6.092222109556477e-04, 8.379774454391282e-04,
    9.873459132155706e-04, 1.058003578493307e-03, 1.056790595419086e-03, 9.954660202586407e-04,
    8.886408885505310e-04, 7.522206292194672e-04, 6.016135746338058e-04, 4.506822359023572e-04,
    3.104111057628484e-04, 1.889522626479438e-04, 9.100030655126845e-05, 1.815599994873786e-05,
    -3.031106538335635e-05, -5.745336885182142e-05, -6.733363514333483e-05, -6.459797687779421e-05,
    -5.383190907453450e-05, -7.108172299812534e-05
  };

  const std::vector<double> input = {
    4.000000000000000e+00, 6.426669113781712e+00, -4.084816027431160e-03, -1.878066190437050e+0,
    1.740740863288972e+00, -1.939950007893213e+00, -4.364642989989242e+00, 1.809512656756737e+00,
    3.383246793255789e+00, 4.990169346243767e-01, 3.396802246667419e+00, 4.859659902057119e+00,
    3.488375603944824e-01, -1.026057667011152e+00, 1.961627751131261e-01, -1.568649212747752e+00,
    -2.124605653163686e+00, 3.916029309655906e-01, 1.449734266612877e+00, 1.170207935117502e+00,
    1.999999999999999e+00, 2.156747621959968e+00, 2.435011657637338e-01, -7.976669132726794e-01,
    -1.096619217338606e-01, -1.292893218813455e+00, -2.441028736296861e+00, 5.485887012742823e-01,
    1.401905293567055e+00, -1.750314406301575e+00, 7.399264893298718e-01, 3.654054875767838e+00,
    -2.357659114423275e+00, -3.309474073038505e+00, 3.175748725344075e+00, -1.600096911697340e+00,
    -6.863268868947208e+00, 2.571860039963515e+00, 3.427491433550636e+00, -6.371017300250693e+00,
    3.819660112500841e-01, 7.249788564229927e+00, -4.974604116583646e+00, -5.931504405874891e+00,
    5.418271906706704e+00, -2.497066016555552e+00, -9.825832024212318e+00, 3.088884871152418e+00,
    4.078891045202960e+00, -6.548167854569027e+00, 1.000000000000024e+00, 7.215251022648846e+00,
    -4.124416704660189e+00, -5.249335442385497e+00, 2.449606536466953e+00, -3.405047016034698e+00,
    -7.381083429884773e+00, 7.048601163530683e-01, 2.065637678554792e+00, -1.547580050992995e+00,
    1.999999999999952e+00, 3.532080897449351e+00, -7.904685398100374e-01, -2.301911644415701e+00,
    -2.261593084236767e+00, -3.575473592887607e+00, -3.127771280864830e+00, -1.451086929951344e+00,
    3.418488247601111e-01, 2.428078861959174e+00, 2.642039521920205e+00, 1.604609970820529e+00,
    1.007031384700326e+00, -7.285856994916259e-01, -2.644418816303922e+00, -2.707106781186549e+00,
    -2.025895768300423e+00, -5.790690406818835e-01, 1.167374324365524e+00, 1.629557737713738e+00,
    2.618033988749890e+00, 3.150203830664179e+00, 3.479997734225964e-01, -9.933202305971012e-01,
    4.970035481532882e-01, -1.255780282667328e+00, -2.523252273262954e+00, 1.278357953181082e+00,
    2.323745840555447e+00, 2.304802973320752e-01, 2.221231742082451e+00, 3.424946517026821e+00,
    4.689618322348209e-01, -1.117822937779668e-01, 7.304123987528524e-01, -1.579369595164835e-01,
    1.128085012493401e-01, 3.873268672159759e-01, 7.451255485834216e-01, 3.211569693249676e+00,
    1.999999999999986e+00, -6.280924248438042e-01, 3.141793793734803e+00, 2.822727063110964e+00,
    -3.913316748476432e+00, -1.579369595164648e-01, 4.987331237372172e+00, -2.828876301122972e+00,
    -2.341359820234657e+00, 8.130486106028723e+00, 2.221231742082481e+00, -4.797274789422697e+00,
    5.526856006212210e+00, 4.571012539001067e+00, -7.981211931904134e+00, -1.255780282667118e+00,
    6.155009917053808e+00, -4.528423680717493e+00, -3.206823018258715e+00, 8.901040762624355e+00,
    2.618033988750149e+00, -4.011707414232134e+00, 4.584903132807465e+00, 2.745493774474633e+00,
    -7.215973194162897e+00, -2.707106781186609e+00, 2.054581945510653e+00, -3.448214287464014e+00,
    -1.508236660692670e+00, 5.314729349274677e+00, 2.642039521920255e+00, -4.904580134230607e-01,
    1.883839503956619e+00, -1.785361103113239e-01, -4.743235794083558e+00, -3.575473592887645e+00,
    -1.536393682238169e+00, -2.480077021189885e+00, -7.058789867569382e-01, 2.986276763839585e+00,
    1.999999999999995e+00, -2.485650379067263e-01, 1.054319727415723e+00, -4.964138253532142e-01,
    -5.162147263877177e+00, -3.405047016034772e+00, -2.179087434381874e-01, -3.494314149885158e+00,
    -2.284287660610369e+00, 4.133610003255131e+00, 1.000000000000017e+00, -3.351394278346055e+00,
    2.093434612846140e+00, 1.108396937972686e+00, -6.649607991913975e+00, -2.497066016555512e+00,
    2.343461531117823e+00, -4.071145490668556e+00, -3.156383729417471e+00, 4.375144869068620e+00,
    3.819660112501549e-01, -3.611385393578468e+00, 1.746534415244799e+00, 9.052471223515746e-01,
    -4.170190988945443e+00, -1.600096911697442e+00, 4.209733552143748e-01, -1.564395278065375e+00,
    -5.535405195386885e-01, 6.124253608540616e-01, 7.399264893299152e-01, 1.615170839239435e+00,
    -8.009178464435953e-01, -1.790566464955358e+00, 1.583495509857728e+00, -1.292893218813281e+00,
    -4.657541607921194e+00, 2.183641551584177e+00, 3.397768839233679e+00, -3.226166531096051e+00,
    1.999999999999853e+00, 7.090923840754289e+00, -2.363528383780275e+00, -3.563895782227414e+00,
    4.482265006604223e+00, -1.568649212747739e+00, -6.738488476499024e+00, 3.330446774341062e+00,
    4.753305035019968e+00, -2.305746259718722e+00, 3.396802246667529e+00, 7.620533159738044e+00,
    -9.663436806906458e-01, -2.463124713360423e+00, 2.384450027212436e+00, -1.939950007892968e+00,
    -4.568550209897076e+00, 1.854173659571886e+00, 3.542919535258467e+00, 1.012706279149933e+00,
    4.000000000000032e+00, 5.478874234573386e+00, 7.134598796339027e-01, -1.088944180658253e+00,
    4.452512685799015e-01, -1.939950007893178e+00, -3.312537346014979e+00, 1.315008216616188e+00,
    3.092198059324607e+00, 5.727479313111625e-01, 3.396802246667435e+00, 5.740733750048110e+00,
    -5.251775283684854e-01, -2.241593627155657e+00, 2.721156698556834e+00, -1.568649212747639e+00,
    -5.726900800780424e+00, 2.921779381855082e+00, 4.253597404311220e+00, -3.748569312253248e+00,
    1.999999999999989e+00, 7.614924634102610e+00, -3.216220271036124e+00, -4.284653530565588e+00,
    5.476317258676247e+00, -1.292893218813215e+00, -7.605618864035197e+00, 3.515079893688291e+00,
    4.081808290128408e+00, -5.528918272274283e+00, 7.399264893301899e-01, 6.065408788896503e+00,
    -3.359151772041982e+00, -3.787143675590877e+00, 3.057008243991181e+00, -1.600096911697161e+00,
    -4.883631898251418e+00, 7.670003376424446e-01, 1.049022344242364e+00, -1.621412578723446e+00,
    3.819660112500098e-01, 8.337979086815931e-01, -5.521335599396140e-01, -1.099759331439174e+00,
    -2.974685651875986e+00, -2.497066016555599e+00, -5.751317162216540e-01, -2.796327443826913e+00,
    -1.907001829208586e+00, 3.190598945535347e+00,
  };

  const std::vector<cuda::std::complex<double>> output = {
    { 2.439599962045054e-03, 0.000000000000000e+00 }, { 7.509041464899204e-03, 0.000000000000000e+00 },
    { -5.480443291605107e-03, 0.000000000000000e+00 } , { 1.437485007113491e-02, 0.000000000000000e+00 },
    { -2.954253938780135e-02, 0.000000000000000e+00 } , { 4.277371937522201e-02, 0.000000000000000e+00 },
    { -9.984572922629760e-02, 0.000000000000000e+00 } , { 3.419510764006435e-01, 0.000000000000000e+00 },
    { 9.537451532434031e-01, 0.000000000000000e+00 } , { 5.957454154386652e-01, 0.000000000000000e+00 },
    { 4.865900798589554e-02, 0.000000000000000e+00 } , { -5.644927000424507e-01, 0.000000000000000e+00 },
    { -9.363002338829249e-01, 0.000000000000000e+00 } , { -9.614168328817083e-01, 0.000000000000000e+00 },
    { -6.164089858173174e-01, 0.000000000000000e+00 } , { -3.204625806200846e-02, 0.000000000000000e+00 },
    { 5.614904757255147e-01, 0.000000000000000e+00 } , { 9.401098311428477e-01, 0.000000000000000e+00 },
    { 9.595548226427686e-01, 0.000000000000000e+00 } , { 6.123990223938300e-01, 0.000000000000000e+00 },
    { 3.145178930109266e-02, 0.000000000000000e+00 } , { -5.612978663977781e-01, 0.000000000000000e+00 },
    { -9.396032513763450e-01, 0.000000000000000e+00 } , { -9.591082228776076e-01, 0.000000000000000e+00 },
    { -6.122623164616379e-01, 0.000000000000000e+00 } , { 1.060303471800111e-03, -1.737529469084180e-03 },
    { 9.334655591286822e-03, -9.108589136042917e-03 } , { -3.889538623759440e-03, 8.331082732569132e-03 },
    { 1.437284545440374e-02, -2.173127992303941e-02 } , { -1.543149543774123e-02, 3.257539117881839e-02 },
    { 3.010436130735872e-02, -5.988861712209756e-02 } , { -4.129536991888487e-02, 1.018984593941769e-01 },
    { 1.726910580040970e-01, -3.218113356492108e-01 } , { 1.082501692805713e+00, -1.002349453380232e+00 },
    { 9.233217607826631e-01, -5.969645247821571e-01 } , { 5.200854412965737e-01, -4.156823096384311e-01 },
    { 4.226170467843104e-01, -2.868650052342631e-01 } , { 6.026808328786841e-01, -4.485444261063845e-01 },
    { 9.876010570861596e-01, -7.128127891923534e-01 } , { 1.206807695195495e+00, -8.786772534206233e-01 },
    { 1.117496704298497e+00, -8.131386978426469e-01 } , { 7.719301436452801e-01, -5.607814245165342e-01 },
    { 4.627978450362391e-01, -3.361343414637152e-01 } , { 4.453889175672469e-01, -3.235498090483362e-01 },
    { 7.340651163551428e-01, -5.334097141035692e-01 } , { 1.090956666723801e+00, -7.927351876615877e-01 },
    { 1.221873843180764e+00, -8.877486582343111e-01 } , { 1.018766764144304e+00, -7.400892128919138e-01 },
    { 6.489750990261092e-01, -4.714509959465011e-01 } , { 4.174162674844585e-01, -3.033010359740317e-01 },
    { -8.622295636628049e-04, -2.447952857675941e-03 } , { -7.103740436157156e-04, -2.323405954326742e-03 },
    { 2.227673172483384e-03, 7.038817005040069e-03 } , { -5.381167527340844e-03, -1.706165329214641e-02 },
    { 1.124961135356509e-02, 3.402157184644701e-02 } , { -2.170840538433798e-02, -6.538046703975291e-02 },
    { 4.519168953530556e-02, 1.325349206994987e-01 } , { -1.385675867544351e-01, -4.073383347698371e-01 },
    { -9.699005511352607e-02, -2.852956589213041e-01 } , { 3.670133044800432e-02, 1.207911700870566e-01 },
    { -2.050484014823475e-02, -6.775480600842432e-02 } , { 1.171244961300200e-02, 3.958022016942286e-02 },
    { -6.795477688989621e-03, -2.238886412032819e-02 } , { 3.311965500699374e-03, 1.134652982182957e-02 },
    { -1.560884617661773e-03, -5.245034771913713e-03 } , { -4.687867674189119e-04, -8.786247636941261e-04 },
    { -7.063324800907350e-05, 3.186464693219508e-05 } , { -4.650123868501471e-05, -4.309064195272798e-07 },
    { 1.928057685417320e-05, -7.330854794236236e-05 } , { -9.311681429907155e-05, -1.445490422533053e-04 },
    { -2.804880093807091e-04, -1.423229211768234e-04 } , { -2.530971875744359e-04, -5.056353023142963e-05 },
    { -5.832220355828718e-06, 5.439778553226764e-05 } , { 1.436929006195354e-04, 8.582086471619614e-05 },
    { 3.284533033508494e-05, 4.562066551766120e-05 } , { -9.366019980262632e-04, -6.061428238030706e-04 },
    { -1.374453353610268e-02, 3.829670490751983e-03 } , { 1.226639943315146e-02, -2.082433404303908e-03 },
    { -2.546473637189456e-02, 3.749139793805725e-03 } , { 1.444320370622049e-02, 4.721129917812395e-03 },
    { -5.855230444689256e-02, 1.015447453202461e-03 } , { 5.766246093742598e-02, 1.833706050271663e-02 },
    { -1.941988227493371e-01, -6.640384962703830e-02 } , { -1.483383337500230e+00, 3.772827867949836e-01 },
    { -6.824332060880891e-01, 2.581500254851655e-01 } , { -4.808899148805546e-01, 1.368846729393098e-01 },
    { -2.380842770584604e+00, 7.842260033902112e-01 } , { -3.745797707549044e+00, 1.211122525467176e+00 },
    { -2.557027149031421e+00, 8.335749997483055e-01 } , { -2.945880375599020e-01, 9.440512089039005e-02 },
    { 1.398341051956118e-01, -4.603200098275473e-02 } , { -7.928016177959440e-01, 2.577503692467635e-01 },
    { -2.829777127067957e-01, 9.209960375093362e-02 } , { 2.116702631024239e+00, -6.877001544094173e-01 },
    { 3.581339957643589e+00, -1.163709988431214e+00 } , { 2.453886738394625e+00, -7.973952926172340e-01 },
    { 8.366814673172082e-01, -2.718287449847542e-01 } , { 1.136726574119233e+00, -3.692143817048440e-01 },
    { 2.184787225979785e+00, -7.097507519157020e-01 } , { 1.231364622076437e+00, -4.000380718018301e-01 },
    { -5.781571514621614e-04, 6.235000847244333e-04 } , { 1.894279166762804e-03, 1.472132586836201e-04 },
    { -3.520197257634733e-03, -1.122068105102792e-03 } , { 6.202999800979268e-03, 3.710197867123933e-03 },
    { -9.264873638183234e-03, -7.749817155768746e-03 } , { 1.387290291006138e-02, 1.625881149276709e-02 },
    { -1.888097525779512e-02, -3.523728780575475e-02 } , { -1.223494319744209e-03, 1.062061047945037e-01 },
    { 8.072768042829212e-02, 3.003596811840945e-02 } , { -2.787025881273571e-02, -2.389732410549679e-02 },
    { 1.254973366262388e-02, 1.571038421468319e-02 } , { -6.400467079767326e-03, -1.049042049372811e-02 },
    { 2.428847836698539e-03, 6.333034580707378e-03 } , { -9.955592069385870e-04, -3.698384861249887e-03 },
    { -1.515668734410834e-04, 1.775667234980907e-03 } , { 3.161995064893119e-04, -2.906898357939427e-04 },
    { 5.481889297179326e-05, 1.933511216802439e-04 } , { 6.283851262491841e-05, 2.121780567405611e-04 },
    { 4.610677867423915e-05, 9.476321469001585e-05 } , { -1.241181374659392e-04, 4.350865397271386e-05 },
    { -2.904588063025229e-04, 1.337991418111028e-04 } , { -2.118425626524423e-04, 2.066359762571723e-04 },
    { 4.199688556960687e-05, 1.292444537694401e-04 } , { 1.528870426327377e-04, 7.779088768934540e-07 },
    { 1.264289238004937e-05, -1.961272699014919e-05 } , { -1.609393145262531e-04, 0.000000000000000e+00 },
    { 3.600808939925731e-04, 0.000000000000000e+00 } , { -6.330863570261479e-04, 0.000000000000000e+00 },
    { 1.505081518928419e-03, 0.000000000000000e+00 } , { -1.520191874645071e-03, 0.000000000000000e+00 },
    { 2.684595381941345e-03, 0.000000000000000e+00 } , { -3.725564560997527e-03, 0.000000000000000e+00 },
    { -3.886241628185837e-03, 0.000000000000000e+00 } , { 1.826439028281668e-02, 0.000000000000000e+00 },
    { -4.374018028448376e-03, 0.000000000000000e+00 } , { 1.134227847935036e-03, 0.000000000000000e+00 },
    { -9.527562053699423e-04, 0.000000000000000e+00 } , { -1.703056838806560e-04, 0.000000000000000e+00 },
    { -5.392913932456000e-05, 0.000000000000000e+00 } , { -3.744618784539133e-04, 0.000000000000000e+00 },
    { -5.624482371402451e-05, 0.000000000000000e+00 } , { 9.670618636988676e-05, 0.000000000000000e+00 },
    { 2.495930728138029e-04, 0.000000000000000e+00 } , { 1.729833742249998e-04, 0.000000000000000e+00 },
    { -9.840289067475361e-05, 0.000000000000000e+00 } , { -1.969844270626186e-04, 0.000000000000000e+00 },
    { 3.024573785437301e-05, 0.000000000000000e+00 } , { 2.677267110881010e-04, 0.000000000000000e+00 },
    { 1.917145514644947e-04, 0.000000000000000e+00 } , { -6.056370011897627e-05, 0.000000000000000e+00 },
    { -5.781571514621614e-04, -6.235000847244333e-04 } , { 1.894279166762804e-03, -1.472132586836201e-04 },
    { -3.520197257634733e-03, 1.122068105102792e-03 } , { 6.202999800979268e-03, -3.710197867123933e-03 },
    { -9.264873638183234e-03, 7.749817155768746e-03 } , { 1.387290291006138e-02, -1.625881149276709e-02 },
    { -1.888097525779512e-02, 3.523728780575475e-02 } , { -1.223494319744209e-03, -1.062061047945037e-01 },
    { 8.072768042829212e-02, -3.003596811840945e-02 } , { -2.787025881273571e-02, 2.389732410549679e-02 },
    { 1.254973366262388e-02, -1.571038421468319e-02 } , { -6.400467079767326e-03, 1.049042049372811e-02 },
    { 2.428847836698539e-03, -6.333034580707378e-03 } , { -9.955592069385870e-04, 3.698384861249887e-03 },
    { -1.515668734410834e-04, -1.775667234980907e-03 } , { 3.161995064893119e-04, 2.906898357939427e-04 },
    { 5.481889297179326e-05, -1.933511216802439e-04 } , { 6.283851262491841e-05, -2.121780567405611e-04 },
    { 4.610677867423915e-05, -9.476321469001585e-05 } , { -1.241181374659392e-04, -4.350865397271386e-05 },
    { -2.904588063025229e-04, -1.337991418111028e-04 } , { -2.118425626524423e-04, -2.066359762571723e-04 },
    { 4.199688556960687e-05, -1.292444537694401e-04 } , { 1.528870426327377e-04, -7.779088768934540e-07 },
    { 1.264289238004937e-05, 1.961272699014919e-05 } , { -9.366019980262632e-04, 6.061428238030706e-04 },
    { -1.374453353610268e-02, -3.829670490751983e-03 } , { 1.226639943315146e-02, 2.082433404303908e-03 },
    { -2.546473637189456e-02, -3.749139793805725e-03 } , { 1.444320370622049e-02, -4.721129917812395e-03 },
    { -5.855230444689256e-02, -1.015447453202461e-03 } , { 5.766246093742598e-02, -1.833706050271663e-02 },
    { -1.941988227493371e-01, 6.640384962703830e-02 } , { -1.483383337500230e+00, -3.772827867949836e-01 },
    { -6.824332060880891e-01, -2.581500254851655e-01 } , { -4.808899148805546e-01, -1.368846729393098e-01 },
    { -2.380842770584604e+00, -7.842260033902112e-01 } , { -3.745797707549044e+00, -1.211122525467176e+00 },
    { -2.557027149031421e+00, -8.335749997483055e-01 } , { -2.945880375599020e-01, -9.440512089039005e-02 },
    { 1.398341051956118e-01, 4.603200098275473e-02 } , { -7.928016177959440e-01, -2.577503692467635e-01 },
    { -2.829777127067957e-01, -9.209960375093362e-02 } , { 2.116702631024239e+00, 6.877001544094173e-01 },
    { 3.581339957643589e+00, 1.163709988431214e+00 } , { 2.453886738394625e+00, 7.973952926172340e-01 },
    { 8.366814673172082e-01, 2.718287449847542e-01 } , { 1.136726574119233e+00, 3.692143817048440e-01 },
    { 2.184787225979785e+00, 7.097507519157020e-01 } , { 1.231364622076437e+00, 4.000380718018301e-01 },
    { -8.622295636628049e-04, 2.447952857675941e-03 } , { -7.103740436157156e-04, 2.323405954326742e-03 },
    { 2.227673172483384e-03, -7.038817005040069e-03 } , { -5.381167527340844e-03, 1.706165329214641e-02 },
    { 1.124961135356509e-02, -3.402157184644701e-02 } , { -2.170840538433798e-02, 6.538046703975291e-02 },
    { 4.519168953530556e-02, -1.325349206994987e-01 } , { -1.385675867544351e-01, 4.073383347698371e-01 },
    { -9.699005511352607e-02, 2.852956589213041e-01 } , { 3.670133044800432e-02, -1.207911700870566e-01 },
    { -2.050484014823475e-02, 6.775480600842432e-02 } , { 1.171244961300200e-02, -3.958022016942286e-02 },
    { -6.795477688989621e-03, 2.238886412032819e-02 } , { 3.311965500699374e-03, -1.134652982182957e-02 },
    { -1.560884617661773e-03, 5.245034771913713e-03 } , { -4.687867674189119e-04, 8.786247636941261e-04 },
    { -7.063324800907350e-05, -3.186464693219508e-05 } , { -4.650123868501471e-05, 4.309064195272798e-07 },
    { 1.928057685417320e-05, 7.330854794236236e-05 } , { -9.311681429907155e-05, 1.445490422533053e-04 },
    { -2.804880093807091e-04, 1.423229211768234e-04 } , { -2.530971875744359e-04, 5.056353023142963e-05 },
    { -5.832220355828718e-06, -5.439778553226764e-05 } , { 1.436929006195354e-04, -8.582086471619614e-05 },
    { 3.284533033508494e-05, -4.562066551766120e-05 } , { 1.060303471800111e-03, 1.737529469084180e-03 },
    { 9.334655591286822e-03, 9.108589136042917e-03 } , { -3.889538623759440e-03, -8.331082732569132e-03 },
    { 1.437284545440374e-02, 2.173127992303941e-02 } , { -1.543149543774123e-02, -3.257539117881839e-02 },
    { 3.010436130735872e-02, 5.988861712209756e-02 } , { -4.129536991888487e-02, -1.018984593941769e-01 },
    { 1.726910580040970e-01, 3.218113356492108e-01 } , { 1.082501692805713e+00, 1.002349453380232e+00 },
    { 9.233217607826631e-01, 5.969645247821571e-01 } , { 5.200854412965737e-01, 4.156823096384311e-01 },
    { 4.226170467843104e-01, 2.868650052342631e-01 } , { 6.026808328786841e-01, 4.485444261063845e-01 },
    { 9.876010570861596e-01, 7.128127891923534e-01 } , { 1.206807695195495e+00, 8.786772534206233e-01 },
    { 1.117496704298497e+00, 8.131386978426469e-01 } , { 7.719301436452801e-01, 5.607814245165342e-01 },
    { 4.627978450362391e-01, 3.361343414637152e-01 } , { 4.453889175672469e-01, 3.235498090483362e-01 },
    { 7.340651163551428e-01, 5.334097141035692e-01 } , { 1.090956666723801e+00, 7.927351876615877e-01 },
    { 1.221873843180764e+00, 8.877486582343111e-01 } , { 1.018766764144304e+00, 7.400892128919138e-01 },
    { 6.489750990261092e-01, 4.714509959465011e-01 } , { 4.174162674844585e-01, 3.033010359740317e-01 },
  };

  const index_t filter_len = static_cast<index_t>(filter.size());
  const index_t signal_len = static_cast<index_t>(input.size());
  const index_t num_channels = 10;

  auto a = make_tensor<TestType>({signal_len});
  auto f = make_tensor<TestType>({filter_len});
  const index_t b_elem_per_channel = signal_len / num_channels;
  auto b = make_tensor<ComplexType>({ b_elem_per_channel, num_channels });

  cudaMemcpyAsync(a.Data(), input.data(), signal_len * sizeof(TestType), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(f.Data(), filter.data(), filter_len * sizeof(TestType), cudaMemcpyHostToDevice, stream);

  this->exec.sync();

  const index_t decimation_factor = num_channels;
  (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

  this->exec.sync();

  for (auto chan = 0; chan < num_channels; chan++) {
    for (auto k = 0; k < b_elem_per_channel; k++) {
      ComplexType gold { output[chan*b_elem_per_channel+k] };      
      ComplexType test { b(k,chan) };
      ASSERT_NEAR(test.real(), gold.real(), this->thresh);
      ASSERT_NEAR(test.imag(), gold.imag(), this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}

// Oversampled channelizer test based on Harris 2003 receiver_40z.m.
// M=40, D=28, 600-tap filter designed with remez. Golden input and output
// are from the reference implementation. This test validates the oversampled
// phase rotation convention.
TYPED_TEST(ChannelizePolyTestDoubleType, Harris2003Oversampled)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = cuda::std::complex<TestType>;

  const index_t input_len = 5600;
  const index_t M = 40;
  const index_t D = 28;
  const index_t num_output = (input_len + D - 1) / D;

  this->pb->template InitAndRunTVGenerator<ComplexType>(
    "00_transforms", "harris2003_oversampled_operators", "channelize",
    {input_len});

  auto a = make_tensor<ComplexType>({input_len});
  auto f = make_tensor<TestType>({600});
  auto b = make_tensor<ComplexType>({num_output, M});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(f, "filter");
  (b = channelize_poly(a, f, M, D)).run(this->exec);

  this->exec.sync();
  MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_golden", this->thresh);

  MATX_EXIT_HANDLER();
}

// Oversampled identity filter test: validates the causal commutator behavior.
// With a single-tap identity filter (E[r,0] = 1 for all r), the filtering step
// simply selects one input sample per branch. The causal commutator determines
// which sample each branch sees at each output time:
//   - Sample k goes to branch (M-1 - k%M)
//   - At output time n, only samples x[0..n*D+D-1] have arrived
//   - Branch r's newest sample: largest k <= n*D+D-1 where k%M == M-1-r
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, OversampledIdentityFilter)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using InnerType = typename test_types::inner_type<TestType>::type;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t num_channels;
    index_t decimation_factor;
  } test_cases[] = {
    { 20, 4, 2 },   // 2x oversampled, M=4
    { 20, 4, 1 },   // 4x oversampled, M=4
    { 30, 10, 3 },  // rational oversample, M=10 D=3
    { 20, 4, 3 },   // rational oversample, M=4 D=3
    { 20, 2, 1 },   // 2x oversampled, M=2
    {  2, 4, 4 },   // maximally decimated, a_len < M
    {  3, 4, 4 },   // maximally decimated, a_len < M
    {  1, 4, 2 },   // oversampled, a_len < M
    {  3, 4, 2 },   // oversampled, a_len < M
    {  2, 6, 3 },   // oversampled rational, a_len < M
  };

  for (size_t tc = 0; tc < sizeof(test_cases)/sizeof(test_cases[0]); tc++) {
    const index_t a_len = test_cases[tc].a_len;
    const index_t M = test_cases[tc].num_channels;
    const index_t D = test_cases[tc].decimation_factor;
    const index_t f_len = M;  // identity filter: one tap per phase
    const index_t num_output = (a_len + D - 1) / D;

    // Use the pybind generator only for random input data and not for the expected output
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled",
      {a_len, f_len, M, D});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({num_output, M});

    this->pb->NumpyToTensorView(a, "a");

    // Identity filter: h = [1, 1, ..., 1] of length M
    // Polyphase decomposition gives E[r, 0] = 1 for all r (single tap per phase)
    for (index_t i = 0; i < M; i++) { f(i) = static_cast<TestType>(1.0); }

    this->exec.sync();

    (b = channelize_poly(a, f, M, D)).run(this->exec);

    this->exec.sync();

    // Compute expected output using the causal commutator model.
    // With identity filter, g_r[n] = x[newest_r] if branch r has a sample, else 0.
    // Then Y[n, :] = M * IFFT(g[n, :])
    for (index_t n = 0; n < num_output; n++) {
      const index_t last_arrived = n * D + D - 1;

      // Compute the IDFT (with positive exponential) of the
      // causal filtered values. This is M * IFFT.
      for (index_t chan = 0; chan < M; chan++) {
        ComplexType accum { 0 };
        for (index_t r = 0; r < M; r++) {
          // Branch remap for Harris convention input mapping (D < M only)
          const index_t r_remapped = (D < M) ? ((r + M - D) % M) : r;
          const index_t s = M - 1 - r_remapped;  // remapped for input access

          // Causal commutator: find newest sample for branch r
          TestType sample_val = static_cast<TestType>(0.0);
          if (last_arrived >= s) {
            const index_t newest = last_arrived - ((last_arrived - s) % M);
            if (newest < a_len) {
              sample_val = a(newest);
            }
          }

          // IDFT with positive exponential (Harris convention)
          const double arg = 2.0 * M_PI * static_cast<double>(r) * static_cast<double>(chan) / static_cast<double>(M);
          double sinx, cosx;
          sincos(arg, &sinx, &cosx);
          ComplexType twiddle { static_cast<InnerType>(cosx), static_cast<InnerType>(sinx) };
          accum += sample_val * twiddle;
        }

        ComplexType bval { b(n, chan) };
        ASSERT_NEAR(accum.real(), bval.real(), this->thresh)
          << "mismatch at n=" << n << " chan=" << chan
          << " M=" << M << " D=" << D << " a_len=" << a_len;
        ASSERT_NEAR(accum.imag(), bval.imag(), this->thresh)
          << "mismatch at n=" << n << " chan=" << chan
          << " M=" << M << " D=" << D << " a_len=" << a_len;
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// Oversampled channelizer tests: integer oversampling (D divides M evenly)
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, OversampledInteger)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    index_t decimation_factor;
  } test_cases[] = {
    // 2x oversampling (D = M/2)
    { 2500, 170, 10, 5 },
    { 2500, 187, 10, 5 },
    { 1800, 120, 8, 4 },
    { 37193, 41*8+4, 8, 4 },
    // 5x oversampling (D = M/5)
    { 2500, 170, 10, 2 },
    // Large channel count
    { 35000, 5*180, 180, 90 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = test_cases[i].decimation_factor;
    const index_t b_len_per_channel = (a_len + decimation_factor - 1) / decimation_factor;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled",
      {a_len, f_len, num_channels, decimation_factor});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// Oversampled channelizer tests: rational oversampling (D does not divide M evenly)
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, OversampledRational)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    index_t decimation_factor;
  } test_cases[] = {
    // M=10, D=3 (rational 10/3 oversampling)
    { 2500, 170, 10, 3 },
    { 2500, 187, 10, 3 },
    // M=10, D=7
    { 2500, 170, 10, 7 },
    // M=9, D=4
    { 1800, 120, 9, 4 },
    // M=8, D=3
    { 37193, 41*8+4, 8, 3 },
    // M=11, D=4
    { 2500, 187, 11, 4 },
    // Large channel count, rational
    { 35000, 5*181+17, 181, 60 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = test_cases[i].decimation_factor;
    const index_t b_len_per_channel = (a_len + decimation_factor - 1) / decimation_factor;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled",
      {a_len, f_len, num_channels, decimation_factor});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// Oversampled channelizer with batched inputs
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, OversampledBatched)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  constexpr int BATCH_DIMS = 2;
  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    index_t decimation_factor;
    cuda::std::array<index_t, BATCH_DIMS> batch_dims;
  } test_cases[] = {
    // Integer oversampling, batched
    { 2500, 170, 10, 5, { 2, 3 } },
    // Rational oversampling, batched
    { 2500, 170, 10, 3, { 2, 3 } },
    { 37193, 41*8+4, 8, 3, { 3, 1 } },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = test_cases[i].decimation_factor;
    const index_t b_len_per_channel = (a_len + decimation_factor - 1) / decimation_factor;
    std::vector<index_t> sizes = { a_len, f_len, num_channels, decimation_factor };
    sizes.insert(sizes.end(), test_cases[i].batch_dims.begin(), test_cases[i].batch_dims.end());
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled", sizes);
    const cuda::std::array<index_t, BATCH_DIMS+1> a_dims = {
      test_cases[i].batch_dims[0], test_cases[i].batch_dims[1], a_len
    };
    const cuda::std::array<index_t, BATCH_DIMS+2> b_dims = {
      test_cases[i].batch_dims[0], test_cases[i].batch_dims[1], b_len_per_channel, num_channels
    };
    auto a = make_tensor<TestType>(a_dims);
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>(b_dims);

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// SmemTiled kernel tests: maximally decimated with M > 256 (forces SmemTiled
// because the existing _Smem kernel requires M <= 256)
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, SmemTiledMaximallyDecimated)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
  } test_cases[] = {
    // M=512, P=10 — exercises MaximallyDecimated SmemTiled with multiple channel tiles
    { 256000, 10*512, 512 },
    // M=512, unaligned filter (not a multiple of M)
    { 256000, 10*512+37, 512 },
    // M=1024, P=4 — large channel count, small P
    { 102400, 4*1024, 1024 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = num_channels;
    const index_t b_len_per_channel = (a_len + num_channels - 1) / num_channels;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize", {a_len, f_len, num_channels, num_channels});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// SmemTiled kernel tests: large-M oversampled cases that exercise multi-tile
// channel tiling in the oversampled path
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, SmemTiledOversampledLargeM)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    index_t decimation_factor;
  } test_cases[] = {
    // M=1024, D=512, 2x integer oversampling, K=2
    { 256000, 8*1024, 1024, 512 },
    // M=512, D=256, 2x integer oversampling, K=2
    { 128000, 10*512, 512, 256 },
    // M=512, D=256, unaligned filter
    { 128000, 10*512+19, 512, 256 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = test_cases[i].decimation_factor;
    const index_t b_len_per_channel = (a_len + decimation_factor - 1) / decimation_factor;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled",
      {a_len, f_len, num_channels, decimation_factor});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// SmemTiled kernel tests: rational oversampling with large K (many distinct
// filter phases) to stress the K-phase filter storage
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, SmemTiledOversampledLargeK)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    index_t decimation_factor;
  } test_cases[] = {
    // M=128, D=97, K=128 (gcd=1, all phases distinct)
    { 25600, 5*128, 128, 97 },
    // M=64, D=37, K=64 (gcd=1)
    { 12800, 8*64, 64, 37 },
    // M=100, D=30, K=10 (gcd=10), moderate K
    { 20000, 7*100, 100, 30 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = test_cases[i].decimation_factor;
    const index_t b_len_per_channel = (a_len + decimation_factor - 1) / decimation_factor;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled",
      {a_len, f_len, num_channels, decimation_factor});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// SmemTiled oversampled with FilterInSmem=true.
// Requires CTILE * K * P * sizeof(filter_t) <= 2048.
// With K=2 (2x integer oversample) and small P, the filter fits in smem.
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, SmemTiledOversampledFilterInSmem)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    index_t decimation_factor;
  } test_cases[] = {
    // M=1024, D=512, K=2, P=4: filter_smem = 64*2*4*sizeof(filter_t) = 2048 bytes (float)
    { 102400, 4*1024, 1024, 512 },
    // M=512, D=256, K=2, P=3: filter_smem = 64*2*3*sizeof(filter_t) = 1536 bytes (float)
    { 51200, 3*512, 512, 256 },
    // M=1024, D=512, K=2, P=2: filter_smem = 64*2*2*sizeof(filter_t) = 1024 bytes (float)
    { 102400, 2*1024, 1024, 512 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = test_cases[i].decimation_factor;
    const index_t b_len_per_channel = (a_len + decimation_factor - 1) / decimation_factor;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled",
      {a_len, f_len, num_channels, decimation_factor});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// Complex filter tests to exercise the complex×complex channelize_cmac path.
// Previous tests all use real filters.
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, ComplexFilter)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    index_t decimation_factor;
  } test_cases[] = {
    // Complex filter, D==M, small M (FusedChan path)
    { 1800, 120, 4, 4 },
    // Complex filter, D==M, medium M (_Smem path)
    { 2500, 170, 10, 10 },
    // Complex filter, D==M, large M (SmemTiled path)
    { 128000, 10*512, 512, 512 },
    // Complex filter, oversampled (SmemTiled oversampled path)
    { 2500, 170, 10, 5 },
    // Complex filter, oversampled, large M
    { 128000, 10*512, 512, 256 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = test_cases[i].decimation_factor;
    const index_t b_len_per_channel = (a_len + decimation_factor - 1) / decimation_factor;

    // Use complex input AND complex filter
    this->pb->template InitAndRunTVGenerator<ComplexType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled",
      {a_len, f_len, num_channels, decimation_factor});

    auto a = make_tensor<ComplexType>({a_len});
    auto f = make_tensor<ComplexType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// Oversampled with very large P so SmemTiled input buffer exceeds 48KB,
// falling back to the generic ChannelizePoly1D (MaximallyDecimated=false) path.
// For complex<float> input: (P+3) * 64 * 8 > 49152 → P > 93.
TYPED_TEST(ChannelizePolyTestNonHalfFloatTypes, GenericOversampledFallback)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = typename test_types::complex_type<TestType>::type;

  struct {
    index_t a_len;
    index_t f_len;
    index_t num_channels;
    index_t decimation_factor;
  } test_cases[] = {
    // M=8, D=4, P=100: input_smem = 103*64*8 = 52736 > 49152 for complex<float>
    { 8000, 100*8, 8, 4 },
    // M=10, D=5, P=200: clearly exceeds 48KB for any type
    { 20000, 200*10, 10, 5 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t num_channels = test_cases[i].num_channels;
    const index_t decimation_factor = test_cases[i].decimation_factor;
    const index_t b_len_per_channel = (a_len + decimation_factor - 1) / decimation_factor;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "channelize_poly_operators", "channelize_oversampled",
      {a_len, f_len, num_channels, decimation_factor});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<ComplexType>({b_len_per_channel, num_channels});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = channelize_poly(a, f, num_channels, decimation_factor)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}
