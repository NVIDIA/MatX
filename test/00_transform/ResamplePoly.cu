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

template <typename T>
class ResamplePolyTest : public ::testing::Test {
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
      thresh = 1.0e-10;
    } else {
      // Revisit this tolerance. We should likely use a relative tolerance
      // rather than absolute for larger values.
      thresh = 1.0e-1;
    }
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  double thresh;;
};

template <typename TensorType>
class ResamplePolyTestNonHalfFloatTypes
    : public ResamplePolyTest<TensorType> {
};

template <typename TensorType>
class ResamplePolyTestFloatTypes
    : public ResamplePolyTest<TensorType> {
};

TYPED_TEST_SUITE(ResamplePolyTestNonHalfFloatTypes, MatXFloatNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(ResamplePolyTestFloatTypes, MatXFloatTypesCUDAExec);

// SimpleOddLength tests use random input and filter values and
// odd-length filters.
TYPED_TEST(ResamplePolyTestNonHalfFloatTypes, SimpleOddLength)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t f_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    // Filter longer than the input signal (pre-upsampling)
    { 3500, 62501, 384, 3125 },
    { 3500, 62501, 7*384, 3125 },
    { 3501, 62501, 384, 3125 },
    // Filter shorter than upsampling factor
    { 137103, 137, 384, 3125 },
    { 137104, 137, 384, 3125 },    
    // Upsampling only (down=1)
    { 7173, 173, 7, 1 },
    // Downsampling only (up=1)
    { 7173, 173, 1, 7 },
    { 51735, 151, 1, 16 },
    // Single sample input
    { 1, 3, 3, 1}
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    const index_t up_len = a_len * up;
    [[maybe_unused]] const index_t b_len = up_len / down + ((up_len % down) ? 1 : 0);
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, f_len, up, down});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<TestType>({b_len});
    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    // example-begin resample_poly-test-1
    // Resample "a" using input signal "f" by rate up/down
    (b = resample_poly(a, f, up, down)).run(this->exec);
    // example-end resample_poly-test-1

    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);    

    // Now test with a multiplicative operator on the input. The resampler is linear,
    // so we can inverse-scale the output to compare against the golden outputs.
    (b = resample_poly(static_cast<TestType>(4.0) * a, f, up, down)).run(this->exec);
    (b = b * static_cast<TestType>(0.25)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// SimpleEvenLength tests use random input and filter values and
// even-length filters.
TYPED_TEST(ResamplePolyTestNonHalfFloatTypes, SimpleEvenLength)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t f_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    // Filter longer than the input signal (pre-upsampling)
    { 3500, 62500, 384, 3125 },
    { 3500, 62500, 7*384, 3125 },
    { 3501, 62500, 384, 3125 },
    // Filter shorter than upsampling factor
    { 137103, 138, 384, 3125 },
    { 137104, 138, 384, 3125 },
    // Upsampling only (down=1)
    { 7173, 174, 7, 1 },
    // Downsampling only (up=1)
    { 7173, 174, 1, 7 },
    { 51735, 150, 1, 16 },
    // Single sample input
    { 1, 2, 3, 1}
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    const index_t up_len = a_len * up;
    [[maybe_unused]] const index_t b_len = up_len / down + ((up_len % down) ? 1 : 0);
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, f_len, up, down});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<TestType>({b_len});
    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = resample_poly(a, f, up, down)).run(this->exec);

    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);    

    // Now test with a multiplicative operator on the input. The resampler is linear,
    // so we can inverse-scale the output to compare against the golden outputs.
    (b = resample_poly(static_cast<TestType>(4.0) * a, f, up, down)).run(this->exec);
    (b = b * static_cast<TestType>(0.25)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// DefaultFilter tests use the default scipy filter
TYPED_TEST(ResamplePolyTestNonHalfFloatTypes, DefaultFilter)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    { 3500, 384, 3125 },
    { 3500, 7*384, 3125 },
    { 3501, 384, 3125 },
    { 137104, 384, 3125 },
    { 7173, 7, 1 },
    { 7173, 1, 7 },
    { 35000, 1, 16 },
    { 35001, 1, 16 },
    { 15, 1, 16 }
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    const index_t f_len = 2 * 10 * std::max(up, down) + 1;
    const index_t up_len = a_len * up;
    [[maybe_unused]] const index_t b_len = up_len / down + ((up_len % down) ? 1 : 0);
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, 1, up, down});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<TestType>({b_len});
    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_default");

    this->exec.sync();

    (b = resample_poly(a, f, up, down)).run(this->exec);

    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_default", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

// DefaultFilter tests use the default scipy filter. For half types, we need short
// vectors to use reasonable thresholds. The numpy implementation will still be
// float32, so we are comparing a MatX fp16/bf16 result to a numpy fp32 result.
TYPED_TEST(ResamplePolyTestFloatTypes, DefaultFilter)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    { 150, 5, 3 },
    { 151, 5, 3 },
    { 350, 7, 1 },
    { 350, 1, 7 },
    { 351, 7, 1 },
    { 351, 1, 7 },
    { 1000000, 5, 1 },
    { 1000000, 1, 5 },
    { 1000000, 2, 3 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    const index_t f_len = 2 * 10 * std::max(up, down) + 1;
    const index_t up_len = a_len * up;
    [[maybe_unused]] const index_t b_len = up_len / down + ((up_len % down) ? 1 : 0);
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, 1, up, down});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<TestType>({b_len});
    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_default");

    this->exec.sync();

    (b = resample_poly(a, f, up, down)).run(this->exec);

    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_default", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ResamplePolyTestNonHalfFloatTypes, Batched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t f_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    { 3500, 62501, 384, 3125 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    const index_t up_len = a_len * up;
    [[maybe_unused]] const index_t b_len = up_len / down + ((up_len % down) ? 1 : 0);
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, f_len, up, down});
   
    const int nA = 16;
    const int nB = 37;
    const int nC = 55;

    auto a = make_tensor<TestType>({a_len});
    auto ac = matx::clone<4>(a, {nA, nB, nC, matx::matxKeepDim});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<TestType>({nA, nB, nC, b_len});
    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");
    (b = resample_poly(ac, f, up, down)).run(this->exec);

    this->exec.sync();

    // Verify that the 4D tensor was handled in a batched fashion
    for (int ia = 0; ia < nA; ia++) {
      for (int ib = 0; ib < nB; ib++) {
        for (int ic = 0; ic < nC; ic++) {
          auto bslice = matx::slice<1>(b, {ia, ib, ic, 0}, {matx::matxDropDim, matx::matxDropDim, matx::matxDropDim, matx::matxEnd});
          MATX_TEST_ASSERT_COMPARE(this->pb, bslice, "b_random", this->thresh);
        }
      }
    }

    // Now use a full 4D tensor rather than just a cloned tensor as input
    auto full = make_tensor<TestType>({nA, nB, nC, a_len});
    (full = ac).run(this->exec);
    (b = 0).run(this->exec);

    this->exec.sync();

    (b = resample_poly(ac, f, up, down)).run(this->exec);

    this->exec.sync();

    // Verify that the 4D tensor was handled in a batched fashion
    for (int ia = 0; ia < nA; ia++) {
      for (int ib = 0; ib < nB; ib++) {
        for (int ic = 0; ic < nC; ic++) {
          auto bslice = matx::slice<1>(b, {ia, ib, ic, 0}, {matx::matxDropDim, matx::matxDropDim, matx::matxDropDim, matx::matxEnd});
          MATX_TEST_ASSERT_COMPARE(this->pb, bslice, "b_random", this->thresh);
        }
      }
    }

    this->exec.sync();
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ResamplePolyTestNonHalfFloatTypes, Identity)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    { 3500, 1, 1 },
    { 3501, 7, 7 }
  };

  auto zero = make_tensor<TestType>({1});
  (zero = 0).run(this->exec);
  this->exec.sync();

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, 1, up, down});

    auto a = make_tensor<TestType>({a_len});
    auto b = make_tensor<TestType>({a_len});
    this->pb->NumpyToTensorView(a, "a");
    (b = resample_poly(a, zero, up, down)).run(this->exec);

    this->exec.sync();

    // The output should equal the input because up == down.
    for (index_t k = 0; k < a_len; k++) {
      TestType ak { a(k) };
      TestType bk { b(k) };
      ASSERT_EQ(ak, bk);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ResamplePolyTestNonHalfFloatTypes, Downsample)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    { 3500, 1, 2 },
    { 21003, 1, 3 }
  };

  auto seven = make_tensor<TestType>({1});
  (seven = 7).run(this->exec);
  this->exec.sync();

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, 1, up, down});

    auto a = make_tensor<TestType>({a_len});
    const index_t b_len = a_len / down;
    auto b = make_tensor<TestType>({b_len});
    this->pb->NumpyToTensorView(a, "a");
    (b = resample_poly(a, seven, up, down)).run(this->exec);

    this->exec.sync();

    for (index_t j = 0; j < b_len; j++) {
      double aj, bj;
      if constexpr (is_complex_v<TestType>) {
        aj = cuda::std::abs(7 * a(j*down));
        bj = cuda::std::abs(b(j));
      } else {
        aj = 7 * a(j*down);
        bj = b(j);
      }
      ASSERT_NEAR(aj, bj, this->thresh);
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ResamplePolyTestNonHalfFloatTypes, Upsample)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    { 3500, 2, 1 },
    { 21003, 3, 1 }
  };

  auto f = make_tensor<TestType>({1});

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, 1, up, down});

    // The resample kernel scales the filter by up, so we use 1/up to get an
    // effective filter of 1.
    (f = 1.0/static_cast<double>(up)).run(this->exec);
    this->exec.sync();

    auto a = make_tensor<TestType>({a_len});
    const index_t b_len = a_len * up;
    auto b = make_tensor<TestType>({b_len});
    this->pb->NumpyToTensorView(a, "a");
    (b = resample_poly(a, f, up, down)).run(this->exec);

    this->exec.sync();

    // Since the filter is single tapped and == 1, we should get the sequence
    // [a_0, 0, ..., 0, a_1, 0, ...] with up-1 zeros between successive values
    // from a. The kernel scales the filter by up, so it was set to 1/up above.
    for (index_t j = 0; j < b_len; j++) {
      double aj, bj;
      if constexpr (is_complex_v<TestType>) {
        aj = (j % up == 0) ? cuda::std::abs(a(j/up)) : 0;
        bj = cuda::std::abs(b(j));
      } else {
        aj = (j % up == 0) ? a(j/up) : 0;
        bj = b(j);
      }
      if (j % up == 0) {
        ASSERT_NEAR(aj, bj, this->thresh);
      } else {
        ASSERT_EQ(bj, 0.0);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// Use non-trivial operators for input and filter tensors to ensure
// that the kernel supports such operators.
TYPED_TEST(ResamplePolyTestNonHalfFloatTypes, Operators)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  struct {
    index_t a_len;
    index_t f_len;
    index_t up;
    index_t down;
  } test_cases[] = {
    { 3500, 62501, 384, 3125 },
    { 3501, 62501, 384, 3125 },
    { 3500, 62500, 384, 3125 },
    { 3501, 62500, 384, 3125 },
  };

  for (size_t i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
    const index_t a_len = test_cases[i].a_len;
    const index_t f_len = test_cases[i].f_len;
    const index_t up = test_cases[i].up;
    const index_t down = test_cases[i].down;
    const index_t up_len = a_len * up;
    [[maybe_unused]] const index_t b_len = up_len / down + ((up_len % down) ? 1 : 0);
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "resample_poly_operators", "resample", {a_len, f_len, up, down});

    auto a = make_tensor<TestType>({a_len});
    auto f = make_tensor<TestType>({f_len});
    auto b = make_tensor<TestType>({b_len});
    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(f, "filter_random");

    this->exec.sync();

    (b = resample_poly(shift<0>(shift<0>(a, 8), -8), shift<0>(shift<0>(f, 3), -3), up, down)).run(this->exec);

    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, b, "b_random", this->thresh);
  }

  MATX_EXIT_HANDLER();
}