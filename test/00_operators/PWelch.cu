////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2023, NVIDIA Corporation
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

#include <pybind11/pybind11.h>
using namespace pybind11::literals; // to bring in the '_a' literal

using namespace matx;

struct TestParams {
    std::string window_name;
    index_t signal_size;
    index_t nperseg;
    index_t noverlap;
    index_t nfft;
    float ftone;
    float sigma;
};

const std::vector<TestParams> CONFIGS = {
  {"none", 8, 8, 2, 8, 0., 0.},
  {"none", 16, 8, 4, 8, 1., 0.},
  {"none", 16, 8, 4, 8, 2., 1.},
  {"none", 16384, 256, 64, 256, 63., 0.},
  {"boxcar", 8, 8, 2, 8, 0., 0.},
  {"hann", 16, 8, 4, 8, 1., 0.},
  {"flattop", 1024, 64, 32, 128, 2., 1.},
};

class PWelchComplexExponentialTest : public ::testing::TestWithParam<TestParams>
{
public:
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();

    if (params.nfft > 8) thresh = 1.f; // use higher threshold for larger fft sizes
  }

  void TearDown() override { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.01f;
  TestParams params = ::testing::TestWithParam<TestParams>::GetParam();
};

template <typename TypeParam>
void helper(PWelchComplexExponentialTest& test)
{
  MATX_ENTER_HANDLER();
  pybind11::dict cfg(
    "signal_size"_a=test.params.signal_size,
    "nperseg"_a=test.params.nperseg,
    "noverlap"_a=test.params.noverlap,
    "nfft"_a=test.params.nfft,
    "ftone"_a=test.params.ftone,
    "sigma"_a=test.params.sigma,
    "window_name"_a=test.params.window_name
  );

  test.pb->template InitAndRunTVGeneratorWithCfg<TypeParam>(
      "00_operators", "pwelch_operators", "pwelch_complex_exponential", cfg);

  tensor_t<TypeParam, 1> x{{test.params.signal_size}};
  test.pb->NumpyToTensorView(x, "x_in");

  auto Pxx  = make_tensor<typename TypeParam::value_type>({test.params.nfft});

  cudaExecutor exec{};

  if (test.params.window_name == "none")
  {
    (Pxx = pwelch(x, test.params.nperseg, test.params.noverlap, test.params.nfft)).run(exec);
  }
  else
  {
    auto w = make_tensor<typename TypeParam::value_type>({test.params.nperseg});
    if (test.params.window_name == "boxcar")
    {
      (w = ones<typename TypeParam::value_type>({test.params.nperseg})).run(exec);
    }
    else if (test.params.window_name == "hann")
    {
      (w = hanning<0,1,typename TypeParam::value_type>({test.params.nperseg})).run(exec);
    }
    else if (test.params.window_name == "flattop")
    {
      (w = flattop<0,1,typename TypeParam::value_type>({test.params.nperseg})).run(exec);
    }
    else
    {
      ASSERT_TRUE(false) << "Unknown window parameter name " + test.params.window_name;
    }
    (Pxx = pwelch(x, w, test.params.nperseg, test.params.noverlap, test.params.nfft)).run(exec);
  }

  exec.sync();

  MATX_TEST_ASSERT_COMPARE(test.pb, Pxx, "Pxx_out", test.thresh);
  MATX_EXIT_HANDLER();
}


TEST_P(PWelchComplexExponentialTest, xin_complex_float)
{
  helper<cuda::std::complex<float>>(*this);
}

TEST_P(PWelchComplexExponentialTest, xin_complex_double)
{
  helper<cuda::std::complex<double>>(*this);
}

INSTANTIATE_TEST_SUITE_P(PWelchComplexExponentialTests, PWelchComplexExponentialTest,::testing::ValuesIn(CONFIGS));


TEST(PWelchOpTest, xin_complex_float)
{
  float thresh = 0.01f;
  index_t signal_size = 16;
  index_t nperseg = 8;
  index_t noverlap = 0;
  index_t nfft = 8;
  auto x = ones<cuda::std::complex<float>>({signal_size});
  cudaExecutor exec{};
  // example-begin pwelch-test-1
  auto Pxx  = make_tensor<float>({nfft});
  auto w = ones<float>({nperseg});
  (Pxx = pwelch(x, w, nperseg, noverlap, nfft)).run(exec);
  // example-end pwelch-test-1

  exec.sync();

  EXPECT_NEAR(Pxx(0), 64, thresh);
  for (index_t k=1; k<nfft; k++)
  {
    EXPECT_NEAR(Pxx(k), 0, thresh) << "failure at index k=" << k;
  }
}
