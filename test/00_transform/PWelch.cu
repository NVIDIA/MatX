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

using namespace matx;

struct TestParams {
    index_t signal_size;
    index_t nperseg;
    index_t noverlap;
    index_t nfft;
    int ftone;
    int sigma;
};
const std::vector<TestParams> CONFIGS = {
  {16, 8, 2, 8, 0, 0},
  {16, 8, 4, 8, 1, 0},
  {16, 8, 4, 8, 2, 1},
  {16384, 256, 64, 256, 63, 0}
};

class PWelchComplexExponentialTest : public ::testing::TestWithParam<TestParams>
{
public:
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();

    if (params.nfft > 8) thresh = 1.f; // use higher threshold for larger fft sizes
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.01f;
  TestParams params = ::testing::TestWithParam<TestParams>::GetParam();
};

template <typename TypeParam>
void helper(PWelchComplexExponentialTest& test)
{
  MATX_ENTER_HANDLER();
  test.pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "pwelch_operators", "pwelch_complex_exponential", {test.params.signal_size, test.params.nperseg, test.params.noverlap, test.params.nfft, test.params.ftone, test.params.sigma});

  tensor_t<TypeParam, 1> x{{test.params.signal_size}};
  test.pb->NumpyToTensorView(x, "x_in");

  // example-begin pwelch-test-1
  auto Pxx  = make_tensor<typename TypeParam::value_type>({test.params.nfft});
  (Pxx = pwelch(x, test.params.nperseg, test.params.noverlap, test.params.nfft)).run();
  // example-end pwelch-test-1

  cudaStreamSynchronize(0);

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

INSTANTIATE_TEST_CASE_P(PWelchComplexExponentialTests, PWelchComplexExponentialTest,::testing::ValuesIn(CONFIGS));

TEST(PWelchOpTest, xin_complex_float)
{
  float thresh = 0.01f;
  index_t signal_size = 16;
  index_t nperseg = 8;
  index_t noverlap = 0;
  index_t nfft = 8;
  auto x = ones<cuda::std::complex<float>>({signal_size});
  auto Pxx  = make_tensor<float>({nfft});
  (Pxx = pwelch(x, nperseg, noverlap, nfft)).run();

  cudaStreamSynchronize(0);

  EXPECT_NEAR(Pxx(0), 64, thresh);
  for (index_t k=1; k<nfft; k++)
  {
    EXPECT_NEAR(Pxx(k), 0, thresh) << "failure at index k=" << k;
  }
}
