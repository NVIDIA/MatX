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
#include "matx_pybind.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;

template <typename T> class CovarianceTest : public ::testing::Test {

protected:
  const index_t cov_dim = 4;
  void SetUp() override
  {
    CheckTestTensorCoreTypeSupport<T>();
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitTVGenerator<T>("00_transforms", "cov_operators", {cov_dim});

    // Half precision needs a bit more tolerance when compared to
    // fp32
    if constexpr (is_complex_half_v<T> || is_matx_half_v<T>) {
      thresh = 0.1f;
    }
  }

  void TearDown() { pb.reset(); }

  tensor_t<T, 2> av{{cov_dim, cov_dim}};
  tensor_t<T, 2> cv{{cov_dim, cov_dim}};

  float thresh = 0.01f;
  std::unique_ptr<detail::MatXPybind> pb;
};

template <typename TensorType>
class CovarianceTestFloatTypes : public CovarianceTest<TensorType> {
};

TYPED_TEST_SUITE(CovarianceTestFloatTypes, MatXFloatTypes);

TYPED_TEST(CovarianceTestFloatTypes, SmallCov)
{
  MATX_ENTER_HANDLER();
  this->pb->RunTVGenerator("cov");
  this->pb->NumpyToTensorView(this->av, "a");
  cov(this->cv, this->av, 0);

  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv, "c_cov", this->thresh);
  MATX_EXIT_HANDLER();
}
