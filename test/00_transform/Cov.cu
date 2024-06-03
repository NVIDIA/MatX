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

template <typename T> class CovarianceTest : public ::testing::Test {

protected:
  const index_t cov_dim1 = 4;
  const index_t cov_dim2 = 3;
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;

  void SetUp() override
  {
    CheckTestTensorCoreTypeSupport<GTestType>();
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitTVGenerator<GTestType>("00_transforms", "cov_operators", {cov_dim1, cov_dim2});

    // Half precision needs a bit more tolerance when compared to
    // fp32
    if constexpr (is_complex_half_v<GTestType> || is_matx_half_v<GTestType>) {
      thresh = 0.1f;
    }
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  tensor_t<GTestType, 2> av{{cov_dim1, cov_dim2}};
  tensor_t<GTestType, 2> cv{{cov_dim2, cov_dim2}};

  float thresh = 0.01f;
  std::unique_ptr<detail::MatXPybind> pb;
};

template <typename TensorType>
class CovarianceTestFloatTypes : public CovarianceTest<TensorType> {
};

TYPED_TEST_SUITE(CovarianceTestFloatTypes, MatXFloatTypesCUDAExec);

TYPED_TEST(CovarianceTestFloatTypes, SmallCov)
{
  MATX_ENTER_HANDLER();
  this->pb->RunTVGenerator("cov");
  this->pb->NumpyToTensorView(this->av, "a");
  // example-begin cov-test-1
  (this->cv = cov(this->av)).run(this->exec);
  // example-end cov-test-1
  this->exec.sync();
  MATX_TEST_ASSERT_COMPARE(this->pb, this->cv, "c_cov", this->thresh);
  MATX_EXIT_HANDLER();
}

// The previous slice code was incorrect, but passing on x86. When compiling on ARM all the sizes
// were incorrect. The slice syntax was fixed, but after doing that it fails on the 8th element
// in the batch
TYPED_TEST(CovarianceTestFloatTypes, BatchedCov)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  this->pb->RunTVGenerator("cov");
  this->pb->NumpyToTensorView(this->av, "a");

  const int m = 3;
  const int n = 5;
  const int k = 7;

  // Test a batched 5D input
  auto batched_in = make_tensor<TestType>({m, n, k, this->cov_dim1, this->cov_dim2});
  auto batched_out = make_tensor<TestType>({m, n, k, this->cov_dim2, this->cov_dim2});
  (batched_in = clone<5>(this->av, {m, n, k, matxKeepDim, matxKeepDim})).run(this->exec);

  (batched_out = cov(batched_in)).run(this->exec);
  this->exec.sync();

  for (int im = 0; im < m; im++) {
    for (int in = 0; in < n; in++) {
      for (int ik = 0; ik < k; ik++) {
        auto bv = slice<2>(batched_out, {im,in,ik,0,0}, {matxDropDim,matxDropDim,matxDropDim,matxEnd,matxEnd});
        MATX_TEST_ASSERT_COMPARE(this->pb, bv, "c_cov", this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}
