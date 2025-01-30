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

template <typename T> class PinvSolverTest : public ::testing::Test {
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;   
protected:
  void SetUp() override
  {
    if constexpr (!detail::CheckSolverSupport<GExecType>()) {
      GTEST_SKIP();
    }

    // Use an arbitrary number of threads for the select threads host exec.
    if constexpr (is_select_threads_host_executor_v<GExecType>) {
      HostExecParams params{4};
      exec = SelectThreadsHostExecutor{params};
    }

    pb = std::make_unique<detail::MatXPybind>();
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.001f;
};

template <typename TensorType>
class PinvSolverTestFloatTypes : public PinvSolverTest<TensorType> {
};


TYPED_TEST_SUITE(PinvSolverTestFloatTypes,
                 MatXFloatNonHalfTypesAllExecs);

TYPED_TEST(PinvSolverTestFloatTypes, PinvBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr cuda::std::array sizes {
    std::pair{100, 50},
    std::pair{100, 100},
    std::pair{50, 100}
  };

  for (const auto& [m, n] : sizes) {
    auto A = make_tensor<TestType>({m, n});
    auto A_pinv = make_tensor<TestType>({n, m});

    this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "pinv", "run", {m, n});
    this->pb->NumpyToTensorView(A, "A");

    // example-begin pinv-test-1
    (A_pinv = pinv(A)).run(this->exec);
    // example-end pinv-test-1
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, A_pinv, "pinv", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(PinvSolverTestFloatTypes, PinvRankDeficient)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr cuda::std::array sizes {
    std::pair{100, 50},
    std::pair{100, 100},
    std::pair{50, 100}
  };

  for (const auto& [m, n] : sizes) {
    auto A = make_tensor<TestType>({m, n});
    auto A_pinv = make_tensor<TestType>({n, m});

    this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "pinv", "run_rank_deficient", {m, n});
    this->pb->NumpyToTensorView(A, "A");

    (A_pinv = pinv(A)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, A_pinv, "pinv", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(PinvSolverTestFloatTypes, PinvBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr int m = 100;
  constexpr int n = 50;
  constexpr int batches = 10;
  auto A = make_tensor<TestType>({batches, m, n});
  auto A_pinv = make_tensor<TestType>({batches, n, m});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "pinv", "run", {batches, m, n});
  this->pb->NumpyToTensorView(A, "A");

  (A_pinv = pinv(A)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, A_pinv, "pinv", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(PinvSolverTestFloatTypes, PinvBatchedRankDeficient)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  constexpr int m = 100;
  constexpr int n = 50;
  constexpr int batches = 10;
  auto A = make_tensor<TestType>({batches, m, n});
  auto A_pinv = make_tensor<TestType>({batches, n, m});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "pinv", "run_rank_deficient", {batches, m, n});
  this->pb->NumpyToTensorView(A, "A");

  (A_pinv = pinv(A)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, A_pinv, "pinv", this->thresh);

  MATX_EXIT_HANDLER();
}