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
constexpr int m = 100;
constexpr int n = 50;

template <typename T> class LUSolverTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;   
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
class LUSolverTestFloatTypes : public LUSolverTest<TensorType> {
};

TYPED_TEST_SUITE(LUSolverTestFloatTypes,
                 MatXFloatNonHalfTypesAllExecs);

TYPED_TEST(LUSolverTestFloatTypes, LUBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using piv_value_type = std::conditional_t<is_cuda_executor_v<ExecType>, int64_t, lapack_int_t>;

  auto Av = make_tensor<TestType>({m, n});
  auto PivV = make_tensor<piv_value_type>({std::min(m, n)});
  auto Lv = make_tensor<TestType>({m, std::min(m, n)});
  auto Uv = make_tensor<TestType>({std::min(m, n), n});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "lu", "run", {m, n});
  this->pb->NumpyToTensorView(Av, "A");
  this->pb->NumpyToTensorView(Lv, "L");
  this->pb->NumpyToTensorView(Uv, "U");

  // example-begin lu-test-1
  (mtie(Av, PivV) =  lu(Av)).run(this->exec);
  // example-end lu-test-1
  this->exec.sync();

  // The upper and lower triangle components are saved in Av. Python saves them
  // as separate matrices with the diagonal of the lower matrix set to 0
  for (index_t i = 0; i < Av.Size(0); i++) {
    for (index_t j = 0; j < Av.Size(1); j++) {
      TestType refv = i > j ? Lv(i, j) : Uv(i, j);
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Av(i, j).real(), refv.real(), this->thresh);
        ASSERT_NEAR(Av(i, j).imag(), refv.imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Av(i, j), refv, this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(LUSolverTestFloatTypes, LUBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  using piv_value_type = std::conditional_t<is_cuda_executor_v<ExecType>, int64_t, lapack_int_t>;
  constexpr int batches = 10;

  auto Av = make_tensor<TestType>({batches, m, n});
  auto PivV = make_tensor<piv_value_type>({batches, std::min(m, n)});
  auto Lv = make_tensor<TestType>({batches, m, std::min(m, n)});
  auto Uv = make_tensor<TestType>({batches, std::min(m, n), n});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "lu", "run", {batches, m, n});
  this->pb->NumpyToTensorView(Av, "A");
  this->pb->NumpyToTensorView(Lv, "L");
  this->pb->NumpyToTensorView(Uv, "U");

  (mtie(Av, PivV) =  lu(Av)).run(this->exec);
  this->exec.sync();

  // The upper and lower triangle components are saved in Av. Python saves them
  // as separate matrices with the diagonal of the lower matrix set to 0
  for (index_t b = 0; b < Av.Size(0); b++) {
    for (index_t i = 0; i < Av.Size(1); i++) {
      for (index_t j = 0; j < Av.Size(2); j++) {
        TestType act = i > j ? Lv(b, i, j) : Uv(b, i, j);
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Av(b, i, j).real(), act.real(), this->thresh);
          ASSERT_NEAR(Av(b, i, j).imag(), act.imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Av(b, i, j), act, this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}