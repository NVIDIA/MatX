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
#include "matx/transforms/transpose.h"

using namespace matx;
constexpr int m = 15;

template <typename T> class DetSolverTest : public ::testing::Test {
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
  float relTol = 2e-5f;
};

template <typename TensorType>
class DetSolverTestFloatTypes : public DetSolverTest<TensorType> {
};

template<typename T>
double getMaxMagnitude(const T& value) {
  if constexpr (is_complex_v<T>) {
    return cuda::std::max(cuda::std::fabs(value.real()), cuda::std::fabs(value.imag()));
  } else {
    return cuda::std::fabs(value);
  }
}

TYPED_TEST_SUITE(DetSolverTestFloatTypes,
                 MatXFloatNonHalfTypesAllExecs);

TYPED_TEST(DetSolverTestFloatTypes, DeterminantBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  auto Av = make_tensor<TestType>({m, m});
  auto detv = make_tensor<TestType>({});
  auto detv_ref = make_tensor<TestType>({});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "det", "run", {m});
  this->pb->NumpyToTensorView(Av, "A");

  // example-begin det-test-1
  (detv = det(Av)).run(this->exec);
  // example-end det-test-1
  this->exec.sync();

  this->pb->NumpyToTensorView(detv_ref, "det");

  // The relative error is on the order of 2e-5 compared to the Python result
  // for float types
  auto thresh =  this->relTol * getMaxMagnitude(detv_ref());

  MATX_TEST_ASSERT_COMPARE(this->pb, detv, "det", thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DetSolverTestFloatTypes, DeterminantBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using inner_type = typename inner_op_type_t<TestType>::type;

  constexpr int batches = 10;

  auto Av = make_tensor<TestType>({batches, m, m});
  auto detv = make_tensor<TestType>({batches});
  auto detv_ref = make_tensor<TestType>({batches});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "det", "run", {batches, m});
  this->pb->NumpyToTensorView(Av, "A");
  this->pb->NumpyToTensorView(detv_ref, "det");

  (detv = det(Av)).run(this->exec);
  this->exec.sync();

  for (index_t b = 0; b < batches; b++) {
    auto detv_ref_b = detv_ref(b);
    auto thresh =  this->relTol * getMaxMagnitude(detv_ref_b);

    if constexpr (is_complex_v<TestType>) {
      ASSERT_NEAR(detv(b).real(), detv_ref_b.real(), thresh);
      ASSERT_NEAR(detv(b).imag(), detv_ref_b.imag(), thresh);
    } else {
      ASSERT_NEAR(detv(b), detv_ref_b, thresh);
    }
  }

  MATX_EXIT_HANDLER();
}