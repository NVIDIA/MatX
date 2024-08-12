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

template <typename T> class CholSolverTest : public ::testing::Test {
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

  std::unique_ptr<detail::MatXPybind> pb;
  GExecType exec{};
  float thresh = 0.001f;
};

template <typename TensorType>
class CholSolverTestNonHalfFloatTypes : public CholSolverTest<TensorType> {
};

TYPED_TEST_SUITE(CholSolverTestNonHalfFloatTypes,
  MatXFloatNonHalfTypesAllExecs);

TYPED_TEST(CholSolverTestNonHalfFloatTypes, CholeskyBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  const cuda::std::array dims {
    16,
    50,
    100,
    130,
    200,
    1000
  };

  for (size_t k = 0; k < dims.size(); k++) {
    this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "cholesky", "run", {dims[k]});
    auto Bv = make_tensor<TestType>({dims[k], dims[k]});
    auto Lv = make_tensor<TestType>({dims[k], dims[k]});
    this->pb->NumpyToTensorView(Bv, "B");
    this->pb->NumpyToTensorView(Lv, "L");

    // example-begin chol-test-1
    (Bv = chol(Bv, SolverFillMode::LOWER)).run(this->exec);
    // example-end chol-test-1
    this->exec.sync();

    // Cholesky fills the lower triangular portion (due to SolverFillMode::LOWER)
    // and destroys the upper triangular portion.
    if constexpr (is_complex_v<TestType>) {
      for (index_t i = 0; i < dims[k]; i++) {
        for (index_t j = 0; j <= i; j++) {
          ASSERT_NEAR(Bv(i, j).real(), Lv(i, j).real(), this->thresh);
          ASSERT_NEAR(Bv(i, j).imag(), Lv(i, j).imag(), this->thresh);
        }
      }
    } else {
      for (index_t i = 0; i < dims[k]; i++) {
        for (index_t j = 0; j <= i; j++) {
          ASSERT_NEAR(Bv(i, j), Lv(i, j), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CholSolverTestNonHalfFloatTypes, CholeskyBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  const cuda::std::array dims {
    16,
    50,
    100,
    130,
    200,
    1000
  };
  constexpr index_t batches = 10;

  for (size_t k = 0; k < dims.size(); k++) {
    this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "cholesky", "run", {batches, dims[k]});
    auto Bv = make_tensor<TestType>({batches, dims[k], dims[k]});
    auto Lv = make_tensor<TestType>({batches, dims[k], dims[k]});
    this->pb->NumpyToTensorView(Bv, "B");
    this->pb->NumpyToTensorView(Lv, "L");

    (Bv = chol(Bv, SolverFillMode::LOWER)).run(this->exec);
    this->exec.sync();

    // Cholesky fills the lower triangular portion (due to SolverFillMode::LOWER)
    // and destroys the upper triangular portion.
    for (index_t b = 0; b < batches; b++) {
      if constexpr (is_complex_v<TestType>) {
        for (index_t i = 0; i < dims[k]; i++) {
          for (index_t j = 0; j <= i; j++) {
            ASSERT_NEAR(Bv(b, i, j).real(), Lv(b, i, j).real(), this->thresh);
            ASSERT_NEAR(Bv(b, i, j).imag(), Lv(b, i, j).imag(), this->thresh);
          }
        }
      } else {
        for (index_t i = 0; i < dims[k]; i++) {
          for (index_t j = 0; j <= i; j++) {
            ASSERT_NEAR(Bv(b, i, j), Lv(b, i, j), this->thresh);
          }
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(CholSolverTestNonHalfFloatTypes, CholeskyWindowed)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  const cuda::std::array dims {
    50,
    100,
    130,
    200,
    1000
  };

  for (size_t k = 0; k < dims.size(); k++) {
    this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "cholesky", "run", {dims[k]});
    auto Bv = make_tensor<TestType>({2*dims[k], 3*dims[k]});
    auto Bslice = slice<2>(Bv, {11, 50}, {dims[k]+11, dims[k]+50});
    auto Cv = make_tensor<TestType>({dims[k], dims[k]});
    auto Lv = make_tensor<TestType>({dims[k], dims[k]});
    this->pb->NumpyToTensorView(Cv, "B");
    this->pb->NumpyToTensorView(Lv, "L");
    (Bslice = Cv).run(this->exec);
    this->exec.sync();

    (Bslice = chol(Bslice, SolverFillMode::LOWER)).run(this->exec);
    this->exec.sync();

    // Cholesky fills the lower triangular portion (due to SolverFillMode::LOWER)
    // and destroys the upper triangular portion.
    if constexpr (is_complex_v<TestType>) {
      for (index_t i = 0; i < dims[k]; i++) {
        for (index_t j = 0; j <= i; j++) {
          ASSERT_NEAR(Bslice(i, j).real(), Lv(i, j).real(), this->thresh);
          ASSERT_NEAR(Bslice(i, j).imag(), Lv(i, j).imag(), this->thresh);
        }
      }
    } else {
      for (index_t i = 0; i < dims[k]; i++) {
        for (index_t j = 0; j <= i; j++) {
          ASSERT_NEAR(Bslice(i, j), Lv(i, j), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}
