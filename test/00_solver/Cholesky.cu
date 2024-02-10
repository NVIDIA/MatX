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
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
};

template <typename TensorType>
class CholSolverTestNonHalfFloatTypes : public CholSolverTest<TensorType> {
};

TYPED_TEST_SUITE(CholSolverTestNonHalfFloatTypes,
  MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(CholSolverTestNonHalfFloatTypes, CholeskyBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;     
  ExecType exec;

  const std::array dims {
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
    (Bv = chol(Bv, CUBLAS_FILL_MODE_LOWER)).run(exec);
    // example-end chol-test-1
    cudaStreamSynchronize(0);

    // Cholesky fills the lower triangular portion (due to CUBLAS_FILL_MODE_LOWER)
    // and destroys the upper triangular portion.
    if constexpr (is_complex_v<TestType>) {
      for (index_t i = 0; i < dims[k]; i++) {
        for (index_t j = 0; j <= i; j++) {
          ASSERT_NEAR(Bv(i, j).real(), Lv(i, j).real(), 0.001);
          ASSERT_NEAR(Bv(i, j).imag(), Lv(i, j).imag(), 0.001);
        }
      }
    } else {
      for (index_t i = 0; i < dims[k]; i++) {
        for (index_t j = 0; j <= i; j++) {
          ASSERT_NEAR(Bv(i, j), Lv(i, j), 0.001);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CholSolverTestNonHalfFloatTypes, CholeskyWindowed)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;     
  ExecType exec;  

  const std::array dims {
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
    (Bslice = Cv).run(exec);
    cudaStreamSynchronize(0);

    (Bslice = chol(Bslice, CUBLAS_FILL_MODE_LOWER)).run(exec);
    cudaStreamSynchronize(0);

    // Cholesky fills the lower triangular portion (due to CUBLAS_FILL_MODE_LOWER)
    // and destroys the upper triangular portion.
    if constexpr (is_complex_v<TestType>) {
      for (index_t i = 0; i < dims[k]; i++) {
        for (index_t j = 0; j <= i; j++) {
          ASSERT_NEAR(Bslice(i, j).real(), Lv(i, j).real(), 0.001);
          ASSERT_NEAR(Bslice(i, j).imag(), Lv(i, j).imag(), 0.001);
        }
      }
    } else {
      for (index_t i = 0; i < dims[k]; i++) {
        for (index_t j = 0; j <= i; j++) {
          ASSERT_NEAR(Bslice(i, j), Lv(i, j), 0.001);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}
