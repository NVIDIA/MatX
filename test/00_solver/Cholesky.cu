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
constexpr int dim_size = 100;

template <typename T> class CholSolverTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<T>("00_solver", "cholesky", "run", {dim_size});
    pb->NumpyToTensorView(Bv, "B");
    pb->NumpyToTensorView(Lv, "L");
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<T, 2> Bv{{dim_size, dim_size}};
  tensor_t<T, 2> Lv{{dim_size, dim_size}};
};

template <typename TensorType>
class CholSolverTestNonComplexFloatTypes : public CholSolverTest<TensorType> {
};

TYPED_TEST_SUITE(CholSolverTestNonComplexFloatTypes,
                 MatXFloatNonComplexNonHalfTypes);

TYPED_TEST(CholSolverTestNonComplexFloatTypes, CholeskyBasic)
{
  MATX_ENTER_HANDLER();

  chol(this->Bv, this->Bv, 0, CUBLAS_FILL_MODE_LOWER);
  cudaStreamSynchronize(0);

  // Cholesky only saves the upper triangle by default, and the lower triangle
  // is garbage. Python saves the opposite.
  for (index_t i = 0; i < this->Bv.Size(0); i++) {
    for (index_t j = 0; j <= i; j++) {
      ASSERT_NEAR(this->Bv(i, j), this->Lv(i, j), 0.001);
    }
  }

  MATX_EXIT_HANDLER();
}
