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

template <typename T> class EigenSolverTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<T>("00_solver", "eig", "run", {dim_size});
    pb->NumpyToTensorView(Bv, "B");
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<T, 2> Bv{{dim_size, dim_size}};
  tensor_t<T, 2> Btv{{dim_size, dim_size}};
  tensor_t<T, 2> Evv{{dim_size, dim_size}};

  tensor_t<T, 2> Wv{{dim_size, 1}};
  tensor_t<T, 1> Wov{{dim_size}};

  tensor_t<T, 2> Gtv{{dim_size, 1}};
  tensor_t<T, 2> Lvv{{dim_size, 1}};
};

template <typename TensorType>
class EigenSolverTestNonComplexFloatTypes : public EigenSolverTest<TensorType> {
};

TYPED_TEST_SUITE(EigenSolverTestNonComplexFloatTypes,
                 MatXFloatNonComplexNonHalfTypes);

TYPED_TEST(EigenSolverTestNonComplexFloatTypes, EigenBasic)
{
  MATX_ENTER_HANDLER();
  eig(this->Evv, this->Wov, this->Bv);

  // Now we need to go through all the eigenvectors and eigenvalues and make
  // sure the results match the equation A*v = lambda*v, where v are the
  // eigenvectors corresponding to the eigenvalue lambda.
  for (index_t i = 0; i < dim_size; i++) {
    auto v = this->Evv.template Slice<2>({0, i}, {matxEnd, i + 1});
    matx::copy(this->Wv, v, 0);

    // Compute lambda*v
    auto b = v * this->Wov(i);
    (this->Lvv = b).run();
    // Compute A*v

    matmul(this->Gtv, this->Bv, this->Wv);
    cudaStreamSynchronize(0);
    // Compare
    for (index_t j = 0; j < dim_size; j++) {
      ASSERT_NEAR(this->Gtv(j, 0), this->Lvv(j, 0), 0.001);
    }
  }

  MATX_EXIT_HANDLER();
}
