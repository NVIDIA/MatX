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
constexpr index_t m = 100;
constexpr index_t n = 50;

template <typename T> class SVDSolverTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<T>("00_solver", "svd", "run", {m, n});
    pb->NumpyToTensorView(Av, "A");
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<T, 2> Av{{m, n}};
  tensor_t<T, 2> Atv{{n, m}};
  tensor_t<T, 1> Sv{{std::min(m, n)}};
  tensor_t<T, 2> Uv{{m, m}};
  tensor_t<T, 2> Vv{{n, n}};

  tensor_t<T, 2> Sav{{m, n}};
  tensor_t<T, 2> Uav{{m, m}};
  tensor_t<T, 2> Vav{{n, n}};

  // Used only for validation
  tensor_t<T, 2> tmpV{{m, n}};
};

template <typename TensorType>
class SVDSolverTestNonComplexFloatTypes : public SVDSolverTest<TensorType> {
};

TYPED_TEST_SUITE(SVDSolverTestNonComplexFloatTypes,
                 MatXFloatNonComplexNonHalfTypes);

TYPED_TEST(SVDSolverTestNonComplexFloatTypes, SVDBasic)
{
  MATX_ENTER_HANDLER();

  // cuSolver only supports col-major solving today, so we need to transpose,
  // solve, then transpose again to compare to Python
  transpose(this->Atv, this->Av, 0);

  auto Atv2 = this->Atv.View({m, n});
  svd(this->Uv, this->Sv, this->Vv, Atv2);

  cudaStreamSynchronize(0);

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.
  // However, U and V are in column-major format, so we have to transpose them
  // back to verify the identity.
  transpose(this->Uav, this->Uv, 0);
  transpose(this->Vav, this->Vv, 0);

  // Zero out s
  (this->Sav = zeros({m, n})).run();
  cudaStreamSynchronize(0);

  // Construct S matrix since it's just a vector from cuSolver
  for (index_t i = 0; i < n; i++) {
    this->Sav(i, i) = this->Sv(i);
  }

  cudaStreamSynchronize(0);

  matmul(this->tmpV, this->Uav, this->Sav); // U * S
  matmul(this->Sav, this->tmpV, this->Vav); // (U * S) * V'
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < this->Av.Size(0); i++) {
    for (index_t j = 0; j < this->Av.Size(1); j++) {
      ASSERT_NEAR(this->Av(i, j), this->Sav(i, j), 0.001) << i << " " << j;
    }
  }

  MATX_EXIT_HANDLER();
}
