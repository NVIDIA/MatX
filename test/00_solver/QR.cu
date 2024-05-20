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

template <typename T> class QRSolverTest : public ::testing::Test {
protected:
  using dtype = float;
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;   
   
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<GTestType>("00_solver", "qr", "run", {m, n});
    pb->NumpyToTensorView(Av, "A");
    pb->NumpyToTensorView(Qv, "Q");
    pb->NumpyToTensorView(Rv, "R");
  }

  void TearDown() override { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  GExecType exec{};   
  tensor_t<GTestType, 2> Av{{m, n}};
  tensor_t<GTestType, 2> Atv{{n, m}};
  tensor_t<GTestType, 1> TauV{{std::min(m, n)}};
  tensor_t<GTestType, 2> Qv{{m, std::min(m, n)}};
  tensor_t<GTestType, 2> Rv{{std::min(m, n), n}};
};

template <typename TensorType>
class QRSolverTestNonComplexFloatTypes : public QRSolverTest<TensorType> {
};

TYPED_TEST_SUITE(QRSolverTestNonComplexFloatTypes,
                 MatXFloatNonComplexNonHalfTypesCUDAExec);

TYPED_TEST(QRSolverTestNonComplexFloatTypes, QRBasic)
{
  MATX_ENTER_HANDLER();

  // example-begin cusolver_qr-test-1
  // cuSolver only supports col-major solving today, so we need to transpose,
  // solve, then transpose again to compare to Python
  (mtie(this->Av, this->TauV) = cusolver_qr(this->Av)).run(this->exec);
  // example-end cusolver_qr-test-1
  cudaStreamSynchronize(0);

  // For now we're only verifying R. Q is a bit more complex to compute since
  // cuSolver/BLAS don't return Q, and instead return Householder reflections
  // that are used to compute Q. Eventually compute Q here and verify
  for (index_t i = 0; i < this->Av.Size(0); i++) {
    for (index_t j = 0; j < this->Av.Size(1); j++) {
      // R is stored only in the top triangle of A
      if (i <= j) {
        ASSERT_NEAR(this->Av(i, j), this->Rv(i, j), 0.001);
      }
    }
  }

  MATX_EXIT_HANDLER();
}
