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
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<T>("00_solver", "lu", "run", {m, n});
    pb->NumpyToTensorView(Av, "A");
    pb->NumpyToTensorView(Lv, "L");
    pb->NumpyToTensorView(Uv, "U");
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  tensor_t<T, 2> Av{{m, n}};
  tensor_t<T, 2> Atv{{n, m}};
  tensor_t<int64_t, 1> PivV{{std::min(m, n)}};
  tensor_t<T, 2> Lv{{m, std::min(m, n)}};
  tensor_t<T, 2> Uv{{std::min(m, n), n}};
};

template <typename TensorType>
class LUSolverTestNonComplexFloatTypes : public LUSolverTest<TensorType> {
};

TYPED_TEST_SUITE(LUSolverTestNonComplexFloatTypes,
                 MatXFloatNonComplexNonHalfTypes);

TYPED_TEST(LUSolverTestNonComplexFloatTypes, LUBasic)
{
  MATX_ENTER_HANDLER();
  lu(this->Av, this->PivV, this->Av);
  cudaStreamSynchronize(0);

  // The upper and lower triangle components are saved in Av. Python saves them
  // as separate matrices with the diagonal of the lower matrix set to 0
  for (index_t i = 0; i < this->Av.Size(0); i++) {
    for (index_t j = 0; j < this->Av.Size(1); j++) {
      if (i > j) { // Lower triangle
        ASSERT_NEAR(this->Av(i, j), this->Lv(i, j), 0.001);
      }
      else {
        ASSERT_NEAR(this->Av(i, j), this->Uv(i, j), 0.001);
      }
    }
  }

  MATX_EXIT_HANDLER();
}
