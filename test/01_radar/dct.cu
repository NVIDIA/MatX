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
#include "matx_radar.h"
#include "matx_signal.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;
using complex = cuda::std::complex<float>;

class DctTests : public ::testing::Test {
protected:
  index_t sig_size = 100;
  void SetUp() override
  {

    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<complex>("01_signal", "dct", "run", {sig_size});

    pb->NumpyToTensorView(xv, "x");
  }

  void TearDown() { pb.reset(); }

  tensor_t<float, 1> xv{{sig_size}};
  std::unique_ptr<detail::MatXPybind> pb;
};

/* Real 1D DCT with N=100 */
TEST_F(DctTests, Real1DN100)
{
  MATX_ENTER_HANDLER();

  tensor_t<float, 1> out{{sig_size}};
  signal::dct(out, xv);
  MATX_TEST_ASSERT_COMPARE(pb, out, "Y", 0.01);

  MATX_EXIT_HANDLER();
}
