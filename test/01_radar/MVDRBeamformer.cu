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
#include "mvdr_beamformer.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;
using complex = cuda::std::complex<float>;

TEST(Radar, MVDRBeamformer)
{
  MATX_ENTER_HANDLER();

  index_t num_beams = 60;
  index_t num_el = 6;
  index_t data_len = 1000;
  index_t snap_len = 2 * num_el;

  auto mvdr = MVDRBeamformer(num_beams, num_el, data_len, snap_len);

  mvdr.Prefetch(0);

  auto pb = std::make_unique<detail::MatXPybind>();
  pb->InitAndRunTVGenerator<complex>("mvdr_beamformer", "mvdr_beamformer",
                                     "run", {data_len, num_beams, num_el});

  auto in_vec = mvdr.GetInVec();
  auto v = mvdr.GetV();
  // auto cov_mat  = mvdr.GetCovMatView();
  auto cov_inv = mvdr.GetCovMatInvView();

  pb->NumpyToTensorView(in_vec, "in_vec");
  pb->NumpyToTensorView(v, "v");

  mvdr.Run(0);
  cudaStreamSynchronize(0);

  auto cbf = mvdr.GetCBFView();

  MATX_TEST_ASSERT_COMPARE(pb, in_vec, "in_vec", 0.01);
  MATX_TEST_ASSERT_COMPARE(pb, cbf, "out_cbf", 0.01);
  MATX_TEST_ASSERT_COMPARE(pb, cov_inv, "cov_inv", 0.01);

  MATX_EXIT_HANDLER();
}