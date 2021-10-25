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

#include "matx.h"
#include "mvdr_beamformer.h"
#include <cassert>
#include <cstdio>
#include <memory>
#include <stdlib.h>

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  uint32_t num_beams = 60;
  uint32_t num_el = 6;
  uint32_t data_len = 65536;
  uint32_t snap_len = 2 * num_el;

  constexpr uint32_t num_iterations = 1;
  float time_ms;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto mvdr = MVDRBeamformer(num_beams, num_el, data_len, snap_len);

  auto v = mvdr.GetV();
  for (index_t i = 0; i < v.Size(0); i++) {
    for (index_t j = 0; j < v.Size(1); j++) {
      v(i, j) = {static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                 static_cast<float>(rand()) / static_cast<float>(RAND_MAX)};
    }
  }

  auto invec = mvdr.GetInVec();
  for (index_t i = 0; i < invec.Size(0); i++) {
    for (index_t j = 0; j < invec.Size(1); j++) {
      invec(i, j) = {static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                     static_cast<float>(rand()) / static_cast<float>(RAND_MAX)};
    }
  }

  mvdr.Prefetch(stream);

  cudaEventRecord(start, stream);

  for (uint32_t i = 0; i < num_iterations; i++) {
    mvdr.Run(stream);
  }

  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  cudaEventElapsedTime(&time_ms, start, stop);

  printf("MVDR Kernel Time = %.2fms per iteration\n", time_ms / num_iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}