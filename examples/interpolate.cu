////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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
#include <cassert>
#include <cstdio>

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();


  // Create a CUDA stream and executor
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaExecutor exec{stream};


  index_t in_count = 10;
  index_t out_count = 100;

  auto x = linspace(0.0f, 1.0f, in_count);
  auto v = sin(x);
  auto xq = linspace(0.0f, 1.0f, out_count);

  // Execute the interpolation
  auto vq = make_tensor<float>({out_count});
  (vq = interp1<InterpMethodLinear>(x, v, xq)).run(exec);
  exec.sync();

  // Print the results
  std::cout << "Interpolation Results:" << std::endl;
  std::cout << "x values: ";
  for (int i = 0; i < 5; i++) {
    std::cout << x(i) << " ";
  }
  std::cout << "..." << std::endl;

  std::cout << "v values: ";
  for (int i = 0; i < 5; i++) {
    std::cout << v(i) << " ";
  }
  std::cout << "..." << std::endl;

  std::cout << "xq values: ";
  for (int i = 0; i < 5; i++) {
    std::cout << xq(i) << " ";
  }
  std::cout << "..." << std::endl;

  std::cout << "Interpolated values: ";
  for (int i = 0; i < 5; i++) {
    std::cout << vq(i) << " ";
  }
  std::cout << "..." << std::endl;

  // Clean up
  cudaStreamDestroy(stream);

  matxPrintMemoryStatistics();

  MATX_CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
