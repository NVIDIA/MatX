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

#include <cassert>
#include <cstdio>
#include <math.h>
#include <memory>

#include "matx.h"

using namespace matx;
int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  using TypeParam = double;
  MATX_ENTER_HANDLER();

  int max_iters=100;
  int N = 120;
  int BATCH = 900;

  auto A = make_tensor<TypeParam, 3> ({BATCH, N, N});
  auto X = make_tensor<TypeParam, 2> ({BATCH, N});
  auto B = make_tensor<TypeParam, 2> ({BATCH, N});
  auto Bout = make_tensor<TypeParam, 2> ({BATCH, N});
  auto norm = make_tensor<TypeParam, 1>({BATCH});
  auto maxn = make_tensor<TypeParam>({});

  cudaExecutor exec{};

  // Simple Poisson matrix
  for(int b = 0; b < BATCH; b++) {
    for(int i = 0; i < N; i++) {
      B(b,i) = TypeParam(1+b);

      for(int j = 0; j < N; j++) {
        if(i==j)
          A(b,i,j) = 2;
        else if( i == j-1)
          A(b,i,j) = -1;
        else if (i == j+1)
          A(b,i,j) = -1;
        else
          A(b,i,j) = 0;
      }
    }
  }

  (X = TypeParam(1)).run(exec);

  (X = cgsolve(A, B, .0001, max_iters)).run(exec);
  // example-begin sync-test-1
  (Bout = matvec(A, X)).run(exec);
  (norm = sum((Bout-B)*(Bout-B))).run(exec);
  (maxn = matx::max(sqrt(norm))).run(exec);

  exec.sync();
  // example-end sync-test-1
  printf ("max l2 norm: %f\n", (float)sqrt(maxn()));

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
