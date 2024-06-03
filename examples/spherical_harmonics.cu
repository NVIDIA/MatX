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
#include <cassert>
#include <cstdio>
#include <math.h>

using namespace matx;

template<typename T> T factorial(int N) {
  T prod = 1;
  for(int i=2; i<=N; i++) {
    prod = prod * i;
  }
  return prod;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  cudaExecutor exec{};

  using ValueType = double;

  int l = 3;
  int m = 2;
  int n = 60;
  ValueType dx = M_PI/n;

  auto col = range<0>({n+1},ValueType(0), ValueType(dx));
  auto az = range<0>({2*n+1}, ValueType(0), ValueType(dx));

  auto [phi, theta] = meshgrid(az, col);

  auto Plm = lcollapse<3>(legendre(l, m, cos(theta)));
 
  ValueType a = (2*l+1)*factorial<ValueType>(l-m);
  ValueType b = 4*M_PI*factorial<ValueType>(l+m);
  ValueType C = cuda::std::sqrt(a/b);

  auto Ylm = C * Plm * exp(cuda::std::complex<ValueType>(0,1)*(m*phi));

  auto [ Xm, Ym, Zm ] = sph2cart(phi, ValueType(M_PI)/2 - theta, abs(real(Ylm)));

  // Output location
  auto X = make_tensor<ValueType>(Xm.Shape());
  auto Y = make_tensor<ValueType>(Ym.Shape());
  auto Z = make_tensor<ValueType>(Zm.Shape());

  (X = Xm, Y = Ym, Z=Zm).run(exec);

  exec.sync();

#if MATX_ENABLE_VIZ
  matx::viz::surf(X, Y, Z, "test-viz.html");
#endif
  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
