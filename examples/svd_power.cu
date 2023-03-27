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

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  //using AType = float;
  using AType = cuda::std::complex<float>;
  using SType = float;

  cudaStream_t stream = 0;
  int batch = 1; 

  int m = 5;
  int n = 4;

  int d = std::min(m,n);
  int k = d;  // number of singular values to find

  auto A = make_tensor<AType>({batch, m, n});
  auto U = make_tensor<AType>({batch, m, k});
  auto VT = make_tensor<AType>({batch, k, n});
  auto S = make_tensor<SType>({batch, k});

  // for correctness checking
  auto UD = make_tensor<AType>({batch, m, k});
  auto UDVT = make_tensor<AType>({batch, m, n});
  auto UUT = make_tensor<AType>({batch, m, m});
  auto UTU = make_tensor<AType>({batch, k, k});
  auto VVT = make_tensor<AType>({batch, n, n});
  auto VTV = make_tensor<AType>({batch, k, k});
  std::array<index_t, U.Rank()> Dshape;
  Dshape.fill(matxKeepDim);
  Dshape[U.Rank()-2] = m;
  // cloning D across
  auto D = clone<U.Rank()>(S, Dshape);

  int iterations = 10;
  {
    randomGenerator_t<AType> gen(A.TotalSize(),0);
    auto x0 = gen.GetTensorView({batch, d}, NORMAL);

    printf("iterations: %d\n", iterations);

    auto random = gen.GetTensorView({batch, m, n}, NORMAL);
    (A = random).run(stream);

    A.PrefetchDevice(stream);
    U.PrefetchDevice(stream);
    S.PrefetchDevice(stream);
    VT.PrefetchDevice(stream);

    (U = 0).run(stream);
    (S = 0).run(stream);
    (VT = 0).run(stream);

    svdpi(U, S, VT, A, x0, iterations, stream, k);

    cudaDeviceSynchronize();
    printf("svdpi:\n");

    printf("S\n");
    print(S);
    printf("U\n");
    print(U);
    printf("VT\n");
    print(VT);

    if( m <=  n) {
      printf("UUT:\n");
      matmul(UUT, U, conj(transpose(U)), stream);
      print(UUT);
    }

    printf("UTU:\n");
    matmul(UTU, conj(transpose(U)) , U, stream);
    print(UTU);

    if( n >= m) {
      printf("VVT:\n");
      matmul(VVT, conj(transpose(VT)), VT, stream);
      print(VVT);
    }

    printf("VTV:\n");
    matmul(VTV, VT, conj(transpose(VT)), stream); // works on r x r

    print(VTV);

    // scale U by eigen values (equivalent to matmul of the diagonal matrix)
    (UD = U * D).run(stream);

    matmul(UDVT, UD, VT, stream);

    printf("A\n");
    print(A);

    printf("UDVT\n");
    print(UDVT);

    (A = A - UDVT).run(stream);

    printf("A-UDVT\n");
    print(A);
  }
  // Same as above but with svdbpi
  {

    (U = 0).run(stream);
    (S = 0).run(stream);
    (VT = 0).run(stream);
    // TODO add k
    svdbpi(U, S, VT, A, iterations, stream);

    cudaDeviceSynchronize();
    printf("svdbpi:\n");

    printf("S\n");
    print(S);
    printf("U\n");
    print(U);
    printf("VT\n");
    print(VT);

    if( m <=  n) {
      printf("UUT:\n");
      matmul(UUT, U, conj(transpose(U)), stream);
      print(UUT);
    }

    printf("UTU:\n");
    matmul(UTU, conj(transpose(U)) , U, stream);
    print(UTU);

    if( n >= m) {
      printf("VVT:\n");
      matmul(VVT, conj(transpose(VT)), VT, stream);
      print(VVT);
    }

    printf("VTV:\n");
    matmul(VTV, VT, conj(transpose(VT)), stream); // works on r x r

    print(VTV);

    // scale U by eigen values (equivalent to matmul of the diagonal matrix)
    (UD = U * D).run(stream);

    matmul(UDVT, UD, VT, stream);

    printf("A\n");
    print(A);

    printf("UDVT\n");
    print(UDVT);

    (A = A - UDVT).run(stream);

    printf("A-UDVT\n");
    print(A);
  }

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
