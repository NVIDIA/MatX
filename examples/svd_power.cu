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

#if 1
  //using AType = float;
  using AType = cuda::std::complex<float>;
  using SType = float;

  cudaStream_t stream = 0;
  cudaExecutor exec{stream};

  int m = 5;
  int n = 4;

  int d = std::min(m,n);
  int k = d;  // number of singular values to find

#if 0
  int batch = 1; 
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
  auto x0 = random<float>({batch, d}, NORMAL);
    
  (A = random<float>({batch, m, n}, NORMAL)).run(exec);

#else
  auto A = make_tensor<AType>({m, n});
  auto U = make_tensor<AType>({m, k});
  auto VT = make_tensor<AType>({k, n});
  auto S = make_tensor<SType>({k});

  // for correctness checking
  auto UD = make_tensor<AType>({m, k});
  auto UDVT = make_tensor<AType>({m, n});
  auto UUT = make_tensor<AType>({m, m});
  auto UTU = make_tensor<AType>({k, k});
  auto VVT = make_tensor<AType>({n, n});
  auto VTV = make_tensor<AType>({k, k});
  auto x0 = random<float>({d}, NORMAL);
  
  (A = random<float>({m, n}, NORMAL)).run(exec);

#endif
  cuda::std::array<index_t, U.Rank()> Dshape;
  Dshape.fill(matxKeepDim);
  Dshape[U.Rank()-2] = m;
  // cloning D across
  auto D = clone<U.Rank()>(S, Dshape);

  float tol = (float)1e-3;
  int iterations = 20;
  
  {

    printf("iterations: %d\n", iterations);

    (U = 0).run(exec);
    (S = 0).run(exec);
    (VT = 0).run(exec);

    (mtie(U, S, VT) = svdpi(A, x0, iterations, k)).run(exec);

    exec.sync();
    printf("svdpi:\n");

    printf("S\n");
    print(S);
    printf("U\n");
    print(U);
    printf("VT\n");
    print(VT);

    if( m <=  n) {
      printf("UUT:\n");
      (UUT = matmul(U, conj(transpose_matrix(U)))).run(exec);
      print(UUT);
    }

    printf("UTU:\n");
    (UTU = matmul(conj(transpose_matrix(U)), U)).run(exec);
    print(UTU);

    if( n >= m) {
      printf("VVT:\n");
      (VVT = matmul(conj(transpose_matrix(VT)), VT)).run(exec);
      print(VVT);
    }

    printf("VTV:\n");
    (VTV = matmul(VT, conj(transpose_matrix(VT)))).run(exec); // works on r x r

    print(VTV);

    // scale U by eigen values (equivalent to matmul of the diagonal matrix)
    (UD = U * D).run(exec);

    (UDVT = matmul(UD, VT)).run(exec);

    printf("A\n");
    print(A);

    printf("UDVT\n");
    print(UDVT);

    (A = A - UDVT).run(exec);

    printf("A-UDVT\n");
    print(A);
  }
  
  // Same as above but with svdbpi
  {

    (U = 0).run(exec);
    (S = 0).run(exec);
    (VT = 0).run(exec);
    // TODO add k
    (mtie(U, S, VT) = svdbpi(A, iterations, tol)).run(exec);

    exec.sync();
    printf("svdbpi:\n");

    printf("S\n");
    print(S);
    printf("U\n");
    print(U);
    printf("VT\n");
    print(VT);

    if( m <=  n) {
      printf("UUT:\n");
      (UUT = matmul(U, conj(transpose_matrix(U)))).run(exec);
      print(UUT);
    }

    printf("UTU:\n");
    (UTU = matmul(conj(transpose_matrix(U)), U)).run(exec);
    print(UTU);

    if( n >= m) {
      printf("VVT:\n");
      (VVT = matmul(conj(transpose_matrix(VT)), VT)).run(exec);
      print(VVT);
    }

    printf("VTV:\n");
    (VTV = matmul(VT, conj(transpose_matrix(VT)))).run(exec); // works on r x r

    print(VTV);

    // scale U by eigen values (equivalent to matmul of the diagonal matrix)
    (UD = U * D).run(exec);

    (UDVT = matmul(UD, VT)).run(exec);

    printf("A\n");
    print(A);

    printf("UDVT\n");
    print(UDVT);

    (A = A - UDVT).run(exec);

    printf("A-UDVT\n");
    print(A);
  }
#endif
  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
