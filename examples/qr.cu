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

  //using AType = double;
  using AType = cuda::std::complex<float>;
  
  cudaStream_t stream = 0;
  int batch = 1; 

  int m = 4;
  int n = 5;
 
  auto A = make_tensor<AType>({batch, m, n});
  auto QR = make_tensor<AType>({batch, m, n});
  auto QTQ = make_tensor<AType>({batch, m, m});
  auto Q = make_tensor<AType>({batch, m, m});
  auto R = make_tensor<AType>({batch, m, n});

  randomGenerator_t<AType> gen(A.TotalSize(),0);
  
  auto random = gen.GetTensorView(A.Shape(), NORMAL);
  (A = random).run(stream);

#if 0
  cudaDeviceSynchronize();
  A(0,0,0) = 10000; A(0,0,1) = 10001;
  A(0,1,0) = 10001; A(0,1,1) = 10002;
  A(0,2,0) = 10002; A(0,2,1) = 10003;
  A(0,3,0) = 10003; A(0,3,1) = 10004;
  A(0,4,0) = 10004; A(0,4,1) = 10005;
#endif

  A.PrefetchDevice(stream);
  Q.PrefetchDevice(stream);
  R.PrefetchDevice(stream);

  qr(Q, R, A, stream);

  matmul(QR, Q, R, stream);
  matmul(QTQ, conj(transpose(Q)), Q, stream);
  cudaDeviceSynchronize();
  
  printf("Q:\n"); print(Q);
  printf("R:\n"); print(R);
  printf("QTQ:\n"); print(QTQ);
  printf("QR:\n"); print(QR);
  printf("A:\n"); print(A);

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
