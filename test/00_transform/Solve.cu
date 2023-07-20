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


template <typename TensorType>
class SolveTestsFloatNonComplexNonHalf : public ::testing::Test {
};

TYPED_TEST_SUITE(SolveTestsFloatNonComplexNonHalf, MatXFloatNonComplexNonHalfTypes);

TYPED_TEST(SolveTestsFloatNonComplexNonHalf, CGSolve)
{
  MATX_ENTER_HANDLER();

  int gN = 4;
  int N = gN * gN;
  int BATCH = 4;

  auto A = make_tensor<TypeParam, 3> ({BATCH, N, N}); 
  auto X = make_tensor<TypeParam, 2> ({BATCH, N}); 
  auto B = make_tensor<TypeParam, 2> ({BATCH, N}); 


  // Simple 1D Poisson matrix
  for(int b = 0; b < BATCH; b++) {
    for(int i = 0; i < N; i++) {
      X(b,i) = TypeParam(0+b);
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

  // example-begin cgsolve-test-1
  (X = cgsolve(A, B, .00001, 10)).run();
  // example-end cgsolve-test-1
  matvec(B, A, X);
  cudaDeviceSynchronize();

  for(int i = 0; i < BATCH; i++) {
    for(int j = 0; j < N; j++) {
      ASSERT_NEAR(B(i,j), TypeParam(1+i), .0001);
    }
  }
  MATX_EXIT_HANDLER();
}

