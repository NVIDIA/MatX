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
class BasicDiffTestsFloat : public ::testing::Test {
};

TYPED_TEST_SUITE(BasicDiffTestsFloat, MatXFloatNonComplexTypes);

TYPED_TEST(BasicDiffTestsFloat, Diffs)
{
  MATX_ENTER_HANDLER();

  int scale = 1;
  if constexpr (is_matx_half_v<TypeParam>) {
    scale = 1000;
  }
  int size = 100;  
  cudaStream_t stream = 0;

  tensor_t<TypeParam, 1> t1i{{size}};
  tensor_t<TypeParam, 1> t1o{{size}};
  
  for (index_t i = 0; i <t1i.Size(0); i++) {
    t1i(i) = TypeParam(i)/TypeParam(size)+TypeParam(1);
  }
  
  // derivative of a constant
  (t1o = deriv(t1i)).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     ASSERT_FLOAT_EQ(t1o(i), TypeParam(0));
  }

  // derivative d/dx(x) sampled at tensor t1i
  (t1o = deriv(wrt(t1i))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     ASSERT_FLOAT_EQ(t1o(i), TypeParam(1));
  }

  (t1o = deriv(pow(wrt(t1i),TypeParam(2)))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     ASSERT_NEAR(t1o(i), TypeParam(2)*x, .001*scale);
  }

  (t1o = deriv(pow(TypeParam(2),wrt(t1i)))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     ASSERT_NEAR(t1o(i), log(TypeParam(2))*pow(TypeParam(2),x),.001*scale);
  }

  (t1o = deriv(pow(wrt(t1i),wrt(t1i)))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     ASSERT_FLOAT_EQ(t1o(i), (log(x)+TypeParam(1))*pow(x,x));
  }
  
  (t1o = deriv(pow(sin(wrt(t1i)),TypeParam(2)))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     ASSERT_NEAR(t1o(i), TypeParam(2)*cos(x)*sin(x),.0001*scale);
  }
  
  (t1o = deriv(pow(wrt(t1i),sin(wrt(t1i))))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     auto s = sin(x);
     auto c = cos(x);
     auto l = log(x);

     ASSERT_NEAR(t1o(i), pow(x,sin(x)) * ( s/x + c*l ),.0001*scale);
  }
  
  auto y = TypeParam(1) + sin(wrt(t1i)) - sin(wrt(t1i));
  (t1o = deriv(pow(y,y))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     ASSERT_NEAR(t1o(i), TypeParam(0) ,.0001*scale);
  }

  (t1o = deriv(pow(sin(wrt(t1i))-sin(wrt(t1i))+TypeParam(1),sin(wrt(t1i))))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
    
    auto x = t1i(i);
    auto c = cos(x);
    auto s = sin(x);
    auto v = c*pow(s,s)*(log(s))+TypeParam(1);

    ASSERT_NEAR(t1o(i), TypeParam(0),.0001*scale);
  }

  (t1o = deriv(sin(wrt(t1i)))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     ASSERT_FLOAT_EQ(t1o(i), cos(x));
  }

  (t1o = deriv(sin(sin(wrt(t1i))))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     ASSERT_NEAR(t1o(i), cos(x)*cos(sin(x)),.00001*scale);
  }
  (t1o = deriv(sin(wrt(t1i))+cos(wrt(t1i)))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     ASSERT_NEAR(t1o(i), cos(t1i(i))-sin(t1i(i)),.00001*scale);
  }
  (t1o = deriv(sin(wrt(t1i))-cos(wrt(t1i)))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     ASSERT_NEAR(t1o(i), cos(t1i(i))+sin(t1i(i)),.00001*scale);
  }
  
  (t1o = deriv(sin(wrt(t1i))*cos(wrt(t1i)))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     ASSERT_NEAR(t1o(i), cos(t1i(i))*cos(t1i(i))-sin(t1i(i))*sin(t1i(i)), .00001*scale);
  }

  // Test disabled for half because the dynamic range is not sufficient to solve this in any meaningful way.
  if(!is_matx_half_v<TypeParam> ) { 
    (t1o = deriv(sin(wrt(t1i))/cos(wrt(t1i)))).run(stream);
    cudaStreamSynchronize(stream);
    for (index_t i = 0; i <t1i.Size(0); i++) {
     ASSERT_NEAR(t1o(i), (sin(t1i(i))*sin(t1i(i))) / (cos(t1i(i))*cos(t1i(i))) + TypeParam(1), .001*scale);
    }
  }
  
  (t1o = deriv(sin(cos(wrt(t1i)))*cos(sin(wrt(t1i))))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     auto s = sin(x);
     auto c = cos(x);
     auto ss = sin(sin(x));
     auto cc = cos(cos(x));
     auto sc = sin(cos(x));
     auto cs = cos(sin(x));

     ASSERT_NEAR(t1o(i), -c*sc*ss - s*cc*cs, .00001*scale);
  }
  
  (t1o = deriv(exp(sin(wrt(t1i))/sin(wrt(t1i))))).run(stream);
  cudaStreamSynchronize(stream);
  for (index_t i = 0; i <t1i.Size(0); i++) {
     auto x = t1i(i);
     auto s = sin(x);
     auto c = cos(x);
     auto s2 = s*s;
     auto c2 = c*c;

     ASSERT_NEAR(t1o(i), TypeParam(0), .0001*scale);
  }
  
  MATX_EXIT_HANDLER();
}

