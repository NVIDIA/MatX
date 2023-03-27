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

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  
  index_t iN = 4;
  index_t iM = 6;
 
  index_t fN = 4;
  index_t fM = 2;
  
  auto in = make_tensor<int>({iN,iM});
  auto filter = make_tensor<int>({fN,fM});
  
  in.SetVals({ {1,2,3,4,5,6},
               {5,4,3,2,1,0},
               {3,4,5,6,7,8},
               {1,2,3,4,5,6},
               });

  filter.SetVals({ {1,2}, 
                   {3,4},
                   {5,6},
                   {7,8}});

#if 1
  index_t oN = iN + fN -1;
  index_t oM = iM + fM -1;
  auto mode = MATX_C_MODE_FULL;
#elif 0
  index_t oN = iN;
  index_t oM = iM;
  auto mode = MATX_C_MODE_SAME;
#else
  index_t oN = iN - fN + 1;
  index_t oM = iM - fM + 1;
  auto mode = MATX_C_MODE_VALID;
#endif
  
  auto out = make_tensor<int>({oN,oM});
  
  conv2d(out, in, filter, mode, 0);

  printf("in:\n");
  print(in);
  printf("filter:\n");
  print(filter);
  printf("out:\n");
  print(out);

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
