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

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  cudaStream_t stream = 0;
  cudaExecutor exec{stream};

  //
  // Creates a COO matrix for the following 4x8 dense matrix with 5 nonzero
  // elements, using the factory method that uses MatX tensors for the 1-dim
  // buffers. The sparse matrix resides in the same memory space as its buffer
  // constituents.
  //
  //   | 1, 2, 0, 0, 0, 0, 0, 0 |
  //   | 0, 0, 0, 0, 0, 0, 0, 0 |
  //   | 0, 0, 0, 0, 0, 0, 0, 0 |
  //   | 0, 0, 3, 4, 0, 5, 0, 0 |
  //
  
  constexpr index_t m = 4;
  constexpr index_t n = 8;
  constexpr index_t nse = 5;

  tensor_t<float, 1> values{{nse}};
  tensor_t<int, 1> row_idx{{nse}};
  tensor_t<int, 1> col_idx{{nse}};

  values.SetVals({ 1, 2, 3, 4, 5 });
  row_idx.SetVals({ 0, 0, 3, 3, 3 });
  col_idx.SetVals({ 0, 1, 2, 3, 5 });

  // Note that sparse tensor support in MatX is still experimental.
  auto Acoo = experimental::make_tensor_coo(values, row_idx, col_idx, {m, n});

  //
  // This shows:
  //
  // tensor_impl_2_f32: Tensor{float} Rank: 2, Sizes:[4, 8], Levels:[4, 8]
  // nse    = 5
  // format = ( d0, d1 ) -> ( d0 : compressed(non-unique), d1 : singleton )
  // crd[0] = ( 0  0  3  3  3 )
  // crd[1] = ( 0  1  2  3  5 )
  // values = ( 1.0000e+00  2.0000e+00  3.0000e+00  4.0000e+00  5.0000e+00 )
  // space  = CUDA managed memory
  //
  print(Acoo);

  // TODO: operations on Acoo

  MATX_EXIT_HANDLER();
}
