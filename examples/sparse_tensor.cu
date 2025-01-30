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
  // Print some formats that are used for the versatile sparse tensor
  // type. Note that common formats like COO and CSR have good library
  // support in e.g. cuSPARSE, but MatX provides a much more general
  // way to define the sparse tensor storage through a DSL (see doc).
  //
  experimental::Scalar::print();   // scalars
  experimental::SpVec::print();    // sparse vectors
  experimental::COO::print();      // various sparse matrix formats
  experimental::CSR::print();
  experimental::CSC::print();
  experimental::DCSR::print();
  experimental::BSR<2,2>::print(); // 2x2 blocks
  experimental::COO4::print();     // 4-dim tensor in COO
  experimental::CSF5::print();     // 5-dim tensor in CSF

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
  // tensor_impl_2_f32: SparseTensor{float} Rank: 2, Sizes:[4, 8], Levels:[4, 8]
  // nse    = 5
  // format = ( d0, d1 ) -> ( d0 : compressed(non-unique), d1 : singleton )
  // crd[0] = ( 0  0  3  3  3 )
  // crd[1] = ( 0  1  2  3  5 )
  // values = ( 1.0000e+00  2.0000e+00  3.0000e+00  4.0000e+00  5.0000e+00 )
  // space  = CUDA managed memory
  //
  print(Acoo);

  //
  // A very naive way to convert the sparse matrix back to a dense
  // matrix. Note that one should **never** use the ()-operator in
  // performance critical code, since sparse data structures do
  // not provide O(1) random access to their elements (compressed
  // levels will use some form of search to determine if an element
  // is present). Instead, conversions (and other operations) should
  // use sparse operations that are tailored for the sparse data
  // structure (such as scanning by row for CSR).
  //
  tensor_t<float, 2> Dense{{m, n}};
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      Dense(i, j) = Acoo(i, j);
    }
  }
  print(Dense);

  // TODO: operations on Acoo

  MATX_EXIT_HANDLER();
}
