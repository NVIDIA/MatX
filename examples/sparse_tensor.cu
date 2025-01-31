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

// Note that sparse tensor support in MatX is still experimental.

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  cudaStream_t stream = 0;
  cudaExecutor exec{stream};

  //
  // Print some formats that are used for the universal sparse tensor
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
  // tensor_impl_2_f32: SparseTensor{float} Rank: 2, Sizes:[4, 8], Levels:[4, 8]
  // nse    = 5
  // format = ( d0, d1 ) -> ( d0 : compressed(non-unique), d1 : singleton )
  // crd[0] = ( 0  0  3  3  3 )
  // crd[1] = ( 0  1  2  3  5 )
  // values = ( 1.0000e+00  2.0000e+00  3.0000e+00  4.0000e+00  5.0000e+00 )
  // space  = CUDA managed memory
  //
  auto vals = make_tensor<float>({5});
  auto idxi = make_tensor<int>({5}); 
  auto idxj = make_tensor<int>({5});
  vals.SetVals({1, 2, 3, 4, 5});
  idxi.SetVals({0, 0, 3, 3, 3});
  idxj.SetVals({0, 1, 2, 3, 5});
  auto Acoo = experimental::make_tensor_coo(vals, idxi, idxj, {4, 8});
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
  auto A = make_tensor<float>({4, 8});
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 8; j++) {
      A(i, j) = Acoo(i, j);
    }
  }
  print(A);

  //
  // SpMM is implemented on COO through cuSPARSE. This is the
  // correct way of performing an efficient sparse operation.
  //
  auto B = make_tensor<float, 2>({8, 4});
  auto C = make_tensor<float>({4, 4});
  B.SetVals({
    { 0,  1,  2,  3}, { 4,  5,  6,  7}, { 8,  9, 10, 11}, {12, 13, 14, 15},
    {16, 17, 18, 19}, {20, 21, 22, 23}, {24, 25, 26, 27}, {28, 29, 30, 31} });
  (C = matmul(Acoo, B)).run(exec);
  print(C);

  //
  // Creates a CSR matrix which is used to solve the following
  // system of equations AX=Y, where X is the unknown.
  //
  // | 1 2 0 0 |   | 1 5 |   |  5 17 |
  // | 0 3 0 0 | x | 2 6 | = |  6 18 |
  // | 0 0 4 0 |   | 3 7 |   | 12 28 |
  // | 0 0 0 5 |   | 4 8 |   | 20 40 |
  //
  auto coeffs = make_tensor<float>({5});
  auto rowptr = make_tensor<int>({5});
  auto colidx = make_tensor<int>({5});
  coeffs.SetVals({1, 2, 3, 4, 5});
  rowptr.SetVals({0, 2, 3, 4, 5});
  colidx.SetVals({0, 1, 1, 2, 3});
  auto Acsr = experimental::make_tensor_csr(coeffs, rowptr, colidx, {4, 4});
  print(Acsr);
  auto X = make_tensor<float>({4, 2});
  auto Y = make_tensor<float>({4, 2});
  Y.SetVals({ {5, 17}, {6, 18}, {12, 28}, {20, 40} });
  (X = solve(Acsr, Y)).run(exec);
  print(X);

  MATX_EXIT_HANDLER();
}
