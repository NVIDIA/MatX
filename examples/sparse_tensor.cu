////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include "matx.h"

// Note that sparse tensor support in MatX is still experimental.

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
  MATX_ENTER_HANDLER();

  cudaStream_t stream = 0;
  cudaExecutor exec{stream};

  //
  // Print some formats that are used for the universal sparse tensor
  // type. Note that common formats like COO and CSR have good library
  // support in e.g. cuSPARSE, but MatX provides a much more general
  // way to define the sparse tensor storage through a DSL (see doc).
  //
  experimental::Scalar::print(); // scalars
  experimental::SpVec::print();  // sparse vectors
  experimental::COO::print();    // various sparse matrix formats
  experimental::CSR::print();
  experimental::CSC::print();
  experimental::DCSR::print();
  experimental::DIAI::print();
  experimental::DIAJ::print();
  experimental::SkewDIAI::print();
  experimental::SkewDIAJ::print();
  experimental::BSR<2, 2>::print(); // 2x2 blocks
  experimental::COO4::print();      // 4-dim tensor in COO
  experimental::CSF5::print();      // 5-dim tensor in CSF

  //
  // Creates a 4x8 COO matrix for the following 4x8 dense matrix with 5 nonzero
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
  // format = ( d0, d1 ) -> ( d0 : compressed(non-unique), d1 : singleton )
  // space  = CUDA managed memory
  // nse    = 5
  // crd[0] = ( 0  0  3  3  3 )
  // crd[1] = ( 0  1  2  3  5 )
  // values = ( 1.0000e+00  2.0000e+00  3.0000e+00  4.0000e+00  5.0000e+00 )
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
  // performance critical code, since sparse storage formats do
  // not provide O(1) random access to their elements (compressed
  // levels will use some form of search to determine if an element
  // is present). Instead, conversions (and other operations) should
  // use sparse operations that are tailored for the sparse storage
  // format (such as scanning by row for CSR).
  //
  auto A1 = make_tensor<float>({4, 8});
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 8; j++) {
      A1(i, j) = Acoo(i, j);
    }
  }
  print(A1);

  //
  // A direct sparse2dense conversion. This is the correct way of
  // performing the conversion, since the underlying implementation
  // knows how to properly manipulate the sparse storage format.
  //
  auto A2 = make_tensor<float>({4, 8});
  (A2 = sparse2dense(Acoo)).run(exec);
  print(A2);

  //
  // Perform a direct SpMM. This is also the correct way of performing
  // an efficient sparse operation.
  //
  auto B = make_tensor<float, 2>({8, 4});
  auto C = make_tensor<float>({4, 4});
  B.SetVals({{0, 1, 2, 3},
             {4, 5, 6, 7},
             {8, 9, 10, 11},
             {12, 13, 14, 15},
             {16, 17, 18, 19},
             {20, 21, 22, 23},
             {24, 25, 26, 27},
             {28, 29, 30, 31}});
  (C = matmul(Acoo, B)).run(exec);
  print(C);

  //
  // Creates a 4x4 CSR matrix which is used to solve the following
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
  Y.SetVals({{5, 17}, {6, 18}, {12, 28}, {20, 40}});
  (X = solve(Acsr, Y)).run(exec);
  print(X);

  //
  // A direct dense2sparse conversion. This is the correct way of
  // performing an efficient sparse operation. Note, however,
  // that assigning a right-hand-side value to a sparse tensor
  // (viz. the lval Acoo) is an experimental operation recently
  // added to MatX, and it is currently restricted to a direct
  // "dense2sparse" operation at the right-hand-side.
  //
  auto D = make_tensor<float, 2>({4, 8});
  D.SetVals({{0, 11, 0, 12, 0, 0, 0, 0},
             {0, 0, 13, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 14},
             {0, 15, 0, 0, 16, 0, 17, 0}});
  (Acoo = dense2sparse(D)).run(exec);
  print(Acoo);

  //
  // Conversions between sparse formats: COO to CSR.
  // For speed-of-operation, the CSC output actually
  // shares some of the buffers with COO on completion.
  //
  auto Acsr2 = experimental::make_zero_tensor_csr<float, int, int>({4, 8});
  (Acsr2 = sparse2sparse(Acoo)).run(exec);
  print(Acsr2);

  //
  // Creates a 6x6 DIA matrix with 3 nonzero diagonals.
  //
  // |  4  1  0  0  0  0 |
  // | -1  4  1  0  0  0 |
  // |  0 -1  4  1  0  0 |
  // |  0  0 -1  4  1  0 |
  // |  0  0  0 -1  4  1 |
  // |  0  0  0  0 -1  4 |
  //
  auto dvals = make_tensor<float>({3 * 6});
  auto doffsets = make_tensor<int>({3});
  dvals.SetVals({-1, -1, -1, -1, -1, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 1});
  doffsets.SetVals({-1, 0, 1});
  auto AdiaJ = experimental::make_tensor_dia<experimental::DIA_INDEX_J>(dvals, doffsets, {6, 6});
  print(AdiaJ);

  //
  // Perform a direct SpMV. This is also the correct way of performing
  // an efficient sparse operation.
  //
  auto V = make_tensor<float>({6});
  auto R = make_tensor<float>({6});
  V.SetVals({1, 2, 3, 4, 5, 6});
  (R = matvec(AdiaJ, V)).run(exec);
  print(R);

  MATX_EXIT_HANDLER();
}
