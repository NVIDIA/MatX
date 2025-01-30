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

#pragma once

#include "matx/core/sparse_tensor.h"

namespace matx {
namespace experimental {

//
// MatX uses a single versatile sparse tensor type that uses a tensor format
// DSL (Domain Specific Language) to describe a vast space of storage formats.
// This file provides a number of convenience factory methods that construct
// sparse tensors in well-known storage formats, like COO, CSR, and CSC,
// directly from the constituent buffers. More factory methods can easily be
// added as the need arises.
//

// Constructs a sparse matrix in COO format directly from the values and
// the two coordinates vectors. The entries should be sorted by row, then
// column. Duplicate entries should not occur. Explicit zeros may be stored.
template <typename ValTensor, typename CrdTensor>
auto make_tensor_coo(ValTensor &val, CrdTensor &row, CrdTensor &col,
                     const index_t (&shape)[2], bool owning = false) {
  static_assert(ValTensor::Rank() == 1 && CrdTensor::Rank() == 1);
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = index_t;
  // Note that the COO API typically does not involve positions.
  // However, under the formal DSL specifications, the top level
  // compression should set up pos[0] = {0, nse}. This is done
  // here, using the same memory space as the other data.
  POS *ptr;
  matxMemorySpace_t space = GetPointerKind(val.GetStorage().data());
  matxAlloc((void **)&ptr, 2 * sizeof(POS), space, 0);
  ptr[0] = 0;
  ptr[1] = val.Size(0);
  raw_pointer_buffer<POS, matx_allocator<POS>> topp{ptr, 2 * sizeof(POS),
                                                    owning};
  basic_storage<decltype(topp)> tp{std::move(topp)};
  raw_pointer_buffer<POS, matx_allocator<POS>> emptyp{nullptr, 0, owning};
  basic_storage<decltype(emptyp)> ep{std::move(emptyp)};
  return sparse_tensor_t<VAL, CRD, POS, COO>(
      shape, val.GetStorage(), {row.GetStorage(), col.GetStorage()}, {tp, ep});
}

// Constructs a sparse matrix in CSR format directly from the values, the row
// positions, and column coordinates vectors. The entries should be sorted by
// row, then column. Explicit zeros may be stored.
template <typename ValTensor, typename PosTensor, typename CrdTensor>
auto make_tensor_csr(ValTensor &val, PosTensor &rowp, CrdTensor &col,
                     const index_t (&shape)[2], bool owning = false) {
  static_assert(ValTensor::Rank() == 1 && CrdTensor::Rank() == 1 &&
                PosTensor::Rank() == 1);
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = typename PosTensor::value_type;
  raw_pointer_buffer<CRD, matx_allocator<CRD>> emptyc{nullptr, 0, owning};
  basic_storage<decltype(emptyc)> ec{std::move(emptyc)};
  raw_pointer_buffer<POS, matx_allocator<POS>> emptyp{nullptr, 0, owning};
  basic_storage<decltype(emptyp)> ep{std::move(emptyp)};
  return sparse_tensor_t<VAL, CRD, POS, CSR>(
      shape, val.GetStorage(), {ec, col.GetStorage()}, {ep, rowp.GetStorage()});
}

// Constructs a sparse matrix in CSC format directly from the values,
// the row coordinates, and column position vectors. The entries should
// be sorted by column, then row. Explicit zeros may be stored.
template <typename ValTensor, typename PosTensor, typename CrdTensor>
auto make_tensor_csc(ValTensor &val, CrdTensor &row, PosTensor &colp,
                     const index_t (&shape)[2], bool owning = false) {
  static_assert(ValTensor::Rank() == 1 && CrdTensor::Rank() == 1 &&
                PosTensor::Rank() == 1);
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = typename PosTensor::value_type;
  raw_pointer_buffer<CRD, matx_allocator<CRD>> emptyc{nullptr, 0, owning};
  basic_storage<decltype(emptyc)> ec{std::move(emptyc)};
  raw_pointer_buffer<POS, matx_allocator<POS>> emptyp{nullptr, 0, owning};
  basic_storage<decltype(emptyp)> ep{std::move(emptyp)};
  return sparse_tensor_t<VAL, CRD, POS, CSC>(
      shape, val.GetStorage(), {ec, row.GetStorage()}, {ep, colp.GetStorage()});
}

} // namespace experimental
} // namespace matx
