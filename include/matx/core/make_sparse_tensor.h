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

// Helper method to create zero storage.
template <typename T>
__MATX_INLINE__ static auto
makeDefaultNonOwningZeroStorage(index_t sz, matxMemorySpace_t space) {
  T *ptr;
  matxAlloc((void **)&ptr, sz * sizeof(T), space, 0);
  for (index_t i = 0; i < sz; i++) {
    ptr[i] = 0;
  }
  raw_pointer_buffer<T, matx_allocator<T>> buf{ptr, sz * sizeof(T),
                                               /*owning=*/false};
  return basic_storage<decltype(buf)>{std::move(buf)};
}

// Helper method to create empty storage.
template <typename T>
__MATX_INLINE__ static auto makeDefaultNonOwningEmptyStorage() {
  raw_pointer_buffer<T, matx_allocator<T>> buf{nullptr, 0, /*owning=*/false};
  return basic_storage<decltype(buf)>{std::move(buf)};
}

//
// MatX implements a universal sparse tensor type that uses a tensor format
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
                     const index_t (&shape)[2]) {
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = index_t;
  // Proper structure.
  MATX_STATIC_ASSERT_STR(val.Rank() == 1 && row.Rank() == 1 && col.Rank() == 1,
                         matxInvalidParameter, "data arrays should be rank-1");
  MATX_ASSERT_STR(val.Size(0) == row.Size(0) && val.Size(0) == col.Size(0),
                  matxInvalidParameter,
                  "data arrays should have consistent length (nse)");
  // Note that the COO API typically does not involve positions.
  // However, under the formal DSL specifications, the top level
  // compression should set up pos[0] = {0, nse}. This is done
  // here, using the same memory space as the other data.
  matxMemorySpace_t space = GetPointerKind(val.GetStorage().data());
  auto tp = makeDefaultNonOwningZeroStorage<POS>(2, space);
  tp.data()[1] = val.Size(0);
  // Construct COO.
  return sparse_tensor_t<VAL, CRD, POS, COO>(
      shape, val.GetStorage(), {row.GetStorage(), col.GetStorage()},
      {tp, makeDefaultNonOwningEmptyStorage<POS>()});
}

// Constructs a zero sparse matrix in COO format (viz. nse=0).
template <typename VAL, typename CRD, typename POS = index_t>
auto make_zero_tensor_coo(const index_t (&shape)[2],
                          matxMemorySpace_t space = MATX_MANAGED_MEMORY) {
  return sparse_tensor_t<VAL, CRD, POS, COO>(
      shape, makeDefaultNonOwningEmptyStorage<VAL>(),
      {makeDefaultNonOwningEmptyStorage<CRD>(),
       makeDefaultNonOwningEmptyStorage<CRD>()},
      {makeDefaultNonOwningZeroStorage<POS>(2, space),
       makeDefaultNonOwningEmptyStorage<POS>()});
}

// Constructs a sparse matrix in CSR format directly from the values, the
// row positions, and column coordinates vectors. The entries should be
// sorted by row, then column. Explicit zeros may be stored. Duplicate
// entries should not occur. Explicit zeros may be stored.
template <typename ValTensor, typename PosTensor, typename CrdTensor>
auto make_tensor_csr(ValTensor &val, PosTensor &rowp, CrdTensor &col,
                     const index_t (&shape)[2]) {
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = typename PosTensor::value_type;
  // Proper structure.
  MATX_STATIC_ASSERT_STR(val.Rank() == 1 && rowp.Rank() == 1 && col.Rank() == 1,
                         matxInvalidParameter, "data arrays should be rank-1");
  MATX_ASSERT_STR(rowp.Size(0) == shape[0] + 1, matxInvalidParameter,
                  "row positions arrays should have length #rows + 1");
  MATX_ASSERT_STR(val.Size(0) == col.Size(0), matxInvalidParameter,
                  "data arrays should have consistent length (nse)");
  // Construct CSR.
  return sparse_tensor_t<VAL, CRD, POS, CSR>(
      shape, val.GetStorage(),
      {makeDefaultNonOwningEmptyStorage<CRD>(), col.GetStorage()},
      {makeDefaultNonOwningEmptyStorage<POS>(), rowp.GetStorage()});
}

// Constructs a zero sparse matrix in CSR format (viz. nse=0).
template <typename VAL, typename CRD, typename POS>
auto make_zero_tensor_csr(const index_t (&shape)[2],
                          matxMemorySpace_t space = MATX_MANAGED_MEMORY) {
  return sparse_tensor_t<VAL, CRD, POS, CSR>(
      shape, makeDefaultNonOwningEmptyStorage<VAL>(),
      {makeDefaultNonOwningEmptyStorage<CRD>(),
       makeDefaultNonOwningEmptyStorage<CRD>()},
      {makeDefaultNonOwningEmptyStorage<POS>(),
       makeDefaultNonOwningZeroStorage<POS>(shape[0] + 1, space)});
}

// Constructs a sparse matrix in CSC format directly from the values, the
// column positions, and row coordinates vectors. The entries should be
// sorted by columns, then row. Explicit zeros may be stored. Duplicate
// entries should not occur. Explicit zeros may be stored.
template <typename ValTensor, typename PosTensor, typename CrdTensor>
auto make_tensor_csc(ValTensor &val, PosTensor &colp, CrdTensor &row,
                     const index_t (&shape)[2]) {
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = typename PosTensor::value_type;
  // Proper structure.
  MATX_STATIC_ASSERT_STR(val.Rank() == 1 && row.Rank() == 1 && colp.Rank() == 1,
                         matxInvalidParameter, "data arrays should be rank-1");
  MATX_ASSERT_STR(colp.Size(0) == shape[1] + 1, matxInvalidParameter,
                  "column positions array should have length #columns + 1");
  MATX_ASSERT_STR(val.Size(0) == row.Size(0), matxInvalidParameter,
                  "data arrays should have consistent length (nse)");
  // Construct CSC.
  return sparse_tensor_t<VAL, CRD, POS, CSC>(
      shape, val.GetStorage(),
      {makeDefaultNonOwningEmptyStorage<CRD>(), row.GetStorage()},
      {makeDefaultNonOwningEmptyStorage<POS>(), colp.GetStorage()});
}

// Constructs a zero sparse matrix in CSC format (viz. nse=0).
template <typename VAL, typename CRD, typename POS>
auto make_zero_tensor_csc(const index_t (&shape)[2],
                          matxMemorySpace_t space = MATX_MANAGED_MEMORY) {
  return sparse_tensor_t<VAL, CRD, POS, CSC>(
      shape, makeDefaultNonOwningEmptyStorage<VAL>(),
      {makeDefaultNonOwningEmptyStorage<CRD>(),
       makeDefaultNonOwningEmptyStorage<CRD>()},
      {makeDefaultNonOwningEmptyStorage<POS>(),
       makeDefaultNonOwningZeroStorage<POS>(shape[1] + 1, space)});
}

} // namespace experimental
} // namespace matx
