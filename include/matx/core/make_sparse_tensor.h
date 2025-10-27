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
#pragma once

#include "matx/core/sparse_tensor.h"

namespace matx {
namespace experimental {

// Helper method to zero memory.
template <typename T>
__MATX_INLINE__ static void setZero(T *ptr, index_t sz,
                                    matxMemorySpace_t space) {
  if (space == MATX_DEVICE_MEMORY || space == MATX_ASYNC_DEVICE_MEMORY) {
    cudaMemset(ptr, 0, sz * sizeof(T));
  } else {
    memset(ptr, 0, sz * sizeof(T));
  }
}

// Helper to create a Storage<T> with zeros
template <typename T>
__MATX_INLINE__ Storage<T> makeZeroStorage(index_t sz, matxMemorySpace_t space) {
  assert(sz > 0);
  Storage<T> storage(sz, space);
  if (storage.data() != nullptr) {
    setZero(storage.data(), sz, space);
  }
  return storage;
}

// Helper to create an empty Storage<T>
template <typename T>
__MATX_INLINE__ Storage<T> makeEmptyStorage() {
  return Storage<T>();
}



// Helper method to fill memory.
template <typename T>
__MATX_INLINE__ static void setVal(T *ptr, T val, matxMemorySpace_t space) {
  if (space == MATX_DEVICE_MEMORY || space == MATX_ASYNC_DEVICE_MEMORY) {
    cudaMemcpy(ptr, &val, sizeof(T), cudaMemcpyHostToDevice);
  } else {
    memcpy(ptr, &val, sizeof(T));
  }
}


//
// MatX implements a universal sparse tensor type that uses a tensor format
// DSL (Domain Specific Language) to describe a vast space of storage formats.
// This file provides a number of convenience factory methods that construct
// sparse tensors in well-known storage formats, like COO, CSR, CSC, and DIA,
// directly from the constituent buffers. More factory methods can easily be
// added as the need arises.
//

// Indexing options.
struct DIA_INDEX_I {};
struct DIA_INDEX_J {};

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
  MATX_STATIC_ASSERT_STR(ValTensor::Rank() == 1 && CrdTensor::Rank() == 1,
                         matxInvalidParameter, "data arrays should be rank-1");
  MATX_ASSERT_STR(val.Size(0) == row.Size(0) && val.Size(0) == col.Size(0),
                  matxInvalidParameter,
                  "data arrays should have consistent length (nse)");
  // Note that the COO API typically does not involve positions.
  // However, under the formal DSL specifications, the top level
  // compression should set up pos[0] = {0, nse}. This is done
  // here, using the same memory space as the other data.
  matxMemorySpace_t space = GetPointerKind(val.GetStorage().data());
  Storage<POS> tp = makeZeroStorage<POS>(2, space);
  setVal(tp.data() + 1, static_cast<POS>(val.Size(0)), space);
  // Construct COO.
  return sparse_tensor_t<VAL, CRD, POS, COO>(
      shape, val.GetStorage(), {row.GetStorage(), col.GetStorage()},
      {std::move(tp), makeEmptyStorage<POS>()});
}

// Constructs a zero sparse matrix in COO format (viz. nse=0).
template <typename VAL, typename CRD, typename POS = index_t>
auto make_zero_tensor_coo(const index_t (&shape)[2],
                          matxMemorySpace_t space = MATX_MANAGED_MEMORY) {
  return sparse_tensor_t<VAL, CRD, POS, COO>(
      shape, makeEmptyStorage<VAL>(),
      {makeEmptyStorage<CRD>(),
       makeEmptyStorage<CRD>()},
      {makeZeroStorage<POS>(2, space),
       makeEmptyStorage<POS>()});
}

// Constructs a sparse matrix in CSR format directly from the values, the
// row positions, and column coordinates vectors. The entries should be
// sorted by row, then column. Duplicate entries should not occur. Explicit
// zeros may be stored.
template <typename ValTensor, typename PosTensor, typename CrdTensor>
auto make_tensor_csr(ValTensor &val, PosTensor &rowp, CrdTensor &col,
                     const index_t (&shape)[2]) {
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = typename PosTensor::value_type;
  // Proper structure.
  MATX_STATIC_ASSERT_STR(ValTensor::Rank() == 1 && PosTensor::Rank() == 1 &&
                             CrdTensor::Rank() == 1,
                         matxInvalidParameter, "data arrays should be rank-1");
  MATX_ASSERT_STR(rowp.Size(0) == shape[0] + 1, matxInvalidParameter,
                  "row positions arrays should have length #rows + 1");
  MATX_ASSERT_STR(val.Size(0) == col.Size(0), matxInvalidParameter,
                  "data arrays should have consistent length (nse)");
  // Construct CSR.
  return sparse_tensor_t<VAL, CRD, POS, CSR>(
      shape, val.GetStorage(),
      {makeEmptyStorage<CRD>(), col.GetStorage()},
      {makeEmptyStorage<POS>(), rowp.GetStorage()});
}

// Constructs a zero sparse matrix in CSR format (viz. nse=0).
template <typename VAL, typename CRD, typename POS>
auto make_zero_tensor_csr(const index_t (&shape)[2],
                          matxMemorySpace_t space = MATX_MANAGED_MEMORY) {
  return sparse_tensor_t<VAL, CRD, POS, CSR>(
      shape, makeEmptyStorage<VAL>(),
      {makeEmptyStorage<CRD>(),
       makeEmptyStorage<CRD>()},
      {makeEmptyStorage<POS>(),
       makeZeroStorage<POS>(shape[0] + 1, space)});
}

// Constructs a sparse matrix in CSC format directly from the values, the
// column positions, and row coordinates vectors. The entries should be
// sorted by columns, then row. Duplicate entries should not occur. Explicit
// zeros may be stored.
template <typename ValTensor, typename PosTensor, typename CrdTensor>
auto make_tensor_csc(ValTensor &val, PosTensor &colp, CrdTensor &row,
                     const index_t (&shape)[2]) {
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = typename PosTensor::value_type;
  // Proper structure.
  MATX_STATIC_ASSERT_STR(ValTensor::Rank() == 1 && PosTensor::Rank() == 1 &&
                             CrdTensor::Rank() == 1,
                         matxInvalidParameter, "data arrays should be rank-1");
  MATX_ASSERT_STR(colp.Size(0) == shape[1] + 1, matxInvalidParameter,
                  "column positions array should have length #columns + 1");
  MATX_ASSERT_STR(val.Size(0) == row.Size(0), matxInvalidParameter,
                  "data arrays should have consistent length (nse)");
  // Construct CSC.
  return sparse_tensor_t<VAL, CRD, POS, CSC>(
      shape, val.GetStorage(),
      {makeEmptyStorage<CRD>(), row.GetStorage()},
      {makeEmptyStorage<POS>(), colp.GetStorage()});
}

// Constructs a zero sparse matrix in CSC format (viz. nse=0).
template <typename VAL, typename CRD, typename POS>
auto make_zero_tensor_csc(const index_t (&shape)[2],
                          matxMemorySpace_t space = MATX_MANAGED_MEMORY) {
  return sparse_tensor_t<VAL, CRD, POS, CSC>(
      shape, makeEmptyStorage<VAL>(),
      {makeEmptyStorage<CRD>(),
       makeEmptyStorage<CRD>()},
      {makeEmptyStorage<POS>(),
       makeZeroStorage<POS>(shape[1] + 1, space)});
}

// Constructs a sparse matrix in DIA format directly from the values and the
// offset vectors. For an m x n matrix, this format uses a linearized storage
// where each diagonal has m or n entries and is accessed by either index I or
// index J, respectively. For index I, diagonals are padded with zeros on the
// left for the lower triangular part and padded with zeros on the right for
// the upper triagonal part. This is vv. when using index J. This format is
// most efficient for matrices with only a few nonzero diagonals that are
// close to the main diagonal.
template <typename IDX, typename ValTensor, typename CrdTensor>
auto make_tensor_dia(ValTensor &val, CrdTensor &off,
                     const index_t (&shape)[2]) {
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = index_t;
  // Proper structure.
  MATX_STATIC_ASSERT_STR(ValTensor::Rank() == 1 && CrdTensor::Rank() == 1,
                         matxInvalidParameter, "data arrays should be rank-1");
  if constexpr (std::is_same_v<IDX, DIA_INDEX_I>) {
    MATX_ASSERT_STR(val.Size(0) == shape[0] * off.Size(0), matxInvalidParameter,
                    "data arrays should contain all diagonals (by row index)");
  } else {
    MATX_ASSERT_STR(val.Size(0) == shape[1] * off.Size(0), matxInvalidParameter,
                    "data arrays should contain all diagonals (by col index)");
  }
  // Note that the DIA API typically does not involve positions.
  // However, under the formal DSL specifications, the top level
  // compression should set up pos[0] = {0, #diags}. This is done
  // here, using the same memory space as the other data.
  matxMemorySpace_t space = GetPointerKind(val.GetStorage().data());
  Storage<POS> tp = makeZeroStorage<POS>(2, space);
  setVal(tp.data() + 1, static_cast<POS>(off.Size(0)), space);
  // Construct DIA-I/J.
  using DIA = std::conditional_t<std::is_same_v<IDX, DIA_INDEX_I>, DIAI, DIAJ>;
  return sparse_tensor_t<VAL, CRD, POS, DIA>(
      shape, val.GetStorage(),
      {off.GetStorage(), makeEmptyStorage<CRD>()},
      {std::move(tp), makeEmptyStorage<POS>()});
}

// Constructs a sparse tensor in uniform batched DIA format directly from
// the values and the offset vectors. For a b x m x n tensor, this format
// effectively stores b times m x n matrices in DIA format, using a uniform
// nonzero structure for each (non-uniform formats are possible as well).
// All diagonals are stored consecutively in linearized format, sorted lower
// to upper, with all diagonals at a certain offset appearing consecutively
// for all batches. With DIA(b,i,j) as indexing, can be indexed by i or j.
template <typename IDX, typename ValTensor, typename CrdTensor>
auto make_tensor_uniform_batched_dia(ValTensor &val, CrdTensor &off,
                                     const index_t (&shape)[3]) {
  using VAL = typename ValTensor::value_type;
  using CRD = typename CrdTensor::value_type;
  using POS = index_t;
  // Proper structure.
  MATX_STATIC_ASSERT_STR(ValTensor::Rank() == 1 && CrdTensor::Rank() == 1,
                         matxInvalidParameter, "data arrays should be rank-1");
  if constexpr (std::is_same_v<IDX, DIA_INDEX_I>) {
    MATX_ASSERT_STR(val.Size(0) == shape[0] * shape[1] * off.Size(0),
                    matxInvalidParameter,
                    "data arrays should contain all diagonals (by row index)");
  } else {
    MATX_ASSERT_STR(val.Size(0) == shape[0] * shape[2] * off.Size(0),
                    matxInvalidParameter,
                    "data arrays should contain all diagonals (by col index)");
  }
  // Note that the DIA API typically does not involve positions.
  // However, under the formal DSL specifications, the top level
  // compression should set up pos[0] = {0, #diags}. This is done
  // here, using the same memory space as the other data.
  matxMemorySpace_t space = GetPointerKind(val.GetStorage().data());
  Storage<POS> tp = makeZeroStorage<POS>(2, space);
  setVal(tp.data() + 1, static_cast<POS>(off.Size(0)), space);
  // Construct Batched DIA-I/J.
  using DIA = std::conditional_t<std::is_same_v<IDX, DIA_INDEX_I>,
                                 BatchedDIAIUniform, BatchedDIAJUniform>;
  return sparse_tensor_t<VAL, CRD, POS, DIA>(
      shape, val.GetStorage(),
      {off.GetStorage(), makeEmptyStorage<CRD>()},
      {std::move(tp), makeEmptyStorage<POS>()});
}

// Convenience constructor for uniform batched tri-diagonal storage.
template <typename IDX, typename ValTensor>
auto make_tensor_uniform_batched_tri_dia(ValTensor &val,
                                         const index_t (&shape)[3]) {
  using VAL = typename ValTensor::value_type;
  using CRD = index_t;
  using POS = index_t;
  // Proper structure.
  MATX_STATIC_ASSERT_STR(ValTensor::Rank() == 1, matxInvalidParameter,
                         "data array should be rank-1");
  if constexpr (std::is_same_v<IDX, DIA_INDEX_I>) {
    MATX_ASSERT_STR(
        val.Size(0) == shape[0] * shape[1] * 3, matxInvalidParameter,
        "data arrays should contain all three diagonals (by row index)");
  } else {
    MATX_ASSERT_STR(
        val.Size(0) == shape[0] * shape[2] * 3, matxInvalidParameter,
        "data arrays should contain all three diagonals (by col index)");
  }
  // Construct the off = { -1, 0, +1 } in values memory space.
  matxMemorySpace_t space = GetPointerKind(val.GetStorage().data());
  Storage<CRD> off = makeZeroStorage<CRD>(3, space);
  setVal(off.data() + 0, static_cast<CRD>(-1), space);
  setVal(off.data() + 2, static_cast<CRD>(+1), space);
  // Note that the DIA API typically does not involve positions.
  // However, under the formal DSL specifications, the top level
  // compression should set up pos[0] = {0, #diags}. This is done
  // here, using the same memory space as the other data.
  Storage<POS> tp = makeZeroStorage<POS>(2, space);
  setVal(tp.data() + 1, static_cast<POS>(3), space);
  // Construct Batched DIA-I/J.
  using DIA = std::conditional_t<std::is_same_v<IDX, DIA_INDEX_I>,
                                 BatchedDIAIUniform, BatchedDIAJUniform>;
  return sparse_tensor_t<VAL, CRD, POS, DIA>(
      shape, val.GetStorage(), {std::move(off), makeEmptyStorage<CRD>()},
      {std::move(tp), makeEmptyStorage<POS>()});
}

} // namespace experimental
} // namespace matx
