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
#ifndef __CUDACC_RTC__
#pragma once

#include <string>

#include "matx/core/sparse_tensor_format.h"
#include "matx/core/tensor_impl.h"
#include "matx/operators/base_operator.h"

namespace matx {

namespace detail {

//
// A sparse_set operation. Assigning to a sparse tensor is very different
// from all other MatX assignments, because the underlying storage and
// buffers may have to be resized to accomodate the output. Therefore,
// for now, we provide a customized set operation that passes a direct
// reference to the executor.
//
template <typename T, typename Op>
class sparse_set : public BaseOp<sparse_set<T, Op>> {
private:
  T &out_;
  mutable typename detail::base_type_t<Op> op_;

public:
  inline sparse_set(T &out, const Op &op) : out_(out), op_(op) {}
  template <typename Ex> __MATX_INLINE__ void run(Ex &&ex) {
    op_.Exec(out_, std::forward<Ex>(ex));
  }
};

} // end namespace detail

namespace experimental {

//
// Sparse tensors
//   VAL : data type of stored elements
//   CRD : data type of coordinates
//   POS : data type of positions
//   TF  : tensor format
//
template <typename VAL, typename CRD, typename POS, typename TF,
          typename StorageV = DefaultStorage<VAL>,
          typename StorageC = DefaultStorage<CRD>,
          typename StorageP = DefaultStorage<POS>,
          typename DimDesc = DefaultDescriptor<TF::DIM>>
class sparse_tensor_t
    : public detail::tensor_impl_t<
          VAL, TF::DIM, DimDesc, detail::SparseTensorData<VAL, CRD, POS, TF>> {
public:
  using sparse_tensor = bool;
  using val_type = VAL;
  using crd_type = CRD;
  using pos_type = POS;
  using Format = TF;

  static constexpr int DIM = TF::DIM;
  static constexpr int LVL = TF::LVL;

  //
  // Constructs a sparse tensor with given shape and contents.
  //
  // The storage format is defined through the template. The contents
  // consist of a buffer for the values (primary storage), and LVL times
  // a buffer with the coordinates and LVL times a buffer with the positions
  // (secondary storage). The semantics of these contents depend on the
  // used storage format.
  //
  // Most users should *not* use this constructor directly, since it depends
  // on intricate knowledge of the storage formats. Instead, users should
  // use the "make_sparse_tensor" methods that provide factory methods in
  // terms of storage formats that are more familiar (COO, CSR, etc).
  //
  __MATX_INLINE__
  sparse_tensor_t(const typename DimDesc::shape_type (&shape)[DIM],
                  StorageV &&vals, StorageC (&&crd)[LVL], StorageP (&&pos)[LVL])
      : detail::tensor_impl_t<VAL, DIM, DimDesc,
                              detail::SparseTensorData<VAL, CRD, POS, TF>>(
            shape) {
    values_ = std::move(vals);
    for (int l = 0; l < LVL; l++) {
      coordinates_[l] = std::move(crd[l]);
      positions_[l] = std::move(pos[l]);
    }
    SetSparseDataImpl();
  }

  // Default destructor.
  __MATX_INLINE__ ~sparse_tensor_t() = default;

  // Sets value storage.
  __MATX_INLINE__ void SetVal(StorageV &&val) { values_ = std::move(val); }

  // Sets coordinates storage.
  __MATX_INLINE__ void SetCrd(int l, StorageC &&crd) {
    coordinates_[l] = std::move(crd);
  }

  // Sets positions storage.
  __MATX_INLINE__ void SetPos(int l, StorageP &&pos) {
    positions_[l] = std::move(pos);
  }

  // Sets sparse data in tensor_impl_t. This method must be called
  // every time changes are made to the underlying storage objects.
  void SetSparseDataImpl() {
    VAL *v = values_.data();
    CRD *c[LVL];
    POS *p[LVL];
    for (int l = 0; l < LVL; l++) {
      c[l] = coordinates_[l].data();
      p[l] = positions_[l].data();
    }
    this->SetSparseData(v, c, p);
  }

  // A direct sparse tensor assignment (viz. (Acoo = ...).exec();).
  template <typename T>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(const T &op) {
    [[maybe_unused]] typename T::tosparse_xform_op valid = true;
    return detail::sparse_set(*this, op);
  }

  // Size getters.
  index_t Nse() const {
    return static_cast<index_t>(values_.size() / sizeof(VAL));
  }
  index_t crdSize(int l) const {
    return static_cast<index_t>(coordinates_[l].size() / sizeof(CRD));
  }
  index_t posSize(int l) const {
    return static_cast<index_t>(positions_[l].size() / sizeof(POS));
  }

private:
  // Primary storage of sparse tensor (explicitly stored element values).
  StorageV values_;

  // Secondary storage of sparse tensor (coordinates and positions).
  // There is potentially one for each level, although some of these
  // may remain empty. The secondary storage is essential to determine
  // where in the original tensor the explicitly stored elements reside.
  StorageC coordinates_[LVL];
  StorageP positions_[LVL];
};

} // end namespace experimental
} // end namespace matx
#endif
