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

#include <string>

#include "matx/core/sparse_tensor_format.h"
#include "matx/core/tensor_impl.h"

namespace matx {
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
    // Initialize primary and secondary storage.
    values_ = std::move(vals);
    for (int l = 0; l < LVL; l++) {
      coordinates_[l] = std::move(crd[l]);
      positions_[l] = std::move(pos[l]);
    }
    // Set the sparse data in tensor_impl.
    VAL *v = values_.data();
    CRD *c[LVL];
    POS *p[LVL];
    for (int l = 0; l < LVL; l++) {
      c[l] = coordinates_[l].data();
      p[l] = positions_[l].data();
      // All non-null data resides in same space.
      assert(!c[l] || GetPointerKind(c[l]) == GetPointerKind(v));
      assert(!p[l] || GetPointerKind(p[l]) == GetPointerKind(v));
    }
    this->SetSparseData(v, c, p);
  }

  // Default destructor.
  __MATX_INLINE__ ~sparse_tensor_t() = default;

  // Size getters.
  index_t Nse() const { return values_.size() / sizeof(VAL); }
  index_t crdSize(int l) const { return coordinates_[l].size() / sizeof(CRD); }
  index_t posSize(int l) const { return positions_[l].size() / sizeof(POS); }

  // Locates position of an element at given indices, or returns -1 when not
  // found.
  template <int L = 0>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t
  GetPos(index_t *lvlsz, index_t *lvl, index_t pos) const {
    if constexpr (L < LVL) {
      using ftype = std::tuple_element_t<L, typename TF::LVLSPECS>;
      if constexpr (ftype::lvltype == LvlType::Dense) {
        // Dense level: pos * size + i.
        // TODO: see below, use a constexpr GetLvlSize(L) instead?
        const index_t dpos = pos * lvlsz[L] + lvl[L];
        if constexpr (L + 1 < LVL) {
          return GetPos<L + 1>(lvlsz, lvl, dpos);
        } else {
          return dpos;
        }
      } else if constexpr (ftype::lvltype == LvlType::Singleton) {
        // Singleton level: pos if crd[pos] == i and next levels match.
        if (this->CRDData(L)[pos] == lvl[L]) {
          if constexpr (L + 1 < LVL) {
            return GetPos<L + 1>(lvlsz, lvl, pos);
          } else {
            return pos;
          }
        }
      } else if constexpr (ftype::lvltype == LvlType::Compressed ||
                           ftype::lvltype == LvlType::CompressedNonUnique) {
        // Compressed level: scan for match on i and test next levels.
        const CRD *c = this->CRDData(L);
        const POS *p = this->POSData(L);
        for (index_t pp = p[pos], hi = p[pos + 1]; pp < hi; pp++) {
          if (c[pp] == lvl[L]) {
            if constexpr (L + 1 < LVL) {
              const index_t cpos = GetPos<L + 1>(lvlsz, lvl, pp);
              if constexpr (ftype::lvltype == LvlType::Compressed) {
                return cpos; // always end scan (unique)
              } else if (cpos != -1) {
                return cpos; // only end scan on success (non-unique)
              }
            } else {
              return pp;
            }
          }
        }
      }
    }
    return -1; // not found
  }

  // Element getter (viz. "lhs = Acoo(0,0);"). Note that due to the compact
  // nature of sparse data structures, these storage formats do not provide
  // cheap random access to their elements. Instead, the implementation will
  // search for a stored element at the given position (which involves a scan
  // at each compressed level). The implicit value zero is returned when the
  // element cannot be found. So, although functional for testing, clients
  // should avoid using getters inside performance critial regions, since
  // the implementation is far worse than O(1).
  template <typename... Is>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ VAL
  operator()(Is... indices) const noexcept {
    static_assert(
        sizeof...(Is) == DIM,
        "Number of indices of operator() must match rank of sparse tensor");
    cuda::std::array<index_t, DIM> dim{indices...};
    cuda::std::array<index_t, LVL> lvl;
    cuda::std::array<index_t, LVL> lvlsz;
    TF::dim2lvl(dim.data(), lvl.data(), /*asSize=*/false);
    // TODO: only compute once and provide a constexpr LvlSize(l) instead?
    TF::dim2lvl(this->Shape().data(), lvlsz.data(), /*asSize=*/true);
    const index_t pos = GetPos(lvlsz.data(), lvl.data(), 0);
    if (pos != -1) {
      return this->Data()[pos];
    }
    return static_cast<VAL>(0); // implicit zero
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
