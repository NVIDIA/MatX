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

#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"
#include "matx/transforms/convert/sparse2sparse_cusparse.h"

namespace matx {
namespace detail {

template <typename OpA>
class Sparse2SparseOp : public BaseOp<Sparse2SparseOp<OpA>> {
private:
  typename detail::base_type_t<OpA> a_;

public:
  using matxop = bool;
  using matx_transform_op = bool;
  using tosparse_xform_op = bool;
  using value_type = typename OpA::value_type;

  __MATX_INLINE__ Sparse2SparseOp(const OpA &a) : a_(a) {}

  __MATX_INLINE__ std::string str() const {
    return "sparse2sparse(" + get_type_str(a_) + ")";
  }

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t
  Rank() {
    return remove_cvref_t<OpA>::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t
  Size(int dim) const {
    return a_.Size(dim);
  }

  template <typename Out, typename Executor>
  void Exec([[maybe_unused]] Out &&out, [[maybe_unused]] Executor &&ex) const {
    if constexpr (is_sparse_tensor_v<OpA> && is_sparse_tensor_v<Out>) {
      // NOTE: sparse assignment O = sparse2sparse(A) takes direct reference!
      sparse2sparse_impl(out, a_, ex);
    } else {
      MATX_THROW(matxNotSupported,
                 "Cannot use sparse2sparse on dense operands");
    }
  }
};

} // end namespace detail

/**
 * Convert a sparse tensor into a sparse tensor. Typically
 * used to convert storage format (e.g. COO to CSR). Note that
 * for speed-of-operation, after this operation, the input and
 * output tensor may share some of the underlying allocated
 * memory (e.g. the values array).
 *
 * Currently only COO to CSR is supported (with CSR "stealing"
 * the j-index and values array from COO, while recomputing its
 * own positions array).
 *
 * @tparam OpA
 *    Data type of A tensor
 *
 * @param A
 *   Sparse input tensor
 *
 * @return
 *   Sparse output tensor
 */
template <typename OpA> __MATX_INLINE__ auto sparse2sparse(const OpA &A) {
  return detail::Sparse2SparseOp(A);
}

} // end namespace matx
