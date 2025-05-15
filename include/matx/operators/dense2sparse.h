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
#ifndef __CUDACC_RTC__
#include "matx/transforms/convert/dense2sparse_cusparse.h"
#endif

namespace matx {
namespace detail {

template <typename OpA>
class Dense2SparseOp : public BaseOp<Dense2SparseOp<OpA>> {
private:
  typename detail::base_type_t<OpA> a_;

public:
  using matxop = bool;
  using matx_transform_op = bool;
  using tosparse_xform_op = bool;
  using value_type = typename OpA::value_type;

  __MATX_INLINE__ Dense2SparseOp(const OpA &a) : a_(a) {}

  __MATX_INLINE__ std::string str() const {
    return "dense2sparse(" + get_type_str(a_) + ")";
  }

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t
  Rank() {
    return remove_cvref_t<OpA>::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t
  Size(int dim) const {
    return a_.Size(dim);
  }

  template <OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
    auto self_has_cap = capability_attributes<Cap>::default_value;
    return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
  }

#ifndef __CUDACC_RTC__
  template <typename Out, typename Executor>
  void Exec([[maybe_unused]] Out &&out, [[maybe_unused]] Executor &&ex) const {
    if constexpr (is_sparse_tensor_v<OpA>) {
      MATX_THROW(matxNotSupported, "Cannot use dense2sparse on sparse input");
    } else {
      // NOTE: sparse assignment O = dense2sparse(A) takes direct reference!
      if constexpr (is_sparse_tensor_v<Out>) {
        dense2sparse_impl(out, a_, ex);
      } else {
        MATX_THROW(matxNotSupported,
                   "Cannot use dense2sparse for dense output");
      }
    }
  }
#endif
};

} // end namespace detail

/**
 * Convert a dense tensor into a sparse tensor.
 *
 * @tparam OpA
 *    Data type of A tensor
 *
 * @param A
 *   Dense input tensor
 *
 * @return
 *   Sparse output tensor
 */
template <typename OpA> __MATX_INLINE__ auto dense2sparse(const OpA &A) {
  return detail::Dense2SparseOp(A);
}

} // end namespace matx
