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
#include "matx/transforms/convert/sparse2dense_cusparse.h"

namespace matx {
namespace detail {

template <typename OpA>
class Sparse2DenseOp : public BaseOp<Sparse2DenseOp<OpA>> {
private:
  typename detail::base_type_t<OpA> a_;

  static constexpr int out_rank = OpA::Rank();
  cuda::std::array<index_t, out_rank> out_dims_;
  mutable detail::tensor_impl_t<typename OpA::value_type, out_rank> tmp_out_;
  mutable typename OpA::value_type *ptr = nullptr;
  mutable bool prerun_done_ = false;

public:
  using matxop = bool;
  using matx_transform_op = bool;
  using sparse2dense_xform_op = bool;
  using value_type = typename OpA::value_type;

  __MATX_INLINE__ Sparse2DenseOp(const OpA &a) : a_(a) {
    for (int r = 0; r < Rank(); r++) {
      out_dims_[r] = a_.Size(r);
    }
  }

  __MATX_INLINE__ std::string str() const {
    return "sparse2dense(" + get_type_str(a_) + ")";
  }

  __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

  template <ElementsPerThread EPT, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto)
  operator()(Is... indices) const {
    return tmp_out_.template operator()<EPT>(indices...);
  }

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto)
  operator()(Is... indices) const {
    return this->operator()<detail::ElementsPerThread::ONE>(indices...);
  }

  template <OperatorCapability Cap>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
    auto self_has_cap = capability_attributes<Cap>::default_value;
    return combine_capabilities<Cap>(self_has_cap,
                                     detail::get_operator_capability<Cap>(a_));
  }

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t
  Rank() {
    return remove_cvref_t<OpA>::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t
  Size(int dim) const {
    return out_dims_[dim];
  }

  template <typename Out, typename Executor>
  void Exec([[maybe_unused]] Out &&out, [[maybe_unused]] Executor &&ex) const {
    if constexpr (is_sparse_tensor_v<OpA>) {
      using Rtype = decltype(cuda::std::get<0>(out));
      if constexpr (is_sparse_tensor_v<Rtype>) {
        MATX_THROW(matxNotSupported,
                   "Cannot use sparse2dense for sparse output");
      } else {
        sparse2dense_impl(cuda::std::get<0>(out), a_, ex);
      }
    } else {
      MATX_THROW(matxNotSupported, "Cannot use sparse2dense on dense input");
    }
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void
  InnerPreRun([[maybe_unused]] ShapeType &&shape,
              [[maybe_unused]] Executor &&ex) const noexcept {
    static_assert(is_sparse_tensor_v<OpA>,
                  "Cannot use sparse2dense on dense input");
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape,
                              [[maybe_unused]] Executor &&ex) const noexcept {
    if (prerun_done_) {
      return;
    }

    InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_,
                               &ptr);
    prerun_done_ = true;
    Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape,
                               [[maybe_unused]] Executor &&ex) const noexcept {
    static_assert(is_sparse_tensor_v<OpA>,
                  "Cannot use sparse2dense on dense input");
    matxFree(ptr);
  }
};

} // end namespace detail

/**
 * Convert a sparse tensor into a dense tensor.
 *
 * @tparam OpA
 *    Data type of A tensor
 *
 * @param A
 *   Sparse input tensor
 *
 * @return
 *   Dense output tensor
 */
template <typename OpA> __MATX_INLINE__ auto sparse2dense(const OpA &A) {
  return detail::Sparse2DenseOp(A);
}

} // end namespace matx
