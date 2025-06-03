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
#include "matx/transforms/solve/solve_cusparse.h"
#ifdef MATX_EN_CUDSS
#include "matx/transforms/solve/solve_cudss.h"
#endif

namespace matx {
namespace detail {

template <typename OpA, typename OpB>
class SolveOp : public BaseOp<SolveOp<OpA, OpB>> {
private:
  typename detail::base_type_t<OpA> a_;
  typename detail::base_type_t<OpB> b_;

  static constexpr int out_rank = OpB::Rank();
  cuda::std::array<index_t, out_rank> out_dims_;
  mutable detail::tensor_impl_t<typename OpA::value_type, out_rank> tmp_out_;
  mutable typename OpA::value_type *ptr = nullptr;

public:
  using matxop = bool;
  using matx_transform_op = bool;
  using solve_xform_op = bool;
  using value_type = typename OpA::value_type;

  __MATX_INLINE__ SolveOp(const OpA &a, const OpB &b) : a_(a), b_(b) {
    for (int r = 0, rank = Rank(); r < rank; r++) {
      out_dims_[r] = b_.Size(r);
    }
  }

  __MATX_INLINE__ std::string str() const {
    return "solve(" + get_type_str(a_) + "," + get_type_str(b_) + ")";
  }

  __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto)
  operator()(Is... indices) const {
    return tmp_out_(indices...);
  }

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t
  Rank() {
    return remove_cvref_t<OpB>::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t
  Size(int dim) const {
    return out_dims_[dim];
  }

  template <typename Out, typename Executor>
  void Exec([[maybe_unused]] Out &&out, [[maybe_unused]] Executor &&ex) const {
    static_assert(!is_sparse_tensor_v<OpB>, "sparse rhs not implemented");
    if constexpr (is_sparse_tensor_v<OpA>) {
      if constexpr (OpA::Format::isDIAI() || OpA::Format::isDIAJ()) {
        sparse_dia_solve_impl(cuda::std::get<0>(out), a_, b_, ex);
      } else {
#ifdef MATX_EN_CUDSS
        sparse_solve_impl(cuda::std::get<0>(out), a_, b_, ex);
#else
        MATX_THROW(matxNotSupported, "Sparse direct solver requires cuDSS");
#endif
      }
    } else {
      MATX_THROW(matxNotSupported,
                 "Direct solver currently only supports sparse system");
    }
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void
  InnerPreRun([[maybe_unused]] ShapeType &&shape,
              [[maybe_unused]] Executor &&ex) const noexcept {
    static_assert(is_sparse_tensor_v<OpA>,
                  "Direct solver currently only supports sparse system");
    if constexpr (is_matx_op<OpB>()) {
      b_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape,
                              [[maybe_unused]] Executor &&ex) const noexcept {
    InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_,
                               &ptr);
    Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape,
                               [[maybe_unused]] Executor &&ex) const noexcept {
    static_assert(is_sparse_tensor_v<OpA>,
                  "Direct solver currently only supports sparse system");
    if constexpr (is_matx_op<OpB>()) {
      b_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
    matxFree(ptr);
  }
};

} // end namespace detail

/**
 * Running X = solve(A, B) solves the system A X^T = B^T for
 * an unknown X given the rhs B. The transposition is used so
 * that the different rhs vectors and solutions are stacked by
 * row (another way to think about this is that X and B are
 * presented using column-major storage). Currently, this
 * operation is only implemented for solving a linear system
 * with a very **sparse** matrix A in CSR or DIA format.
 *
 * @tparam OpA
 *    Data type of A tensor (sparse)
 * @tparam OpB
 *    Data type of B tensor
 *
 * @param A
 *   A Sparse tensor with system coefficients
 * @param B
 *   B Dense tensor of known values
 *
 * @return
 *   Operator that produces the output tensor X with the solution
 */
template <typename OpA, typename OpB>
__MATX_INLINE__ auto solve(const OpA &A, const OpB &B) {
  return detail::SolveOp(A, B);
}

} // end namespace matx
