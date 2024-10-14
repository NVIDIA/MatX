////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
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

#include <type_traits>

#include "matx/core/error.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"

namespace matx {
template <typename T, int RANK, typename Storage, typename Desc> class tensor_t; ///< Tensor detail type
template <typename T> class BaseOp; ///< Base operator type

namespace detail {

template <typename T, int RANK, typename Desc> class tensor_impl_t; ///< Tensor implementation type


/**
 * Assignment from one operator/View into a View
 *
 * The set function is used in lieu of the assignment operator to avoid
 * ambiguity in certain scenarios. It can be used in the same scenarios
 * as the assignment operator.
 *
 * @tparam T
 *   Operator to use as output
 * @tparam Op
 *   Operator to use as input
 **/
template <typename T, typename Op>
class set : public BaseOp<set<T, Op>> {
private:
  mutable typename detail::base_type_t<T> out_;
  mutable typename detail::base_type_t<Op> op_;

public:
  // Type specifier for reflection on class
  using value_type = typename T::value_type;
  using tensor_type = T;
  using op_type = Op;
  using matx_setop = bool;
  using matx_width = bool; ///< Signal we can do vector types from this operator

  __MATX_INLINE__ const std::string str() const {
    return get_type_str(out_) + "=" + get_type_str(op_);
  }

  auto &get_lhs() {
    return out_;
  }

  auto &get_rhs() {
    return op_;
  }

  /**
   * Constructor to assign an operator to a view
   *
   * @param out
   *   Output destination view
   *
   * @param op
   *   Input operator
   */
  inline set(T &out, const Op &op) : out_(out), op_(op)
  {
    static_assert(is_matx_op_lvalue<T>() == true, "Invalid operator on LHS of set/operator=");
    static_assert(!is_matx_transform_op<T>(), "Cannot use transform operator on LHS of assignment");

    // set() is a placeholder when using mtie() for multiple return types, so we don't need to check compatible
    // sizes
    if constexpr (!is_mtie<T>()) {
      ASSERT_COMPATIBLE_OP_SIZES(op);
    }
  }

  template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
  __MATX_DEVICE__ __MATX_HOST__ inline decltype(auto) operator()(Is... indices) const noexcept
  {
    if constexpr (is_matx_half_v<T> &&
                  std::is_integral_v<decltype(detail::get_value<InWidth, OutWidth>(op_, indices...))>) {
      out_.template operator()<InWidth, OutWidth>(indices...) = static_cast<float>(detail::get_value<InWidth, OutWidth>(op_, indices...));
    }
    else {
      out_.template operator()<InWidth, OutWidth>(indices...) = detail::get_value<InWidth, OutWidth>(op_, indices...);
    }

    return out_.template operator()<InWidth, OutWidth>(indices...);
  }

  // Workaround for nvcc bug. It won't allow the dual if constexpr branch workaround inside of lambda
  // functions, so we have to make a separate one.
  template <VecWidth InWidth, VecWidth OutWidth, typename... Ts>
  __MATX_DEVICE__ __MATX_HOST__ inline auto _internal_mapply(Ts&&... args) const noexcept {
    if constexpr (is_matx_half_v<T> &&
                  std::is_integral_v<decltype(detail::get_value<InWidth, OutWidth>(op_, args...))>) {
      auto r = static_cast<float>(detail::get_value<InWidth, OutWidth>(op_, args...));
      out_.template operator()<InWidth, OutWidth>(args...) = r;
      return r;
    }
    else {
      auto r = detail::get_value<InWidth, OutWidth>(op_, args...);
      out_.template operator()<InWidth, OutWidth>(args...) = r;
      return r;
    }
  }

  template <VecWidth InWidth, VecWidth OutWidth>
  __MATX_DEVICE__ __MATX_HOST__ inline decltype(auto) operator()(cuda::std::array<index_t, T::Rank()> idx) const noexcept
  {
    auto res = cuda::std::apply([&](auto &&...args)  {
        return _internal_mapply<InWidth, OutWidth>(args...);
      }, idx
    );
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
  {
    if constexpr (is_matx_op<T>()) {
      out_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }    
    if constexpr (is_matx_op<Op>()) {
      op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept  
  {
    if constexpr (is_matx_op<T>()) {
      out_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }     
    if constexpr (is_matx_op<Op>()) {
      op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
  }

  // Used as a shortcut where the RHS is an executor and LHS is a tensor. In this case we
  // want to avoid the RHS from allocating any temporary output memory, so we call
  // InnerPreRun on it to call any nested PreRun calls, then output directly into the LHS
  // tensor.
  template <typename ShapeType, typename Executor>
  void TransformExec(ShapeType &&shape, Executor &&ex) const noexcept {
    op_.InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    op_.Exec(cuda::std::make_tuple(out_), std::forward<Executor>(ex));
    op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
  }

  static constexpr bool IsTransformSet() {
    if constexpr (is_matx_transform_op<Op>() && is_tensor_view_v<T>) {
      return true;
    }
    else {
      return false;
    }
  }


  VecWidth GetMaxWidth() const {
    return MinCompatibleWidth(out_, op_);
  }  

  /**
   * Get the rank of the operator
   *
   * @return
   *   Rank of the operator
   */
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return T::Rank(); }

  /**
   * Get the rank of the operator along a single dimension
   *
   * @param dim
   *   Dimension to retrieve size
   * @return
   *   Size of dimension
   */
  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
  {
    static_assert(T::Rank() >= 1, "Size function only works on tensors of rank 1 and higher");
    return out_.Size(dim);
  }
};


}
}
