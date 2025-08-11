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
#include "matx/core/capabilities.h"
#include "matx/core/operator_utils.h"

namespace matx {
template <typename T, int RANK, typename Storage, typename Desc> class tensor_t; ///< Tensor detail type
template <typename T> class BaseOp; ///< Base operator type

namespace detail {

template <typename T, int RANK, typename Desc, typename Data> class tensor_impl_t; ///< Tensor implementation type


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


#ifndef __CUDACC_RTC__
  __MATX_INLINE__ const std::string str() const {
    return get_type_str(out_) + "=" + get_type_str(op_);
  }
#endif


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
      MATX_ASSERT_COMPATIBLE_OP_SIZES(op);
    }
  }


  // Workaround for nvcc bug. It won't allow the dual if constexpr branch workaround inside of lambda
  // functions, so we have to make a separate one.
  template <typename CapType, typename... Ts>
  __MATX_DEVICE__ __MATX_HOST__ inline auto _internal_mapply(Ts&&... args) const noexcept {
    auto r = detail::get_value<CapType>(op_, args...);
    out_(args...) = r;
    return r;
  }

  template <typename CapType, typename ShapeType>
  __MATX_DEVICE__ __MATX_HOST__ inline decltype(auto) operator()(cuda::std::array<ShapeType, T::Rank()> idx) const noexcept
  {
    auto res = cuda::std::apply([&](auto &&...args)  {
        return _internal_mapply<CapType>(args...);
      }, idx
    );

    return res;
  }

  template <typename CapType, typename... Is>
  __MATX_DEVICE__ __MATX_HOST__ inline decltype(auto) operator()(Is... indices) const noexcept
  {
#ifdef __CUDA_ARCH__
        if constexpr (CapType::jit) {
          if ((threadIdx.x * CapType::ept) >= Size(Rank() - 1)) {
            return detail::GetJitSentinelValue<CapType, value_type>();
          }
        }
#endif
    const auto in_val = detail::get_value<CapType>(op_, indices...);
    using out_type = decltype(out_.template operator()<CapType>(indices...));

#ifdef __CUDA_ARCH__    
    // If we get a scalar on the input and a vector output, construct a vector of these scalars to write out
    if constexpr (CapType::jit) {
      if (out_.Rank() == 0 || threadIdx.x < out_.Size(out_.Rank() - 1)) {
        if constexpr (!is_vector_v<decltype(in_val)> && is_vector_v<out_type>) {
          Vector<remove_cvref_t<decltype(in_val)>, static_cast<size_t>(CapType::ept)> vec{in_val};
          out_.template operator()<CapType>(indices...) = vec;
        }
        else {
          out_.template operator()<CapType>(indices...) = in_val;
        }
      }
      return in_val;      
    }
    else 
#endif    
    {
      if constexpr (!is_vector_v<decltype(in_val)> && is_vector_v<out_type>) {
        Vector<remove_cvref_t<decltype(in_val)>, static_cast<size_t>(CapType::ept)> vec{in_val};
        out_.template operator()<CapType>(indices...) = vec;
      }
      else {
        out_.template operator()<CapType>(indices...) = in_val;
      }
      return in_val;
    }
  }  


  template <typename... Is>
  __MATX_DEVICE__ __MATX_HOST__ inline decltype(auto) operator()(Is... indices) const noexcept  
  {
    return (*this).template operator()<DefaultCapabilities>(indices...);
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

  template <detail::OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] const InType& in) const {
    auto self_has_cap = capability_attributes<Cap>::default_value;
    if constexpr (Cap == detail::OperatorCapability::SUPPORTS_JIT) {
      printf("set get_capability %d %d %d %d\n", (int)Cap, detail::get_operator_capability<Cap>(out_, in), detail::get_operator_capability<Cap>(op_, in),  combine_capabilities<Cap>(self_has_cap, 
        detail::get_operator_capability<Cap>(out_, in),
        detail::get_operator_capability<Cap>(op_, in)));
    }
    return combine_capabilities<Cap>(self_has_cap, 
                                      detail::get_operator_capability<Cap>(out_, in),
                                      detail::get_operator_capability<Cap>(op_, in));
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
    if constexpr (T::Rank() == 0) {
      return 1;
    }
    else {
      return out_.Size(dim);
    }
  }
};


}
}
