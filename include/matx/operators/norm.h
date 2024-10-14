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


#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"
#include "matx/transforms/norm.h"

namespace matx
{
  namespace detail {
    template <typename OpA, typename NormType>
    class NormOp : public BaseOp<NormOp<OpA, NormType>>
    {
      private:
        using out_type = typename inner_op_type_t<typename remove_cvref_t<OpA>::value_type>::type;
        typename detail::base_type_t<OpA> a_;
        NormOrder order_;
        static constexpr int ORank = std::is_same_v<NormType, detail::NormTypeVector> ? OpA::Rank() - 1 : OpA::Rank() - 2;
        cuda::std::array<index_t, ORank> out_dims_;
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, ORank> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr; 

      public:
        using matxop = bool;
        using value_type = out_type;
        using matx_transform_op = bool;
        using norm_xform_op = bool;
        using matx_inner_op_impl = bool; // Indicates this operator uses matx operators for its implementation

      __MATX_INLINE__ std::string str() const { 
        if constexpr (std::is_same_v<NormType, detail::NormTypeVector>) {
          return "vector_norm()"; 
        }
        else {
          return "matrix_norm";
        }
      }

      __MATX_INLINE__ NormOp(const OpA &op, NormOrder order) : a_(op), order_(order) {
        if constexpr (std::is_same_v<NormType, detail::NormTypeVector>) {
          MATX_ASSERT_STR(order == NormOrder::NONE || order == NormOrder::L1 || order == NormOrder::L2, matxInvalidParameter,
            "Invalid norm order used for vector mode");
        }

        for (int r = 0; r < ORank; r++) {
          out_dims_[r] = a_.Size(r);
        }
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        return tmp_out_.template operator()<InWidth, OutWidth>(indices...);
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        norm_impl<NormType>(cuda::std::get<0>(out), a_, order_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return ORank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      } 

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));

        detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

        Exec(std::make_tuple(tmp_out_), std::forward<Executor>(ex));
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        matxFree(ptr);
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return out_dims_[dim];
      }
    };
  }


  /**
   * @brief Compute a vector norm
   *
   * Computes various types of matrix and vector norms based on the order
   *
   * @tparam Op Type of input values to evaluate
   * @param op Input values to evaluate
   * @param order Order of norm
   * @return norm operator
   */
  template <typename Op>
  __MATX_INLINE__ auto vector_norm(const Op &op,
                            NormOrder order = NormOrder::NONE) {
    return detail::NormOp<Op, detail::NormTypeVector>(op, order);
  }


  /**
   * @brief Compute a vector norm
   *
   * Computes various types of vector norms based on the order
   *
   * @tparam Op Type of input values to evaluate
   * @param op Input values to evaluate
   * @param dims Dimensions to perform norm over   
   * @param order Order of norm
   * @return norm operator
   */
  template <typename Op, int D>
  __MATX_INLINE__ auto vector_norm(const Op &op, const int (&dims)[D],
                            NormOrder order = NormOrder::NONE) {
    auto perm = detail::getPermuteDims<Op::Rank()>(dims);
    auto permop = permute(op, perm);
    return detail::NormOp<decltype(permop), detail::NormTypeVector>(permop, order);
  }

  /**
   * @brief Compute a matrix norm
   *
   * Computes various types of matrix and matrix norms based on the order
   *
   * @tparam Op Type of input values to evaluate
   * @param op Input values to evaluate
   * @param order Order of norm
   * @return norm operator
   */
  template <typename Op>
  __MATX_INLINE__ auto matrix_norm(const Op &op,
                            NormOrder order = NormOrder::NONE) {
    return detail::NormOp<Op, detail::NormTypeMatrix>(op, order);
  }


  /**
   * @brief Compute a matrix norm
   *
   * Computes various types of matrix norms based on the order
   *
   * @tparam Op Type of input values to evaluate
   * @param op Input values to evaluate
   * @param dims Dimensions to perform norm over   
   * @param order Order of norm
   * @return norm operator
   */
  template <typename Op, int D>
  __MATX_INLINE__ auto matrix_norm(const Op &op, const int (&dims)[D],
                            NormOrder order = NormOrder::NONE) {
    auto perm = detail::getPermuteDims<Op::Rank()>(dims);
    auto permop = permute(op, perm);
    return detail::NormOp<Op, detail::NormTypeMatrix>(permop, order);
  }  
} // end namespace matx
