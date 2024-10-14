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
#include "matx/transforms/matmul/matmul_cuda.h"
#ifdef MATX_EN_CPU_MATMUL
  #include "matx/transforms/matmul/matmul_cblas.h"
#endif

namespace matx
{
  namespace detail {
    template <typename OpA, typename OpB, typename PermDims>
    class MatMulOp : public BaseOp<MatMulOp<OpA, OpB, PermDims>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        typename detail::base_type_t<OpB> b_;
        float alpha_;
        float beta_;
        PermDims perm_; 
        static constexpr int out_rank = cuda::std::max(OpA::Rank(), OpB::Rank());
        cuda::std::array<index_t, out_rank> out_dims_;
        // This should be tensor_impl_t, but need to work around issues with temp types returned in matmul
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, out_rank> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr; 

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;
        using matx_transform_op = bool;
        using matmul_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
            return "matmul(" + get_type_str(a_) + "," + get_type_str(b_) + ")";
        }

        __MATX_INLINE__ MatMulOp(const OpA &a, const OpB &b, float alpha, float beta, PermDims perm) : 
              a_(a), b_(b), alpha_(alpha), beta_(beta), perm_(perm) {
          if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
            for (int r = 0; r < Rank(); r++) {
              if (r == Rank() - 2) {
                out_dims_[perm_[r]] = a_.Size(r);
              }
              else if (r == Rank() - 1) {
                out_dims_[perm_[r]] = b_.Size(r);
              }
              else {
                out_dims_[perm_[r]] = OpA::Rank() > OpB::Rank() ? a_.Size(r) : b_.Size(r);
              }
            }
          }
          else {
            for (int r = 0; r < Rank() - 2; r++) {
              out_dims_[(size_t)r] = OpA::Rank() > OpB::Rank() ? a_.Size(r) : b_.Size(r);
            }

            out_dims_[Rank() - 2] = a_.Size(OpA::Rank() - 2);
            out_dims_[Rank() - 1] = b_.Size(OpB::Rank() - 1);
          }
        }

        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<InWidth, OutWidth>(indices...);
        }
   

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return out_rank;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
            matmul_impl(permute(cuda::std::get<0>(out), perm_), a_, b_, ex, alpha_, beta_);
          }
          else {
            matmul_impl(cuda::std::get<0>(out), a_, b_, ex, alpha_, beta_);
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }     

          if constexpr (is_matx_op<OpB>()) {
            b_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }           
        }      

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));  

          detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

          Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<OpB>()) {
            b_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          matxFree(ptr);         
        }
    };
  }


  /**
   * Run a GEMM (generic matrix multiply))
   *
   * Creates a new GEMM plan in the cache if none exists, and uses that to execute
   * the GEMM. This function is preferred over creating a plan directly for both
   * efficiency and simpler code. Since it only uses the signature of the GEMM to
   * decide if a plan is cached, it may be able to reused plans for different
   * A/B/C matrices as long as they were configured with the same dimensions.
   *
   * @tparam OpA
   *    Data type of A tensor or operator
   * @tparam OpB
   *    Data type of B tensor or operator
   *
   * @param A
   *   A Tensor or Operator of shape `... x m x k`
   * @param B
   *   B Tensor or Operator of shape `... x k x n`
   * @param alpha
   *   Scalar multiplier to apply to operator A
   * @param beta
   *   Scalar multiplier to apply to operator C on input
   * 
   * @return 
   *   Operator that produces the output tensor C of shape `... x m x n`
   */
  template<typename OpA, typename OpB>
  __MATX_INLINE__ auto matmul(const OpA &A, const OpB &B, float alpha = 1.0, float beta = 0.0) {
    return detail::MatMulOp(A, B, alpha, beta, detail::no_permute_t{});
  }

  /**
   * Run a GEMM (generic matrix multiply))
   *
   * Creates a new GEMM plan in the cache if none exists, and uses that to execute
   * the GEMM. This function is preferred over creating a plan directly for both
   * efficiency and simpler code. Since it only uses the signature of the GEMM to
   * decide if a plan is cached, it may be able to reused plans for different
   * A/B/C matrices as long as they were configured with the same dimensions.
   *
   * @tparam OpA
   *    Data type of A tensor or operator
   * @tparam OpB
   *    Data type of B tensor or operator
   *
   * @param A
   *   A Tensor or Operator of shape `... x m x k`
   * @param B
   *   B Tensor or Operator of shape `... x k x n`
   * @param axis
   *   the axis of the tensor or operator to perform the gemm along
   * @param alpha
   *   Scalar multiplier to apply to operator A
   * @param beta
   *   Scalar multiplier to apply to operator C on input
   * 
   * @return 
   *   Operator that produces the output tensor C of shape `... x m x n`
   */
  template<typename OpA, typename OpB>
  __MATX_INLINE__ auto matmul(const OpA &A, const OpB &B, const int32_t (&axis)[2], float alpha = 1.0, float beta = 0.0) {
    MATX_STATIC_ASSERT(OpA::Rank() == OpB::Rank(), "matmul: inputs must have same rank to use matmul with axis parameter");
    MATX_STATIC_ASSERT(OpA::Rank() == OpB::Rank(), "matmul: inputs and outputs must have same rank to use matmul with axis parameter");

    auto perm = detail::getPermuteDims<OpA::Rank()>(axis);
    auto in1 = permute(A, perm);
    auto in2 = permute(B, perm);

    return detail::MatMulOp(in1, in2, alpha, beta, perm);
  }  
}
