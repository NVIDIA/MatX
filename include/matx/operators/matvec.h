////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// COpBright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above cOpBright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above cOpBright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the cOpBright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COpBRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
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
#include "matx/transforms/matvec.h"

namespace matx
{
  namespace detail {
    template <typename OpA, typename OpB>
    class MatVecOp : public BaseOp<MatVecOp<OpA, OpB>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        typename detail::base_type_t<OpB> b_;
        float alpha_;
        float beta_;
        static constexpr int RANK = remove_cvref_t<OpB>::Rank();
        cuda::std::array<index_t, RANK> out_dims_;
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, RANK> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr; 

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;
        using matx_transform_op = bool;
        using matvec_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          return "matvec(" + get_type_str(a_) + "," + get_type_str(b_)  + ")";
        }

        __MATX_INLINE__ MatVecOp(const OpA &A, const OpB &B, float alpha, float beta) : 
              a_(A), b_(B), alpha_(alpha), beta_(beta) {
          
          for (int r = 0; r < Rank(); r++) {
            out_dims_[r] = a_.Size(r);
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
          return remove_cvref_t<OpB>::Rank();
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex)  const{
          matvec_impl(cuda::std::get<0>(out), a_, b_, ex, alpha_, beta_);
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
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
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
   * Run a GEMV without a plan
   *
   * Performs the GEMV:  C = beta*C + alpha*A*B where A is a matrix and B and C are vectors.
   *
   * Creates a new GEMM plan in the cache if none exists, and uses that to execute
   * the GEMM. This function is preferred over creating a plan directly for both
   * efficiency and simpler code. Since it only uses the signature of the GEMM to
   * decide if a plan is cached, it may be able to reused plans for different
   * A/B/C matrices as long as they were configured with the same dimensions.
   *
   * @tparam OpA
   *    Data type of A operator
   * @tparam OpB
   *    Data type of B operator
   *
   * @param A
   *   A input operator
   * @param B
   *   B input operator
   * @param alpha
   *   Scalar multiplier to apply to operator A
   * @param beta
   *   Scalar multiplier to apply to operator C on input
   *
   */
  template <typename OpA, typename OpB>
  __MATX_INLINE__ auto matvec(const OpA &A, const OpB &B,
              float alpha = 1.0, float beta = 0.0)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
    return detail::MatVecOp(A, B, alpha, beta);
  }

}
