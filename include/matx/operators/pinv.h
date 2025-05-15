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
#ifndef __CUDACC_RTC__
#include "matx/transforms/pinv.h"
#endif

namespace matx {
  
namespace detail {
  template<typename OpA>
  class PinvOp : public BaseOp<PinvOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      float rcond_;
      cuda::std::array<index_t, OpA::Rank()> out_dims_;
      mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, OpA::Rank()> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;
      mutable bool prerun_done_ = false; 

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using pinv_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "pinv()"; }
      __MATX_INLINE__ PinvOp(const OpA &a, float rcond) : a_(a), rcond_(rcond) {
        for (int r = 0; r < Rank(); r++) {
          if (r >= Rank() - 2) {
            out_dims_[r] = (r == Rank() - 1) ? a_.Size(Rank() - 2) : a_.Size(Rank() - 1);
          }
          else {
            out_dims_[r] = a_.Size(r);
          }
        } 
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
#ifdef __CUDA_ARCH__
        if constexpr (CapType::jit) {
          if ((threadIdx.x * CapType::ept) >= Size(Rank() - 1)) {
            return detail::GetJitSentinelValue<CapType, value_type>();
          }
        }
#endif
        return tmp_out_.template operator()<CapType>(indices...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        return this->operator()<DefaultCapabilities>(indices...);
      }

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return out_dims_[dim];
      }

#ifndef __CUDACC_RTC__
      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        printf("pinv exec\n");
        pinv_impl(cuda::std::get<0>(out), a_, ex, rcond_);
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }       
      }      

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if (prerun_done_) {
          return;
        }

        InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));  

        detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

        prerun_done_ = true;
        Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        matxFree(ptr);
      }
#endif
  };
}

/**
 * Returns an appropriate rcond based on the inner type. This is slightly
 * higher than the machine epsilon, as these work better to mask small/zero singular
 * values in singular or ill-conditioned matrices.
 */
template <typename T>
__MATX_INLINE__ constexpr float get_default_rcond() {
  if constexpr (is_fp32_inner_type_v<T>) {
    return 1e-6f;
  } else {
    return 1e-15f;
  }
}

/**
 * Perfom a generalized inverse of a matrix using its singular-value decomposition (SVD).
 * It automatically removes small singular values for stability.
 * 
 * If rank > 2, operations are batched.
 * 
 * @tparam OpA
 *   Tensor or operator type of input A
 * 
 * @param a
 *   Input tensor or operator of shape `... x m x n`
 * @param rcond
 *   Cutoff for small singular values. For stability, singular values
 *   smaller than `rcond * largest_singular_value` are set to 0 for each matrix
 *   in the batch. By default, `rcond` is approximately the machine epsilon of the tensor dtype
 *   (`1e-6 `for float types and `1e-15` for double types).
 * 
 * @return
 *   Operator that produces a tensor of size `... x n x m` representing the pseudo-inverse of the input
 */
template<typename OpA>
__MATX_INLINE__ auto pinv(const OpA &a, float rcond = get_default_rcond<typename OpA::value_type>()) {
  return detail::PinvOp(a, rcond);
}

}