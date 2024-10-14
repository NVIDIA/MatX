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
#include "matx/transforms/resample_poly.h"

namespace matx {



namespace detail {
  template<typename OpA, typename FilterType>
  class ResamplePolyOp : public BaseOp<ResamplePolyOp<OpA, FilterType>>
  {
    private:
      using out_t = std::conditional_t<is_complex_v<typename OpA::value_type>, 
            typename FilterType::value_type, typename FilterType::value_type>;          
      typename detail::base_type_t<OpA> a_;
      typename detail::base_type_t<FilterType> f_;
      index_t up_;
      index_t down_;
      cuda::std::array<index_t, OpA::Rank()> out_dims_;
      mutable detail::tensor_impl_t<out_t, OpA::Rank()> tmp_out_;
      mutable out_t *ptr = nullptr;       

    public:
      using matxop = bool;
      using matx_transform_op = bool;
      using resample_poly_xform_op = bool;
      using value_type = out_t;            

      __MATX_INLINE__ std::string str() const { return "resample_poly(" + get_type_str(a_) + "," + get_type_str(f_) + ")";}
      __MATX_INLINE__ ResamplePolyOp(const OpA &a, const FilterType &f, index_t up, index_t down) : 
          a_(a), f_(f), up_(up), down_(down) 
      { 
        const index_t up_len = a_.Size(OpA::Rank() - 1) * up_;
        const index_t b_len = up_len / down_ + ((up_len % down_) ? 1 : 0);

        for (int r = 0; r < Rank(); r++) {
          out_dims_[r] = a_.Size(r);
        }

        out_dims_[OpA::Rank() - 1] = b_len;
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        return tmp_out_.template operator()<InWidth, OutWidth>(indices...);
      }
      
      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "resample_poly() only supports the CUDA executor currently");

        resample_poly_impl(cuda::std::get<0>(out), a_, f_, up_, down_, ex.getStream());
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }


      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<FilterType>()) {
          f_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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

        if constexpr (is_matx_op<FilterType>()) {
          f_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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
 * @brief 1D polyphase resampler
 * 
 * @tparam InType Type of input
 * @tparam FilterType Type of filter
 * @param in Input operator
 * @param f Filter operator
 * @param up Factor by which to upsample
 * @param down Factor by which to downsample
 * 
 * @returns Operator representing the filtered inputs
 */
template <typename InType, typename FilterType>
inline auto resample_poly(const InType &in, const FilterType &f,
                   index_t up, index_t down) {
  return detail::ResamplePolyOp(in, f, up, down);
}

}
