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
#include "matx/transforms/corr.h"

namespace matx
{
  namespace detail {
    template <typename OpA, typename OpB, typename PermDims>
    class CorrOp : public BaseOp<CorrOp<OpA, OpB, PermDims>>
    {
      private:
        using out_t = std::conditional_t<is_complex_v<typename OpA::value_type>, 
              typename OpA::value_type, typename OpB::value_type>;
        constexpr static int max_rank = cuda::std::max(OpA::Rank(), OpB::Rank());
        typename detail::base_type_t<OpA> a_;
        typename detail::base_type_t<OpB> b_;
        matxConvCorrMode_t mode_;
        matxConvCorrMethod_t method_;
        PermDims perm_;
        cuda::std::array<index_t, max_rank> out_dims_;
        mutable detail::tensor_impl_t<out_t, max_rank> tmp_out_;
        mutable out_t *ptr = nullptr; 

      public:
        using matxop = bool;
        using value_type = out_t;
        using matx_transform_op = bool;
        using conv_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          return "conv1d(" + get_type_str(a_) + "," + get_type_str(b_)  + ")";
        }

        __MATX_INLINE__ CorrOp(const OpA &A, const OpB &B, matxConvCorrMode_t mode, [[maybe_unused]] matxConvCorrMethod_t method, PermDims perm) : 
              a_(A), b_(B), mode_(mode), method_(method), perm_(perm) {

          // Currently when using the axis parameter the rank of inputs must be equal
          if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
            for (int r = 0; r < Rank(); r++) {
              const int axis = perm[r];
              if (axis == Rank() - 1) {
                const auto max_axis = cuda::std::max(a_.Size(r), b_.Size(r));
                const auto min_axis = cuda::std::min(a_.Size(r), b_.Size(r));
              
                if (mode_ == MATX_C_MODE_FULL) {
                  out_dims_[axis] = a_.Size(r) + b_.Size(r) - 1;
                }
                else if (mode_ == MATX_C_MODE_SAME) {
                  out_dims_[axis] = max_axis;
                }
                else if (mode_ == MATX_C_MODE_VALID) {
                  out_dims_[axis] = max_axis - min_axis + 1;
                }                  
              }
              else {
                out_dims_[axis] = b_.Size(r);
              }
            }                      
          }
          else {
            if constexpr (OpA::Rank() > OpB::Rank()) {
              for (int r = 0; r < Rank(); r++) {
                out_dims_[r] = a_.Size(r);
              }
            }
            else {
              for (int r = 0; r < Rank(); r++) {
                out_dims_[r] = b_.Size(r);
              }
            }

            const auto max_axis = cuda::std::max(a_.Size(OpA::Rank()-1), b_.Size(OpB::Rank()-1));
            const auto min_axis = cuda::std::min(a_.Size(OpA::Rank()-1), b_.Size(OpB::Rank()-1));
            if (mode_ == MATX_C_MODE_FULL) {
              out_dims_[max_rank-1] = max_axis + min_axis - 1;
            }
            else if (mode_ == MATX_C_MODE_SAME) {
              out_dims_[max_rank-1] = max_axis;
            }
            else if (mode_ == MATX_C_MODE_VALID) {
              out_dims_[max_rank-1] = max_axis - min_axis + 1;
            }                
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
          return max_rank;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          MATX_ASSERT_STR(!(is_host_executor_v<Executor> && method_ == MATX_C_METHOD_DIRECT), matxNotSupported, "direct corr() only supports the CUDA executor currently");
          MATX_STATIC_ASSERT_STR((Rank() == cuda::std::tuple_element_t<0, remove_cvref_t<Out>>::Rank()), 
                matxInvalidParameter, "corr: inputs and outputs must have same rank to use corr with axis parameter");
          if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
            corr_impl(permute(cuda::std::get<0>(out), perm_), a_, b_, mode_, method_, ex);
          }
          else {
            corr_impl(cuda::std::get<0>(out), a_, b_, mode_, method_, ex);
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
 * @brief Correlate two input operators
 * 
 * @tparam In1Type Type of first input
 * @tparam In2Type Type of second input
 * @param i1 First input operator
 * @param i2 Second input operator
 * @param mode Mode of correlation
 * @param method Method for correlation
 */
template <typename In1Type, typename In2Type>
__MATX_INLINE__ auto corr(const In1Type &i1, const In2Type &i2,
                   matxConvCorrMode_t mode, [[maybe_unused]] matxConvCorrMethod_t method) {
  return detail::CorrOp(i1, i2, mode, method, detail::no_permute_t{});     
}  


/**
 * @brief Correlate two input operators
 * 
 * @tparam In1Type Type of first input
 * @tparam In2Type Type of second input
 * @param i1 First input operator
 * @param i2 Second input operator
 * @param axis the axis to perform correlation
 * @param mode Mode of correlation
 * @param method Method for correlation
 */
template <typename In1Type, typename In2Type>
__MATX_INLINE__ auto corr(const In1Type &i1, const In2Type &i2,
                   const int32_t (&axis)[1],
                   matxConvCorrMode_t mode, [[maybe_unused]] matxConvCorrMethod_t method) {
  MATX_STATIC_ASSERT(In1Type::Rank() == In2Type::Rank(), "corr: inputs must have same rank to use conv1d with axis parameter");

  auto perm = detail::getPermuteDims<std::max(In1Type::Rank(), In2Type::Rank())>(axis);

  auto in1 = permute(i1, perm);
  auto in2 = permute(i2, perm);                    
  return detail::CorrOp(in1, in2, mode, method, perm);
}

}