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
#include "matx/transforms/conv.h"

namespace matx
{
  namespace detail {
    template <typename OpA, typename PermDims>
    class SoftmaxOp : public BaseOp<SoftmaxOp<OpA, PermDims>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        PermDims perm_;
        cuda::std::array<index_t, OpA::Rank()> out_dims_;
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, OpA::Rank()> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr; 

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;
        using matx_transform_op = bool;
        using softmax_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          return "softmax(" + get_type_str(a_) + ")";
        }

        __MATX_INLINE__ SoftmaxOp(const OpA &A, PermDims perm) : 
              a_(A), perm_(perm) {
          for (int r = 0; r < OpA::Rank(); r++) {
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
          return OpA::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          static_assert(is_cuda_executor_v<Executor>, "softmax() only supports the CUDA executor currently");

          if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
            softmax_impl(cuda::std::get<0>(out), a_, perm_, ex.getStream());
          }
          else {
            softmax_impl(cuda::std::get<0>(out), a_, ex.getStream());
          }
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

          matxFree(ptr);
        }   
    };
  }

/**
 * Calculate the softmax of values in a tensor treated as a flat vector
 *
 * softmax computes the exponential of each value divided by the sum of the exponentials
 * of items in the reduced set. The axes in which to perform the softmax over determine
 * which axes the sum will be computed over, but the input tensor rank and sizes match
 * between input and output. Note that traditional definitions of softmax are simply 
 * exp(x)/sum(exp(x)), but this is not how most libraries are implemented. Instead, x 
 * is biased by a correction factor of max(x).
 *
 * @tparam InType
 *   Input data type
 *
 * @param in
 *   Input data to compute the softmax 
 */
template <typename InType>
__MATX_INLINE__ auto softmax(const InType &in)
{
  return detail::SoftmaxOp(in, detail::no_permute_t{});     
}  



/**
 * Calculate the softmax of values in a tensor treated as a flat vector
 *
 * softmax computes the exponential of each value divided by the sum of the exponentials
 * of items in the reduced set. The axes in which to perform the softmax over determine
 * which axes the sum will be computed over, but the input tensor rank and sizes match
 * between input and output. Note that traditional definitions of softmax are simply 
 * exp(x)/sum(exp(x)), but this is not how most libraries are implemented. Instead, x 
 * is biased by a correction factor of max(x).
 *
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Rank of dimension array
 *
 * @param in
 *   Input data to compute the softmax 
 * @param dims
 *   C-style array containing the dimensions to sum over
 */
template <typename InType, int D>
__MATX_INLINE__ auto softmax(const InType &in, const int (&dims)[D])
{
#ifdef __CUDACC__
  MATX_NVTX_START("softmax(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "softmax dimensions must be <= Rank of input");

  return detail::SoftmaxOp(in, detail::to_array(dims));
#endif  
}



}
