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
    template <typename OpA, typename PermDims, typename ReductionOp>
    class ReduceOp : public BaseOp<ReduceOp<OpA, PermDims>>
    {
      private:
        static constexpr int ORank = permute_rank<OpA, PermDims, ReductionOp>::rank;
        typename detail::base_type_t<OpA> a_;
        PermDims perm_;
        ReductionOp reduction_op_;
        bool init_;
        cuda::std::array<index_t, ORank> out_dims_;
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, ORank> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr; 

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;
        using matx_transform_op = bool;
        using reduce_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          return "reduce(" + get_type_str(a_) + ")";
        }

        __MATX_INLINE__ ReduceOp(const OpA &A, PermDims perm, ReductionOp rop, bool init) : 
              a_(A), perm_(perm), reduction_op_(rop), init_(init) {
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

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return ORank;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          static_assert(is_cuda_executor_v<Executor>, "reduce() only supports the CUDA executor currently");

          if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
            reduce_impl(cuda::std::get<0>(out), a_, perm_, reduction_op_, ex.getStream(), init_);
          }
          else {
            reduce_impl(cuda::std::get<0>(out), a_, ex.getStream());
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
 * Perform a reduction
 *
 * Performs a reduction from tensor "in" ingo a 0D operator using reduction
 * operation ReduceOp. Without axes, reductions are performed over the entire
 * input operator.
 *
 * @tparam InType
 *   Input data type
 * @tparam ReduceOp
 *   Reduction operator type
 *
 * @param in
 *   Input data to compute the reduce 
 * @param op
 *   Reduction operator
 * @param init
 *   Initialize the data
 */
template <typename InType, typename ReduceOp>
__MATX_INLINE__ auto reduce(const InType &in, ReduceOp op, bool init = true)
{
  return detail::ReduceOp(in, detail::no_permute_t{}, op, init);     
}  



/**
 * Perform a reduction
 *
 * Performs a reduction from tensor "in" ingo a 0D operator using reduction
 * operation ReduceOp. In general, the reductions are performed over the
 * innermost dimensions, where the number of dimensions is the difference
 * between the input and number of axes. For example, when axes is the same as the
 * input rank, the reduction is performed over the entire tensor. For
 * anything higher, the reduction is performed across the number of ranks below
 * the input tensor that the output tensor is. For example, if the input tensor
 * is a 4D tensor and the reduction is on a single axis, the reduction is performed
 * across the innermost dimension of the input. 
 *
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Rank of dimension array
 * @tparam ReduceOp
 *   Reduction operator type
 *
 * @param in
 *   Input data to compute the reduce 
 * @param dims
 *   C-style array containing the dimensions to sum over
 * @param op
 *   Reduction operator
 * @param init
 *   Initialize the data
 */
template <typename InType, int D, typename ReduceOp>
__MATX_INLINE__ auto reduce(const InType &in, const int (&dims)[D], ReduceOp op, bool init = true)
{
  MATX_NVTX_START("reduce(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  
  static_assert(D < InType::Rank(), "reduce dimensions must be <= Rank of input");

  return detail::ReduceOp(in, detail::to_array(dims), op, init);
}



}
