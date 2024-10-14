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
#include "matx/core/nvtx.h"
#include "matx/core/operator_utils.h"
#include "matx/transforms/filter.h"

namespace matx
{

namespace detail {
  template<typename OpA, typename FilterType, size_t NR, size_t NNR>
  class FilterOp : public BaseOp<FilterOp<OpA, FilterType, NR, NNR>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      cuda::std::array<FilterType, NR> h_rec_;
      cuda::std::array<FilterType, NNR> h_nonrec_;
      cuda::std::array<index_t, OpA::Rank()> out_dims_;
      mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, OpA::Rank()> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr; 

    public:
      using matxop = bool;
      using value_type = void;
      using matx_transform_op = bool;
      using filter_xform_op = bool;

      __MATX_INLINE__ std::string str() const { 
        return "filter(" + get_type_str(a_) + ")";
      }
      __MATX_INLINE__ FilterOp(const OpA &a, const cuda::std::array<FilterType, NR> h_rec,
            const cuda::std::array<FilterType, NNR> h_nonrec) : a_(a), h_rec_(h_rec), h_nonrec_(h_nonrec) { 
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

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "filter() only supports the CUDA executor currently");   

        filter_impl(cuda::std::get<0>(out), a_, h_rec_, h_nonrec_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
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

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
      }

  };
}

/**
 * FIR and IIR filtering without a plan
 *
 * matxFilter_t provides an interface for executing recursive (IIR) and
 *non-recursive (FIR) filters. The IIR filter uses the algorithm from "S. Maleki
 *and M. Burtscher. "Automatic Hierarchical Parallelization of Linear
 *Recurrences." 23rd ACM International Conference on Architectural Support for
 *Programming Languages and Operating Systems. March 2018." for an optimized
 *implementation on highly-parallel processors. While the IIR implementation is
 *fast for recursive filters, it is inefficient for non-recursive filtering. If
 *the number of recursive coefficients is 0, the filter operation will revert to
 *use an algorithm optimized for non-recursive filters.
 *
 * @note If you are only using non-recursive filters, it's advised to use the
 *convolution API directly instead since it can be easier to use.
 *
 * @tparam OpA
 *   Input type
 * @tparam NR
 *   Number of recursive coefficients
 * @tparam NNR
 *   Number of non-recursive coefficients
 * @tparam FilterType
 *   Type of filter
 *
 * @param a
 *   Input operator
 * @param h_rec
 *   Recursive coefficients
 * @param h_nonrec
 *   Non-recursive coefficients
 *
 **/
  template <typename OpA, size_t NR, size_t NNR, typename FilterType>
  __MATX_INLINE__ auto filter(const OpA &a, 
            const cuda::std::array<FilterType, NR> h_rec,
            const cuda::std::array<FilterType, NNR> h_nonrec) {
    return detail::FilterOp(a, h_rec, h_nonrec);
  }

}