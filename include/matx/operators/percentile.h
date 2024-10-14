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
#include "matx/operators/permute.h"
#include "matx/transforms/percentile.h"

namespace matx {


namespace detail {
  template<typename OpA, int ORank>
  class PercentileOp : public BaseOp<PercentileOp<OpA,ORank>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      uint32_t q_;
      PercentileMethod method_;
      cuda::std::array<index_t, ORank> out_dims_; 
      mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, ORank> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr; 

    public:
      using matxop = bool;
      using value_type = typename remove_cvref_t<OpA>::value_type;
      using matx_transform_op = bool;
      using prod_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "percentile(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ PercentileOp(const OpA &a, unsigned char q, PercentileMethod method) : a_(a), q_(q), method_(method) {
        for (int r = 0; r < ORank; r++) {
          out_dims_[r]    = (r == ORank - 1) ? 1 : a_.Size(r);
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
        percentile_impl(cuda::std::get<0>(out), a_, q_, method_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return ORank;
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
        return out_dims_[dim];
      }

  };
}

/**
 * Compute product of numbers along axes
 *
 * Returns a tensor representing the product of all items in the reduction
 *
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 *
 * @param in
 *   Input data to reduce
 * @param q
 *   Percentile to compute (between 0-100)
 * @param dims
 *   Array containing dimensions to compute over
 * @param method
 *   Method of interpolation
 * @returns Operator with reduced values of prod-reduce computed
 */
template <typename InType, int D>
__MATX_INLINE__ auto percentile(const InType &in, unsigned char q, const int (&dims)[D], PercentileMethod method = PercentileMethod::LINEAR)
{
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  MATX_ASSERT_STR(q < 100, matxInvalidParameter, "Percentile must be < 100");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::PercentileOp<decltype(permop), InType::Rank() - D>(permop, q, method);
}

/**
 * Compute product of numbers
 *
 * Returns a tensor representing the product of all items in the reduction
 *
 * @tparam InType
 *   Input data type
 *
 * @param in
 *   Input data to reduce
 * @param q
 *   Percentile to compute (between 0-100)
 * @param method
 *   Method of interpolation
 * @returns Operator with reduced values of prod-reduce computed
 */
template <typename InType>
__MATX_INLINE__ auto percentile(const InType &in, unsigned char q, PercentileMethod method = PercentileMethod::LINEAR)
{
  return detail::PercentileOp<decltype(in), 0>(in, q, method);
}

}