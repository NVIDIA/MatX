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
#include "matx/transforms/reduce.h"

namespace matx {



namespace detail {
  template<typename OpA, int ORank>
  class MaxOp : public BaseOp<MaxOp<OpA, ORank>>
  {
    private:
      OpA a_;
      cuda::std::array<index_t, ORank> out_dims_;
      mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, ORank> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr;    

    public:
      using matxop = bool;
      using value_type = typename remove_cvref_t<OpA>::value_type;
      using matx_transform_op = bool;
      using max_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "max(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ MaxOp(OpA a) : a_(a) { 
        for (int r = 0; r < ORank; r++) {
          out_dims_[r] = a_.Size(r);
        }        
      };

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return tmp_out_(indices...);
      };

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        max_impl(cuda::std::get<0>(out), a_, ex);
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
      }       

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return out_dims_[dim];
      }

  };
}

/**
 * Compute max reduction of an operator along axes
 *
 * Returns an operator representing the max of all numbers in the reduction
 *
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 *
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @returns Operator with reduced values of max-reduce computed
 */
template <typename InType, int D>
__MATX_INLINE__ auto max(const InType &in, const int (&dims)[D])
{
  static_assert(D <= InType::Rank(), "reduction dimensions must be <= Rank of input");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::MaxOp<decltype(permop), InType::Rank() - D>(permop);
}

template <typename InType, int D>
[[deprecated("Use max() instead of rmax() for reductions")]]
__MATX_INLINE__ auto rmax(const InType &in, const int (&dims)[D])
{
  static_assert(D <= InType::Rank(), "reduction dimensions must be <= Rank of input");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::MaxOp<decltype(permop), InType::Rank() - D>(permop);
}

/**
 * Compute max reduction of an operator
 *
 * Returns an operator representing the max of all numbers in the reduction
 *
 * @tparam InType
 *   Input data type
 *
 * @param in
 *   Input data to reduce
 * @returns Operator with reduced values of max-reduce computed
 */
template <typename InType>
__MATX_INLINE__ auto max(const InType &in)
{
  return detail::MaxOp<decltype(in), 0>(in);
}

template <typename InType>
[[deprecated("Use max() instead of rmax() for reductions")]]
__MATX_INLINE__ auto rmax(const InType &in)
{
  return detail::MaxOp<decltype(in), 0>(in);
}

}
