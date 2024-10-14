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
// AND argmax EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR argmax DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON argmax THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN argmax WAY OUT OF THE USE
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
  class ArgMaxOp : public BaseOp<ArgMaxOp<OpA, ORank>>
  {
    private:
      typename detail::base_type_t<OpA> a_;

    public:
      using matxop = bool;
      using value_type = typename remove_cvref_t<OpA>::value_type;
      using matx_transform_op = bool;
      using argmax_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "argmax(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ ArgMaxOp(const OpA &a) : a_(a) { 
     
      };

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on argmax(). ie: (mtie(O, I) = argmax(A))");   
        argmax_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), a_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return matxNoRank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return 0;
      }

  };
}

/**
 * Compute min reduction of a tensor and returns value + index along specified axes
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
 * @returns Operator with reduced values of argmax-reduce computed
 */
template <typename InType, int D>
__MATX_INLINE__ auto argmax(const InType &in, const int (&dims)[D])
{
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::ArgMaxOp<decltype(permop), InType::Rank() - D>(permop);
}

/**
 * Compute min reduction of a tensor and returns value + index
 *
 * @tparam InType
 *   Input data type
 *
 * @param in
 *   Input data to reduce
 * @returns Operator with reduced values of argmax-reduce computed
 */
template <typename InType>
__MATX_INLINE__ auto argmax(const InType &in)
{
  return detail::ArgMaxOp<decltype(in), 0>(in);
}

}
