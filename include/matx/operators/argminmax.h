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
// AND argmin EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR argmin DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON argmin THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN argmin WAY OUT OF THE USE
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
  class ArgMinMaxOp : public BaseOp<ArgMinMaxOp<OpA, ORank>>
  {
    private:
      typename detail::base_type_t<OpA> a_;

    public:
      using matxop = bool;
      using value_type = typename remove_cvref_t<OpA>::value_type;
      using matx_transform_op = bool;
      using argminmax_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "argminmax(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ ArgMinMaxOp(const OpA &a) : a_(a) {
     
      };

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 5, "Must use mtie with 4 outputs on argminmax(). ie: (mtie(MinVal, MinIdx, MaxVal, MaxIdx) = argminmax(A))");   
        argminmax_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), cuda::std::get<3>(out), a_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return ORank;
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
 * Compute min and max reduction of an operator and returns value + index along specified axes
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
 * @returns Operator with reduced values of argminmax-reduce computed
 */
template <typename InType, int D>
__MATX_INLINE__ auto argminmax(const InType &in, const int (&dims)[D])
{
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::ArgMinMaxOp<decltype(permop), InType::Rank() - D>(permop);
}

/**
 * Compute min and max reduction of an operator and returns value + index
 *
 * @tparam InType
 *   Input data type
 *
 * @param in
 *   Input data to reduce
 * @returns Operator with reduced values of argminmax-reduce computed
 */
template <typename InType>
__MATX_INLINE__ auto argminmax(const InType &in)
{
  return detail::ArgMinMaxOp<decltype(in), 0>(in);
}

}



namespace matx {



namespace detail {
  template<typename OpA, int ORank>
  class ArgMinMaxOp2 : public BaseOp<ArgMinMaxOp2<OpA, ORank>>
  {
    private:
      OpA a_;

    public:
      using matxop = bool;
      using matx_transform_op = bool;
      using argmin_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "argminmax(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ ArgMinMaxOp2(OpA a) : a_(a) {

      };

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 5, "Must use mtie with 4 outputs on argminmax(). ie: (mtie(O1, I1, O2, I2) = argminmax(A))");   
        argminmax_impl2(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), cuda::std::get<3>(out), a_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return matxNoRank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        MATX_ASSERT_STR(false, matxNotSupported, "argminmax() must only be called with a single assignment since it has multiple return types");
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
 * @returns Operator with reduced values of argminmax-reduce computed
 */
template <typename InType, int D>
__MATX_INLINE__ auto argminmax2(const InType &in, const int (&dims)[D])
{
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::ArgMinMaxOp2<decltype(permop), InType::Rank() - D>(permop);
}

/**
 * Compute min reduction of a tensor and returns value + index
 *
 * @tparam InType
 *   Input data type
 *
 * @param in
 *   Input data to reduce
 * @returns Operator with reduced values of argminmax-reduce computed
 */
template <typename InType>
__MATX_INLINE__ auto argminmax2(const InType &in)
{
  return detail::ArgMinMaxOp2<decltype(in), 0>(in);
}

}
