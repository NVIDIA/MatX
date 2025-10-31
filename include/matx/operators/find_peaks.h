////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// sum rights reserved.
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
// DISCLAIMED. IN NO EVENT SHsum THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
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
#include "matx/transforms/find_peaks.h"

namespace matx {



namespace detail {
  template<typename OpA>
  class FindPeaksOp : public BaseOp<FindPeaksOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      typename remove_cvref_t<OpA>::value_type height_;
      typename remove_cvref_t<OpA>::value_type threshold_;

    public:
      using matxop = bool;
      using value_type = typename remove_cvref_t<OpA>::value_type;
      using matx_transform_op = bool;
      using find_peaks_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "find_peaks(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ FindPeaksOp(const OpA &a, value_type height, 
                                                value_type threshold) : 
                                                a_(a), height_(height), threshold_(threshold) { 
        MATX_LOG_TRACE("{} constructor: height={}, threshold={}", str(), height, threshold);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on find_peaks(). ie: (mtie(O, num_found) = find_peaks(A, height, threshold))");     
        static_assert(remove_cvref_t<decltype(cuda::std::get<1>(out))>::Rank() == 0 &&
                      std::is_same_v<typename remove_cvref_t<decltype(cuda::std::get<1>(out))>::value_type, int>, 
                      "Num elements output must be a scalar integer tensor");
        static_assert(std::is_same_v<typename remove_cvref_t<decltype(cuda::std::get<0>(out))>::value_type, index_t>, 
                      "Peak indices output must be a 1D matx::index_t tensor");
        find_peaks_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), a_, height_, threshold_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return remove_cvref_t<OpA>::Rank();
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
        return a_.Size(dim);
      }

  };
}


/**
 * Compute peak search of input
 *
 * Returns a tensor representing the indices of peaks found in the input operator. The first output parameter holds the indices
 * while the second holds the number of indices/peaks found. The output index tensor must be large enough to hold all of the peaks 
 * found or the behavior is undefined.
 *
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Number of right-most dimensions to reduce over
 *
 * @param in
 *   Input data to reduce
 * @param height
 *   Height threshold for peak detection. Values below this threshold are not considered peaks.
 * @param threshold
 *   Threshold for peak detection. Neighboring values must be larger in vertical distance than this threshold
 * @returns Operator with reduced values of peak search computed
 */
template <typename InType>
__MATX_INLINE__ auto find_peaks(const InType &in,
                                 typename InType::value_type height,
                                 typename InType::value_type threshold)
{
  static_assert(InType::Rank() == 1, "Input to find_peaks() must be rank 1");
  return detail::FindPeaksOp<decltype(in)>(in, height, threshold);
}

}
