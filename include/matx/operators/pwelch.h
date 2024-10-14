////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// COpBright (c) 2023, NVIDIA Corporation
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
#include "matx/transforms/pwelch.h"

namespace matx
{
  namespace detail {
    template <typename OpX, typename OpW>
    class PWelchOp : public BaseOp<PWelchOp<OpX,OpW>>
    {
      private:
        typename detail::base_type_t<OpX> x_;
        typename detail::base_type_t<OpW> w_;

        index_t nperseg_;
        index_t noverlap_;
        index_t nfft_;
        cuda::std::array<index_t, 1> out_dims_;
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpX>::value_type, 1> tmp_out_;
        mutable typename remove_cvref_t<OpX>::value_type *ptr = nullptr; 

      public:
        using matxop = bool;
        using value_type = typename OpX::value_type::value_type;
        using matx_transform_op = bool;
        using pwelch_xform_op = bool;

        static_assert(is_complex_v<typename OpX::value_type>, "pwelch() must have a complex input type");

        __MATX_INLINE__ std::string str() const {
          return "pwelch(" + get_type_str(x_) + "," + get_type_str(w_) + ")";
        }

        __MATX_INLINE__ PWelchOp(const OpX &x, const OpW &w, index_t nperseg, index_t noverlap, index_t nfft) :
              x_(x), w_(w), nperseg_(nperseg), noverlap_(noverlap), nfft_(nfft) {

          MATX_STATIC_ASSERT_STR(OpX::Rank() == 1, matxInvalidDim, "pwelch:  Only input rank of 1 is supported presently");
          for (int r = 0; r < OpX::Rank(); r++) {
            out_dims_[r] = nfft_;
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
          return remove_cvref_t<OpX>::Rank();
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex)  const{
          static_assert(is_cuda_executor_v<Executor>, "pwelch() only supports the CUDA executor currently");
          pwelch_impl(cuda::std::get<0>(out), x_, w_, nperseg_, noverlap_, nfft_, ex.getStream());
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpX>()) {
            x_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          } 

          if constexpr (is_matx_op<OpW>()) {
            w_.PreRun(Shape(w_), std::forward<Executor>(ex));
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
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpX>()) {
            x_.PostRun(Shape(x_), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<OpW>()) {
            w_.PostRun(Shape(w_), std::forward<Executor>(ex));
          }

          matxFree(ptr);
        }               
    };
  }

  /**
   *  Operator to estimate the power spectral density of signal using Welch's method.
   *
   * @tparam xType
   *   Input time domain data type
   * @tparam wType
   *   Input window type
   * @param x
   *   Input time domain tensor
   * @param w
   *   Input window operator
   * @param nperseg
   *   Length of each segment
   * @param noverlap
   *   Number of points to overlap between segments.  Defaults to 0
   * @param nfft
   *   Length of FFT used per segment.  nfft >= nperseg.  Defaults to nfft = nperseg
   *
   * @returns Operator with power spectral density of x
   *
   */

  template <typename xType, typename wType>
    __MATX_INLINE__ auto pwelch(const xType& x, const wType& w, index_t nperseg, index_t noverlap, index_t nfft)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    return detail::PWelchOp(x, w, nperseg, noverlap, nfft);
  }

  template <typename xType>
    __MATX_INLINE__ auto pwelch(const xType& x, index_t nperseg, index_t noverlap, index_t nfft)
  {
    return detail::PWelchOp(x, std::nullopt, nperseg, noverlap, nfft);
  }
}
