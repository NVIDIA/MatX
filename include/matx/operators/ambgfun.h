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
#include "matx/core/operator_options.h"

#include "matx/transforms/ambgfun.h"

namespace matx
{
  namespace detail {
    template <typename OpX, typename OpY>
    class AmbgFunOp : public BaseOp<AmbgFunOp<OpX, OpY>>
    {
      private:
        // Make these base_type once we get rid of std::optional
        mutable OpX x_;
        mutable OpY y_;
        double fs_;
        AMBGFunCutType_t cut_;
        float cut_val_;
        cuda::std::array<index_t, 2> out_dims_;
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpX>::value_type, 2> tmp_out_;
        mutable typename remove_cvref_t<OpX>::value_type *ptr = nullptr;
        mutable bool prerun_done_ = false;         

      public:
        using matxop = bool;
        using value_type = typename OpX::value_type;
        using matx_transform_op = bool;
        using ambgfun_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          if constexpr (std::is_same_v<OpY, EmptyY>) {
            return "ambgfun(" + get_type_str(x_) + ")";
          }
          else {
            return "ambgfun(" + get_type_str(x_) + "," + get_type_str(y_) + ")";
          }
        }

        __MATX_INLINE__ AmbgFunOp(const OpX &x, const OpY &y, double fs, AMBGFunCutType_t cut, float cut_val) : 
              x_(x), y_(y), fs_(fs), cut_(cut), cut_val_(cut_val) {
          MATX_LOG_TRACE("{} constructor: fs={}, cut={}", str(), fs, static_cast<int>(cut));
          static_assert(OpX::Rank() == 1, "Input to ambgfun must be rank 1");                
          if (cut == AMBGFUN_CUT_TYPE_2D) {
            out_dims_[0] = 2 * x_.Size(0) - 1;
            out_dims_[1] = (index_t)cuda::std::pow(2.0, (double)std::ceil(std::log2(2 * x_.Size(0) - 1)));
          }
          else if (cut == AMBGFUN_CUT_TYPE_DELAY) {
            out_dims_[0] = 1;
            out_dims_[1] = (index_t)cuda::std::pow(2.0, (double)std::ceil(std::log2(2 * x_.Size(0) - 1)));            
          }
          else if (cut == AMBGFUN_CUT_TYPE_DOPPLER) {
            out_dims_[0] = 1;
            out_dims_[1] = 2 * x_.Size(0) - 1;               
          }
          else {
            MATX_ASSERT_STR(false, matxInvalidParameter, "Invalid cut type in ambgfun()");
          }
        }


        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<CapType>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<DefaultCapabilities>(indices...);
        }
        
        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          // No specific capabilities enforced
          auto self_has_cap = capability_attributes<Cap>::default_value;
          if constexpr (std::is_same_v<OpY, EmptyY>) {
            // Single-input ambgfun: only combine capabilities from x_
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(x_, in));
          } else {
            // Two-input ambgfun: combine capabilities from both x_ and y_
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(x_, in), detail::get_operator_capability<Cap>(y_, in));
          }
        }          

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return 2;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }   

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          static_assert(is_cuda_executor_v<Executor>, "ambgfun() only supports the CUDA executor currently");
          static_assert(cuda::std::tuple_element_t<0, remove_cvref_t<Out>>::Rank() == 2, "Output tensor of ambgfun must be 2D");
          ambgfun_impl(cuda::std::get<0>(out), x_, y_, fs_, cut_, cut_val_, ex.getStream());
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpX>()) {
            x_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }     

          if constexpr (is_matx_op<OpY>()) {
            y_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }             
        }      

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if (prerun_done_) {
            return;
          }

          InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));         

          detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

          prerun_done_ = true;
          Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpX>()) {
            x_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<OpY>()) {
            y_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          matxFree(ptr); 
        }
    };
  }



/**
 * Cross-ambiguity function
 *
 * Generates a cross-ambiguity magnitude function from inputs x and y. The
 * ambiguity function generates a 2D delay vs doppler matrix of the cross
 * ambiguity of x and y.
 *
 * @tparam XTensor
 *   x vector type
 * @tparam YTensor
 *   Y vector type
 * @param x
 *   First input signal
 * @param y
 *   Second input signal
 * @param fs
 *   Sampling frequency
 * @param cut
 *   Type of cut. 2D is effectively no cut. Delay cut returns a cut with zero
 * time delay. Doppler generates a cut with zero Doppler shift. Note that in
 * both Delay and Doppler mode, the output matrix must be a 2D tensor where the
 * first dimension is 1 to match the type of 2D mode.
 * @param cut_val
 *   Value to perform the cut at
 * @returns
 *   2D output matrix where rows are the Doppler (Hz) shift and columns are the
 * delay in seconds.
 *
 */
template <typename XTensor, typename YTensor>
__MATX_INLINE__ auto ambgfun(const XTensor &x,
                    const YTensor &y, double fs, AMBGFunCutType_t cut,
                    float cut_val = 0.0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  return detail::AmbgFunOp(x, y, fs, cut, cut_val);
}

/**
 * Ambiguity function
 *
 * Generates an ambiguity magnitude function from input signal x. The ambiguity
 * function generates a 2D delay vs doppler matrix of the input signal.
 *
 * @tparam XTensor
 *   x vector type
 * @param x
 *   First input signal
 * @param fs
 *   Sampling frequency
 * @param cut
 *   Type of cut. 2D is effectively no cut. Delay cut returns a cut with zero
 * time delay. Doppler generates a cut with zero Doppler shift. Note that in
 * both Delay and Doppler mode, the output matrix must be a 2D tensor where the
 * first dimension is 1 to match the type of 2D mode.
 * @param cut_val
 *   Value to perform the cut at
 * @returns
 *   2D output matrix where rows are the Doppler (Hz) shift and columns are the
 * delay in seconds.
 *
 */
template <typename XTensor>
__MATX_INLINE__ auto ambgfun(const XTensor &x,
                    double fs, AMBGFunCutType_t cut, float cut_val = 0.0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  detail::EmptyY nil;
  return detail::AmbgFunOp(x, nil, fs, cut, cut_val);
}

}
