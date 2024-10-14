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
#include "matx/transforms/channelize_poly.h"

namespace matx {

namespace detail {
  template<typename OpA, typename FilterType>
  class ChannelizePolyOp : public BaseOp<ChannelizePolyOp<OpA, FilterType>>
  {
    private:
      // Channelizer outputs are complex-valued due to the IFFT that is applied
      // to the filtered per-channel values. The output type is the higher of
      // the precisions of the input and filter. For example, if the input has
      // type cuda::std::complex<float> and the filter has type double, out_t
      // will be cuda::std::complex<double>.
      using out_t = cuda::std::common_type_t<
        complex_from_scalar_t<typename OpA::value_type>, complex_from_scalar_t<typename FilterType::value_type>>;
      typename detail::base_type_t<OpA> a_;
      FilterType f_;
      index_t num_channels_;
      index_t decimation_factor_;
      cuda::std::array<index_t, OpA::Rank() + 1> out_dims_;
      mutable detail::tensor_impl_t<out_t, OpA::Rank() + 1> tmp_out_;
      mutable out_t *ptr = nullptr;       

    public:
      using matxop = bool;
      using matx_transform_op = bool;
      using channelize_poly_xform_op = bool;
      using value_type = out_t;            

      __MATX_INLINE__ std::string str() const { return "channelize_poly(" + get_type_str(a_) + "," + get_type_str(f_) + ")";}
      __MATX_INLINE__ ChannelizePolyOp(const OpA &a, const FilterType &f, index_t num_channels, index_t decimation_factor) :
          a_(a), f_(f), num_channels_(num_channels), decimation_factor_(decimation_factor)
      { 
        const index_t b_len = (a_.Size(OpA::Rank() - 1) + num_channels - 1) / num_channels;

        for (int r = 0; r < OpA::Rank()-1; r++) {
          out_dims_[r] = a_.Size(r);
        }

        out_dims_[Rank() - 2] = b_len;
        out_dims_[Rank() - 1] = num_channels;
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) {
        return tmp_out_(indices...);
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "channelize_poly() only supports the CUDA executor currently");

        channelize_poly_impl(cuda::std::get<0>(out), a_, f_, num_channels_, decimation_factor_, ex.getStream());
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank() + 1;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }     

        if constexpr (is_matx_op<FilterType>()) {
          f_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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

        if constexpr (is_matx_op<FilterType>()) {
          f_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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
 * @brief 1D polyphase channelizer
 * 
 * @tparam InType Type of input.
 * @tparam FilterType Type of filter.
 * @param in Input operator that represents the input signal. The last dimension of this tensor
 * is assumed to contain a single input signal with all preceding dimensions being batch dimensions
 * that are channelized independently.
 * @param f Filter operator that represents the filter coefficients. This must be a 1D tensor.
 * @param num_channels Number of channels to create.
 * @param decimation_factor Factor by which to downsample the input signal into the channels. Currently,
 * the only supported value of decimation_factor is a value equal to num_channels. This corresponds to
 * the maximally decimated, or critically sampled, case. It is also possible for decimation_factor to
 * be less than num_channels, which corresponds to an oversampled case with overlapping channels, but
 * this implementation does not yet support oversampled cases.
 * 
 * @returns Operator representing the channelized signal. The output tensor rank is one higher than the
 * input tensor rank. The first Rank-2 dimensions are all batch dimensions. The second-to-last dimension
 * is the sample dimension and the last dimension is the channel dimension.
 */
template <typename InType, typename FilterType>
inline auto channelize_poly(const InType &in, const FilterType &f, index_t num_channels, index_t decimation_factor) {
  return detail::ChannelizePolyOp(in, f, num_channels, decimation_factor);
}

}
