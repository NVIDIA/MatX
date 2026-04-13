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
#include "matx/core/props.h"
#include "matx/operators/base_operator.h"
#include "matx/transforms/channelize_poly.h"

namespace matx {

namespace detail {

  // Forward declaration - a partial specialization of make_channelize_poly_op is defined below ChannelizePolyOp
  template <typename OpA, typename FilterType, typename List>
  struct make_channelize_poly_op;

  template<typename OpA, typename FilterType, typename... CurrentProps>
  class ChannelizePolyOp : public BaseOp<ChannelizePolyOp<OpA, FilterType, CurrentProps...>>
  {
    private:
      // Channelizer outputs are complex-valued due to the IFFT that is applied
      // to the filtered per-channel values. The output type is the higher of
      // the precisions of the input and filter. For example, if the input has
      // type cuda::std::complex<float> and the filter has type double, out_t
      // will be cuda::std::complex<double>.
      using default_out_t = cuda::std::common_type_t<
        ::matx::detail::complex_from_scalar_t<typename OpA::value_type>, ::matx::detail::complex_from_scalar_t<typename FilterType::value_type>>;
      using out_t = get_property_or<PropOutput, default_out_t, CurrentProps...>::type;
      static_assert(is_complex_v<out_t>, "Output type of channelize_poly must be complex");
      typename detail::base_type_t<OpA> a_;
      FilterType f_;
      index_t num_channels_;
      index_t decimation_factor_;
      cuda::std::array<index_t, OpA::Rank() + 1> out_dims_;
      mutable detail::tensor_impl_t<out_t, OpA::Rank() + 1> tmp_out_;
      mutable out_t *ptr = nullptr;
      mutable bool prerun_done_ = false;       

    public:
      using matxop = bool;
      using matx_transform_op = bool;
      using channelize_poly_xform_op = bool;
      using value_type = out_t;

      __MATX_INLINE__ std::string str() const { return "channelize_poly(" + get_type_str(a_) + "," + get_type_str(f_) + ")";}
      __MATX_INLINE__ ChannelizePolyOp(const OpA &a, const FilterType &f, index_t num_channels, index_t decimation_factor) :
          a_(a), f_(f), num_channels_(num_channels), decimation_factor_(decimation_factor)
      {
        MATX_LOG_TRACE("{} constructor: num_channels={}, decimation_factor={}", str(), num_channels, decimation_factor); 
        const index_t b_len = (a_.Size(OpA::Rank() - 1) + decimation_factor - 1) / decimation_factor;

        for (int r = 0; r < OpA::Rank()-1; r++) {
          out_dims_[r] = a_.Size(r);
        }

        out_dims_[Rank() - 2] = b_len;
        out_dims_[Rank() - 1] = num_channels;
      }

      // Const versions
      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return tmp_out_.template operator()<CapType>(indices...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return this->operator()<DefaultCapabilities>(indices...);
      }


      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, 
                                           detail::get_operator_capability<Cap>(a_, in),
                                           detail::get_operator_capability<Cap>(f_, in));
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank() + 1;
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "channelize_poly() only supports the CUDA executor currently");

        // The accumulator type should always be real (it will be promoted to complex when necessary), so the
        // default accumulator type is the output's inner type. The outputs of channelize_poly are always complex
        // due to the IFFT, but the filtering that is applied prior to the IFFT can be either real or complex.
        using accum_type = get_property_or<PropAccum, typename inner_op_type_t<value_type>::type, CurrentProps...>::type;
        channelize_poly_impl<decltype(cuda::std::get<0>(out)), decltype(a_), FilterType, accum_type>(
          cuda::std::get<0>(out), a_, f_, num_channels_, decimation_factor_, ex.getStream());
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

      template <typename... NewProps>
      constexpr auto props() const {
        using AllProps = typename merge_props_unique<type_list<CurrentProps...>, NewProps...>::type;
        return make_channelize_poly_op<OpA, FilterType, AllProps>::make(a_, f_, num_channels_, decimation_factor_);
      }
  };

  template <typename OpA, typename FilterType, typename... Props>
  struct make_channelize_poly_op<OpA, FilterType, type_list<Props...>> {
      template <typename A, typename F>
      static constexpr auto make(A &&a, F &&f, index_t num_channels, index_t decimation_factor) {
          using AType = remove_cvref_t<A>;
          using FType = remove_cvref_t<F>;
          return ChannelizePolyOp<AType, FType, Props...>(
              std::forward<A>(a), std::forward<F>(f), num_channels, decimation_factor);
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
 * @param num_channels Number of channels (or frequency sub-bands) to create.
 * @param decimation_factor Factor by which to downsample the input signal into the channels. When
 * decimation_factor equals num_channels, this is the maximally decimated (critically sampled) case.
 * When decimation_factor is less than num_channels, this is the oversampled case with overlapping
 * channels. Both integer and rational oversampling ratios are supported.
 * 
 * @returns Operator representing the channelized signal. The output tensor rank is one higher than the
 * input tensor rank. The first Rank-2 dimensions are all batch dimensions. The second-to-last dimension
 * is the sample dimension and the last dimension is the channel dimension. The per-channel output
 * size is (in.Size(InType::Rank()-1) + decimation_factor - 1) / decimation_factor.
 *
 * Let the letters M and D denote the number of channels and the decimation factor,
 * respectively. The case where M == D is the maximally decimated (critically sampled) case. In this case,
 * the channelizer generates M output samples (one per channel) for each M input samples. The case where
 * D < M is the oversampled case. In this case, the channelizer generates M output samples (one per channel)
 * for each D input samples. Thus, the total output sample count is roughly M/D times the input sample count.
 *
 * Logically, the polyphase channelizer delivers samples to a set of M commutator branches,
 * each representing a channel or frequency sub-band. For each output step, the samples in that commutator branch are convolved
 * with a phase of the prototype filter. The outputs of these convolutions are then the inputs to an M-point
 * IFFT. The results obtained depend on the order in which samples are delivered to the commutator branches
 * and the filter phase used per branch for each output.
 *
 * In MatX, samples are always delivered to the commutator branches in counter-clockwise order. In the
 * maximally decimated case, the sample-to-branch order is M-1, M-2, ..., 0 as in the
 * paper by Harris et al [1]. The filter phases are fixed per channel, so channel 0 corresponds to phase 0,
 * channel 1 corresponds to phase 1, etc. If comparing to another implementation that starts with a different
 * commutator branch (say, branch 0) and then proceeds in the same order (counter-clockwise), one can either prime the
 * other channelizer with a single zero input (discarding the outputs, if M outputs are generated after the
 * first sample) or prime the MatX channelizer with an appropriate number of zeros (e.g., M-1 to match an
 * implementation that starts at branch 0) to obtain equivalent results.
 *
 * The oversampled case, where D < M, is more complicated. The order
 * in which samples are delivered to the commutator branches is still counter-clockwise, but the samples
 * are logically delivered to branches D-1, D-2, ..., 0 for each set of D inputs. With each new set of D
 * inputs, the previous samples are logically shifted through the 2D filter bank. As an implementation detail,
 * rather than shifting samples, we rotate the filter phases applied to a given commutator branch. Each
 * branch rotates through a set of K = M / gcd(M, D) phases, where gcd(M, D) is the greatest common divisor
 * of M and D. Note that the maximally decimated case is a special case of the oversampled case where samples
 * shift through the 2D filter bank in groups of M, thus maintaining the same filter phase for each channel/branch.
 *
 * Comparing the oversampled case to another channelizer with a different convention can be difficult due
 * to the rotation of the filter phases. For example, for a channelizer that starts at branch 0 and delivers
 * M outputs after the first sample, that first sample will effectively rotate the filter phases as a result
 * of generating a set of M outputs. This is unlike the maximally decimated case where the filter phases are
 * fixed per channel, so we cannot simply prime the other channelizer with a single zero input to obtain equivalent
 * results. Instead, we need to provide enough zero inputs to the other channelizer so that the two channelizers
 * will start with the same filter phases. For a channelizer that starts at branch 0 and delivers M outputs
 * after a single sample, proceeding with the D-1, D-2, ..., 0 order thereafter, priming requires
 * 1 + (K - 1) * D zero samples, where K = M / gcd(M, D) as before.
 *
 * The polyphase channelizer supports the following properties:
 * - PropAccum: Type of accumulator. This type should always be real, but it will be promoted to
 *   complex when necessary.
 * - PropOutput: Type of output. This type should always be complex.
 *
 * The default accumulator type is the output type, but the output type is context-dependent. The
 * PropOutput property allows the user to explicitly set the output type and the PropAccum property
 * allows the user to explicitly set the accumulator type. See the examples section for an example
 * of the property syntax.
 *
 * [1] "Digital Receivers and Transmitters Using Polyphase Filter Banks for Wireless Communications",
 * F. J. Harris, C. Dick, M. Rice, IEEE Transactions on Microwave Theory and Techniques,
 * Vol. 51, No. 4, Apr. 2003.
 */
template <typename InType, typename FilterType>
inline auto channelize_poly(const InType &in, const FilterType &f, index_t num_channels, index_t decimation_factor) {
  return detail::ChannelizePolyOp<InType, FilterType>(in, f, num_channels, decimation_factor);
}

}
