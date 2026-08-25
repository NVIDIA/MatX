////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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

#include <cuda/std/cmath>
#include <cuda/std/numbers>

#include "matx/core/constants.h"
#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx {
namespace experimental {

/**
 * @brief Parameters for bulk motion compensation of SAR FX data.
 */
struct SarBulkMocompParams {
  /** Frequency represented by sample floor(num_samples / 2), in Hz. */
  double phase_reference_frequency;

  /** Frequency difference between adjacent samples, in Hz. */
  double sample_frequency_spacing;

  /** Phase-history sign convention. Must be +1 or -1. */
  int sgn;
};

namespace detail {

template <typename FxOp, typename RangeOffsetOp>
class SarBulkMocompOp
    : public matx::BaseOp<SarBulkMocompOp<FxOp, RangeOffsetOp>> {
private:
  using fx_op_type = matx::detail::base_type_t<FxOp>;
  using range_offset_op_type = matx::detail::base_type_t<RangeOffsetOp>;

public:
  using matxop = bool;
  using value_type = typename fx_op_type::value_type;
  using range_type = typename range_offset_op_type::value_type;
  using fx_scalar_type = typename matx::inner_op_type_t<value_type>::type;
  using self_type = SarBulkMocompOp<FxOp, RangeOffsetOp>;

  template <typename Candidate>
  static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr bool
  ContainsBlockReduction() {
    if constexpr (requires {
                    typename Candidate::matx_jit_contains_block_reduction;
                  }) {
      return Candidate::matx_jit_contains_block_reduction::value;
    } else {
      return requires { typename Candidate::matx_jit_block_reduction; };
    }
  }

  using matx_jit_contains_block_reduction = cuda::std::bool_constant<
      ContainsBlockReduction<fx_op_type>() ||
      ContainsBlockReduction<range_offset_op_type>()>;

  template <typename CapType>
  struct RangeReadCapabilities : CapType::scalar_cap {
    // The lower-rank range expression is indexed explicitly. Suppress nested
    // JIT operators' thread-based bounds checks and let leaf tensors check the
    // mapped range indices instead.
    static constexpr bool pass_through_threads = CapType::jit;
  };

#ifdef MATX_EN_JIT
  struct JIT_Storage {
    typename matx::detail::inner_storage_or_self_t<fx_op_type> fx_;
    typename matx::detail::inner_storage_or_self_t<range_offset_op_type>
        range_offset_;
    range_type reference_phase_over_pi_per_meter_;
    range_type sample_phase_over_pi_per_meter_;
    index_t center_sample_;
  };

  __MATX_INLINE__ JIT_Storage ToJITStorage() const {
    return JIT_Storage{matx::detail::to_jit_storage(fx_),
                       matx::detail::to_jit_storage(range_offset_),
                       reference_phase_over_pi_per_meter_,
                       sample_phase_over_pi_per_meter_, center_sample_};
  }

  __MATX_INLINE__ std::string get_jit_class_name() const {
    return "JITSarBulkMocomp";
  }

  __MATX_INLINE__ auto get_jit_op_str() const {
    const std::string class_name = get_jit_class_name();
    return cuda::std::make_tuple(
        class_name,
        std::string("template <typename FxOp, typename RangeOffsetOp> struct ") +
            class_name + R"MATX_JIT( {
  using value_type = typename FxOp::value_type;
  using range_type = typename RangeOffsetOp::value_type;
  using fx_scalar_type = typename inner_op_type_t<value_type>::type;
  using matxop = bool;

  template <typename Candidate>
  static __MATX_INLINE__ constexpr __MATX_DEVICE__ bool
  ContainsBlockReduction()
  {
    if constexpr (requires {
                    typename Candidate::matx_jit_contains_block_reduction;
                  }) {
      return Candidate::matx_jit_contains_block_reduction::value;
    } else {
      return requires { typename Candidate::matx_jit_block_reduction; };
    }
  }

  using matx_jit_contains_block_reduction = cuda::std::bool_constant<
      ContainsBlockReduction<FxOp>() ||
      ContainsBlockReduction<RangeOffsetOp>()>;

  template <typename CapType>
  struct RangeReadCapabilities : CapType::scalar_cap {
    static constexpr bool pass_through_threads = CapType::jit;
  };

  typename detail::inner_storage_or_self_t<detail::base_type_t<FxOp>> fx_;
  typename detail::inner_storage_or_self_t<detail::base_type_t<RangeOffsetOp>>
      range_offset_;
  range_type reference_phase_over_pi_per_meter_;
  range_type sample_phase_over_pi_per_meter_;
  index_t center_sample_;

  template <typename CapType, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const
  {
    static_assert(sizeof...(Is) == Rank(),
                  "sar_bulk_mocomp index count must match operator rank");

    cuda::std::array<index_t, Rank()> fx_indices{
        static_cast<index_t>(indices)...};
    cuda::std::array<index_t, Rank() - 1> range_indices{};
    MATX_LOOP_UNROLL
    for (int32_t dim = 0; dim < Rank() - 1; ++dim) {
      range_indices[dim] = fx_indices[dim];
    }

    static constexpr bool fuse_block_reduction =
        CapType::jit && matx_jit_contains_block_reduction::value;
    using FxCap = cuda::std::conditional_t<
        fuse_block_reduction &&
            !ContainsBlockReduction<FxOp>(),
        typename CapType::scalar_cap, CapType>;
    using RangeCap = cuda::std::conditional_t<
        fuse_block_reduction && ContainsBlockReduction<RangeOffsetOp>(),
        CapType,
        RangeReadCapabilities<CapType>>;

    const range_type range_offset =
        get_value<RangeCap>(range_offset_, range_indices);
    const auto fx_value = get_value<FxCap>(fx_, fx_indices);

    if constexpr (fuse_block_reduction ||
                  CapType::ept == ElementsPerThread::ONE) {
      return ApplyCorrection(fx_value, range_offset, fx_indices[Rank() - 1]);
    } else {
      constexpr index_t EPT = static_cast<index_t>(CapType::ept);
      Vector<value_type, EPT> result;
      const index_t first_sample = fx_indices[Rank() - 1] * EPT;
      MATX_LOOP_UNROLL
      for (index_t lane = 0; lane < EPT; ++lane) {
        result.data[lane] = ApplyCorrection(fx_value.data[lane], range_offset,
                                            first_sample + lane);
      }
      return result;
    }
  }

  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()
  {
    return FxOp::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const
  {
    return fx_.Size(dim);
  }

private:
  __MATX_INLINE__ __MATX_DEVICE__ value_type
  ApplyCorrection(const value_type &fx_value, range_type range_offset,
                  index_t sample) const
  {
    const range_type sample_offset =
        static_cast<range_type>(sample - center_sample_);
    const range_type phase_over_pi_per_meter =
        cuda::std::fma(sample_offset, sample_phase_over_pi_per_meter_,
                       reference_phase_over_pi_per_meter_);
    const range_type phase_over_pi = phase_over_pi_per_meter * range_offset;
    range_type sin_phase;
    range_type cos_phase;
    if constexpr (cuda::std::is_same_v<range_type, double> &&
                  cuda::std::is_same_v<fx_scalar_type, float>) {
      // Keep the range-sensitive phase construction and argument reduction in
      // double, but evaluate the bounded angle with single-precision sincospif.
      // This avoids the costly full-range double-precision sincospi path on
      // GPUs with reduced FP64 throughput without first narrowing the large
      // unreduced phase.
      const double periods = cuda::std::nearbyint(phase_over_pi * 0.5);
      const double reduced_phase_over_pi =
          cuda::std::fma(periods, -2.0, phase_over_pi);
      float fast_sin_phase;
      float fast_cos_phase;
      sincospif(static_cast<float>(reduced_phase_over_pi), &fast_sin_phase,
                &fast_cos_phase);
      sin_phase = fast_sin_phase;
      cos_phase = fast_cos_phase;
    } else if constexpr (cuda::std::is_same_v<range_type, double>) {
      sincospi(phase_over_pi, &sin_phase, &cos_phase);
    } else {
      sincospif(phase_over_pi, &sin_phase, &cos_phase);
    }
    const value_type correction{static_cast<fx_scalar_type>(cos_phase),
                                static_cast<fx_scalar_type>(sin_phase)};
    return fx_value * correction;
  }
};
)MATX_JIT");
  }
#endif

  __MATX_INLINE__ SarBulkMocompOp(const FxOp &fx,
                                  const RangeOffsetOp &range_offset,
                                  const SarBulkMocompParams &params)
      : fx_(fx), range_offset_(range_offset),
        reference_phase_over_pi_per_meter_(
            static_cast<range_type>(-static_cast<double>(params.sgn) * 4.0 *
                                    params.phase_reference_frequency /
                                    matx::constants::speed_of_light)),
        sample_phase_over_pi_per_meter_(static_cast<range_type>(
            -static_cast<double>(params.sgn) * 4.0 *
            params.sample_frequency_spacing / matx::constants::speed_of_light)),
        center_sample_(fx.Size(Rank() - 1) / 2) {
    static_assert(!matx::is_dynamic_rank_op_v<FxOp> &&
                      !matx::is_dynamic_rank_op_v<RangeOffsetOp>,
                  "sar_bulk_mocomp does not currently support dynamic-rank "
                  "inputs");
    static_assert(
        !matx::is_distributed_tensor_v<FxOp> &&
            !matx::is_distributed_expression_v<FxOp> &&
            !matx::is_distributed_tensor_v<RangeOffsetOp> &&
            !matx::is_distributed_expression_v<RangeOffsetOp>,
        "sar_bulk_mocomp does not currently support distributed inputs");
    static_assert(Rank() >= 2,
                  "sar_bulk_mocomp requires FX data with rank 2 or greater");
    static_assert(range_offset_op_type::Rank() == Rank() - 1,
                  "sar_bulk_mocomp range offset rank must be one less than "
                  "the FX data rank");
    static_assert(
        cuda::std::is_same_v<value_type, cuda::std::complex<float>> ||
            cuda::std::is_same_v<value_type, cuda::std::complex<double>>,
        "sar_bulk_mocomp currently supports only complex<float> and "
        "complex<double> FX data");
    static_assert(cuda::std::is_same_v<range_type, float> ||
                      cuda::std::is_same_v<range_type, double>,
                  "sar_bulk_mocomp currently supports only float and double "
                  "range-offset value types");

    if (fx_.Size(Rank() - 1) <= 0) {
      MATX_THROW(matxInvalidSize,
                 "sar_bulk_mocomp requires at least one frequency sample");
    }
    if (params.sgn != -1 && params.sgn != 1) {
      MATX_THROW(matxInvalidParameter,
                 "sar_bulk_mocomp sgn must be +1 or -1");
    }
    if (static_cast<range_type>(params.sample_frequency_spacing) ==
        static_cast<range_type>(0)) {
      MATX_THROW(
          matxInvalidParameter,
          "sar_bulk_mocomp sample frequency spacing must be nonzero in the "
          "range-offset type");
    }

    MATX_LOOP_UNROLL
    for (int32_t dim = 0; dim < Rank() - 1; ++dim) {
      if (range_offset_.Size(dim) != fx_.Size(dim)) {
        MATX_THROW(
            matxInvalidSize,
            "sar_bulk_mocomp range-offset shape must match all FX dimensions "
            "except the final sample dimension");
      }
    }

    MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
  }

  __MATX_INLINE__ std::string str() const {
    return "sar_bulk_mocomp(" + matx::detail::get_type_str(fx_) + ")";
  }

  template <typename CapType, typename... Is>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto
  operator()(Is... indices) const {
    static_assert(sizeof...(Is) == Rank(),
                  "sar_bulk_mocomp index count must match operator rank");

    cuda::std::array<index_t, Rank()> fx_indices{
        static_cast<index_t>(indices)...};
    cuda::std::array<index_t, Rank() - 1> range_indices{};
    MATX_LOOP_UNROLL
    for (int32_t dim = 0; dim < Rank() - 1; ++dim) {
      range_indices[dim] = fx_indices[dim];
    }

    // Always false for the non-JIT path. This is retained here to keep the JIT and non-JIT
    // implementations as close as possible to help prevent implementation drift.
    static constexpr bool fuse_block_reduction =
        CapType::jit && matx_jit_contains_block_reduction::value;
    using FxCap = cuda::std::conditional_t<
        fuse_block_reduction &&
            !ContainsBlockReduction<fx_op_type>(),
        typename CapType::scalar_cap, CapType>;
    using RangeCap = cuda::std::conditional_t<
        fuse_block_reduction &&
            ContainsBlockReduction<range_offset_op_type>(),
        CapType,
        RangeReadCapabilities<CapType>>;

    const range_type range_offset =
        matx::detail::get_value<RangeCap>(range_offset_, range_indices);
    const auto fx_value = matx::detail::get_value<FxCap>(fx_, fx_indices);

    if constexpr (fuse_block_reduction ||
                  CapType::ept == matx::detail::ElementsPerThread::ONE) {
      return ApplyCorrection(fx_value, range_offset, fx_indices[Rank() - 1]);
    } else {
      constexpr index_t EPT = static_cast<index_t>(CapType::ept);
      matx::detail::Vector<value_type, EPT> result;
      const index_t first_sample = fx_indices[Rank() - 1] * EPT;
      MATX_LOOP_UNROLL
      for (index_t lane = 0; lane < EPT; ++lane) {
        result.data[lane] = ApplyCorrection(fx_value.data[lane], range_offset,
                                            first_sample + lane);
      }
      return result;
    }
  }

  template <typename... Is>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto
  operator()(Is... indices) const {
    return this->template operator()<matx::detail::DefaultCapabilities>(
        indices...);
  }

  template <matx::detail::OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto
  get_capability([[maybe_unused]] InType &in) const {
    if constexpr (Cap == matx::detail::OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
      const auto fx_jit_name =
          matx::detail::get_operator_capability<Cap>(fx_, in);
      const auto range_jit_name =
          matx::detail::get_operator_capability<Cap>(range_offset_, in);
      return get_jit_class_name() + "<" + fx_jit_name + "," +
             range_jit_name + ">";
#else
      return "";
#endif
    } else if constexpr (Cap ==
                         matx::detail::OperatorCapability::JIT_CACHE_KEY) {
#ifdef MATX_EN_JIT
      auto key = matx::detail::MakeJITCacheKeyForType<self_type>(
          "JITSarBulkMocomp");
      matx::detail::HashJITCacheValue(key, Rank());
      return matx::detail::combine_capabilities<Cap>(
          key, matx::detail::get_operator_capability<Cap>(fx_, in),
          matx::detail::get_operator_capability<Cap>(range_offset_, in));
#else
      return matx::detail::MakeInvalidJITCacheKey();
#endif
    } else if constexpr (Cap ==
                         matx::detail::OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
      const auto [key, value] = get_jit_op_str();
      if (in.find(key) == in.end()) {
        in[key] = value;
      }
      matx::detail::get_operator_capability<Cap>(fx_, in);
      matx::detail::get_operator_capability<Cap>(range_offset_, in);
      return true;
#else
      return false;
#endif
    } else if constexpr (Cap ==
                         matx::detail::OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
      return matx::detail::combine_capabilities<Cap>(
          true, matx::detail::get_operator_capability<Cap>(fx_, in),
          matx::detail::get_operator_capability<Cap>(range_offset_, in));
#else
      return false;
#endif
    } else if constexpr (
        Cap == matx::detail::OperatorCapability::ELEMENTS_PER_THREAD ||
        Cap == matx::detail::OperatorCapability::MAX_EPT_VEC_LOAD) {
      auto self_has_cap =
          matx::detail::capability_attributes<Cap>::default_value;
      if constexpr (matx_jit_contains_block_reduction::value) {
        if constexpr (ContainsBlockReduction<fx_op_type>() &&
                      ContainsBlockReduction<range_offset_op_type>()) {
          return matx::detail::combine_capabilities<Cap>(
              self_has_cap,
              matx::detail::get_operator_capability<Cap>(fx_, in),
              matx::detail::get_operator_capability<Cap>(range_offset_, in));
        } else if constexpr (ContainsBlockReduction<fx_op_type>()) {
          return matx::detail::get_operator_capability<Cap>(fx_, in);
        } else {
          return matx::detail::get_operator_capability<Cap>(range_offset_, in);
        }
      } else {
        return matx::detail::combine_capabilities<Cap>(
            self_has_cap,
            matx::detail::get_operator_capability<Cap>(fx_, in));
      }
    } else if constexpr (
        Cap == matx::detail::OperatorCapability::SET_ELEMENTS_PER_THREAD) {
      auto fx_in = in;
      auto range_in = in;
      if constexpr (matx_jit_contains_block_reduction::value) {
        if constexpr (!ContainsBlockReduction<fx_op_type>()) {
          fx_in.ept = matx::detail::ElementsPerThread::ONE;
        }
        if constexpr (!ContainsBlockReduction<range_offset_op_type>()) {
          range_in.ept = matx::detail::ElementsPerThread::ONE;
        }
      } else {
        range_in.ept = matx::detail::ElementsPerThread::ONE;
      }
      return matx::detail::combine_capabilities<Cap>(
          matx::detail::capability_attributes<Cap>::default_value,
          matx::detail::get_operator_capability<Cap>(fx_, fx_in),
          matx::detail::get_operator_capability<Cap>(range_offset_, range_in));
    } else if constexpr (
        Cap == matx::detail::OperatorCapability::DYN_SHM_SIZE) {
      return matx::detail::get_operator_capability<Cap>(fx_, in) +
             matx::detail::get_operator_capability<Cap>(range_offset_, in);
    } else {
      auto self_has_cap =
          matx::detail::capability_attributes<Cap>::default_value;
      return matx::detail::combine_capabilities<Cap>(
          self_has_cap, matx::detail::get_operator_capability<Cap>(fx_, in),
          matx::detail::get_operator_capability<Cap>(range_offset_, in));
    }
  }

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t
  Rank() {
    return fx_op_type::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto
  Size(int dim) const noexcept {
    return fx_.Size(dim);
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun(ShapeType &&shape,
                              Executor &&exec) const noexcept {
    if constexpr (matx::is_matx_op<FxOp>()) {
      fx_.PreRun(cuda::std::forward<ShapeType>(shape),
                 cuda::std::forward<Executor>(exec));
    }
    if constexpr (matx::is_matx_op<RangeOffsetOp>()) {
      range_offset_.PreRun(cuda::std::forward<ShapeType>(shape),
                           cuda::std::forward<Executor>(exec));
    }
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun(ShapeType &&shape,
                               Executor &&exec) const noexcept {
    if constexpr (matx::is_matx_op<FxOp>()) {
      fx_.PostRun(cuda::std::forward<ShapeType>(shape),
                  cuda::std::forward<Executor>(exec));
    }
    if constexpr (matx::is_matx_op<RangeOffsetOp>()) {
      range_offset_.PostRun(cuda::std::forward<ShapeType>(shape),
                            cuda::std::forward<Executor>(exec));
    }
  }

private:
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ value_type
  ApplyCorrection(const value_type &fx_value, range_type range_offset,
                  index_t sample) const {
    const range_type sample_offset =
        static_cast<range_type>(sample - center_sample_);
    const range_type phase_over_pi_per_meter = cuda::std::fma(
        sample_offset, sample_phase_over_pi_per_meter_,
        reference_phase_over_pi_per_meter_);
    const range_type phase_over_pi = phase_over_pi_per_meter * range_offset;
    range_type sin_phase;
    range_type cos_phase;
#if defined(__CUDA_ARCH__)
    if constexpr (cuda::std::is_same_v<range_type, double> &&
                  cuda::std::is_same_v<fx_scalar_type, float>) {
      // A float output cannot retain double-precision trig results. Reduce the
      // large phase modulo 2 in double before converting the bounded angle to
      // float, which makes the lower-cost sincospif path accurate enough for
      // the destination type.
      const double periods = cuda::std::nearbyint(phase_over_pi * 0.5);
      const double reduced_phase_over_pi =
          cuda::std::fma(periods, -2.0, phase_over_pi);
      float fast_sin_phase;
      float fast_cos_phase;
      sincospif(static_cast<float>(reduced_phase_over_pi), &fast_sin_phase,
                &fast_cos_phase);
      sin_phase = fast_sin_phase;
      cos_phase = fast_cos_phase;
    } else if constexpr (cuda::std::is_same_v<range_type, double>) {
      sincospi(phase_over_pi, &sin_phase, &cos_phase);
    } else {
      sincospif(phase_over_pi, &sin_phase, &cos_phase);
    }
#else
    const range_type reduced_phase_over_pi = cuda::std::remainder(
        phase_over_pi, static_cast<range_type>(2));
    const range_type phase =
        static_cast<range_type>(cuda::std::numbers::pi) *
        reduced_phase_over_pi;
    sin_phase = cuda::std::sin(phase);
    cos_phase = cuda::std::cos(phase);
#endif
    const value_type correction{static_cast<fx_scalar_type>(cos_phase),
                                static_cast<fx_scalar_type>(sin_phase)};
    return fx_value * correction;
  }

  mutable fx_op_type fx_;
  mutable range_offset_op_type range_offset_;
  range_type reference_phase_over_pi_per_meter_;
  range_type sample_phase_over_pi_per_meter_;
  index_t center_sample_;
};

} // namespace detail

/**
 * @brief Apply bulk motion compensation to SAR phase-history data in the FX
 * domain.
 *
 * The final two dimensions of \p fx are interpreted as pulse and frequency
 * sample. Every leading dimension is a batch dimension. The range offset must
 * match all FX dimensions except the final frequency-sample dimension.
 *
 * Sample floor(num_samples / 2) has frequency
 * \p params.phase_reference_frequency. The operator applies
 * exp(j * -sgn * 4*pi/c * frequency * target_reference_range) to every FX
 * sample. Phase construction and reduction use the value type of
 * \p target_reference_range, which must currently be float or double. On CUDA,
 * double-range phase for complex<float> FX data uses a float trigonometric
 * evaluation after double-precision argument reduction.
 *
 * @tparam FxOp FX tensor or operator type.
 * @tparam TargetRangeOp Target-reference-range tensor or operator type.
 * @param fx Complex FX-domain data with shape [batch..., pulses, samples].
 * @param target_reference_range Target per-pulse reference range with shape
 * [batch..., pulses].
 * @param params Frequency grid and phase-sign parameters.
 * @return A lazy, allocation-free motion-compensated operator.
 */
template <typename FxOp, typename TargetRangeOp>
  requires (is_matx_op_c<FxOp> && is_matx_op_c<TargetRangeOp>)
__MATX_INLINE__ auto
sar_bulk_mocomp(const FxOp &fx, const TargetRangeOp &target_reference_range,
                const SarBulkMocompParams &params) {
  return detail::SarBulkMocompOp<FxOp, TargetRangeOp>(
      fx, target_reference_range, params);
}

/**
 * @brief Change the bulk motion-compensation reference of SAR FX data.
 *
 * This overload can be used to effectively change the motion-compensation
 * point. It applies compensation for target_reference_range minus
 * initial_reference_range. Both range operators must have the same shape as
 * all FX dimensions except its final frequency-sample dimension.
 *
 * @tparam FxOp FX tensor or operator type.
 * @tparam InitialRangeOp Initial-reference-range tensor or operator type.
 * @tparam TargetRangeOp Target-reference-range tensor or operator type.
 * @param fx Complex FX-domain data with shape [batch..., pulses, samples].
 * @param initial_reference_range Existing per-pulse reference range.
 * @param target_reference_range Desired per-pulse reference range.
 * @param params Frequency grid and phase-sign parameters.
 * @return A lazy, allocation-free reference-change operator.
 */
template <typename FxOp, typename InitialRangeOp, typename TargetRangeOp>
  requires (is_matx_op_c<FxOp> && is_matx_op_c<InitialRangeOp> &&
            is_matx_op_c<TargetRangeOp>)
__MATX_INLINE__ auto
sar_bulk_mocomp(const FxOp &fx, const InitialRangeOp &initial_reference_range,
                const TargetRangeOp &target_reference_range,
                const SarBulkMocompParams &params) {
  return detail::SarBulkMocompOp<FxOp, decltype(target_reference_range -
                                                initial_reference_range)>(
      fx, target_reference_range - initial_reference_range, params);
}

} // namespace experimental
} // namespace matx
