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

#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"
#include "matx/operators/scalar_internal.h"

namespace matx {
namespace detail {
template <typename OpA, typename MathType>
class UnwrapOp : public BaseOp<UnwrapOp<OpA, MathType>> {
public:
  using matxop = bool;
  using value_type = typename OpA::value_type;
  using self_type = UnwrapOp<OpA, MathType>;

  // Propagate dynamic tensor marker through expression tree
  using dynamic_tensor_expr = cuda::std::bool_constant<
      is_dynamic_tensor_v<OpA> || is_dynamic_rank_op_v<OpA>>;

#ifdef MATX_EN_JIT
  struct JIT_Storage {
    typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> op_;
    int axis_;
    MathType discont_;
    MathType period_;
    MathType half_period_;
  };

  JIT_Storage ToJITStorage() const {
    return JIT_Storage{detail::to_jit_storage(op_), axis_, discont_, period_, half_period_};
  }

  __MATX_INLINE__ std::string get_jit_class_name() const {
    return "JITUnwrapOp";
  }

  __MATX_INLINE__ auto get_jit_op_str() const {
    const int actual_rank = jit_rank();
    std::string func_name = get_jit_class_name();
    cuda::std::array<index_t, Rank()> out_dims_;
    for (int i = 0; i < actual_rank; ++i) {
      out_dims_[i] = Size(i);
    }

    return cuda::std::make_tuple(
      func_name,
      std::string("template <typename OpA, typename MathType> struct " + func_name + " {\n") +
        "  using value_type = typename OpA::value_type;\n" +
        "  using matxop = bool;\n" +
        "  constexpr static int Rank_ = " + std::to_string(actual_rank) + ";\n" +
        "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = { " +
            detail::array_to_string(out_dims_, actual_rank) + " };\n" +
        "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> op_;\n" +
        "  int axis_;\n" +
        "  MathType discont_;\n" +
        "  MathType period_;\n" +
        "  MathType half_period_;\n" +
        "  template <typename CapType, typename... Is>\n" +
        "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const {\n" +
        "    if constexpr (Rank_ == 0) {\n" +
        "      return get_value<CapType>(op_);\n" +
        "    }\n" +
        "    else {\n" +
        "      constexpr index_t EPT = static_cast<index_t>(CapType::ept);\n" +
        "      auto get_lane_scalar = [](const auto &v, index_t lane) {\n" +
        "        if constexpr (CapType::ept == ElementsPerThread::ONE) {\n" +
        "          (void)lane;\n" +
        "          return static_cast<MathType>(v);\n" +
        "        }\n" +
        "        else {\n" +
        "          return static_cast<MathType>(v.data[lane]);\n" +
        "        }\n" +
        "      };\n" +
        "      cuda::std::array<index_t, Rank_> idx{indices...};\n" +
        "      const index_t out_idx = idx[axis_];\n" +
        "      const auto cur = get_value<CapType>(op_, idx);\n" +
        "      cuda::std::array<MathType, static_cast<size_t>(EPT)> correction{};\n" +
        "      if (out_idx != 0) {\n" +
        "        cuda::std::array<index_t, Rank_> seq_idx = idx;\n" +
        "        seq_idx[axis_] = 0;\n" +
        "        auto prev = get_value<CapType>(op_, seq_idx);\n" +
        "        const MathType neg_half_period = -half_period_;\n" +
        "        for (index_t i = 1; i <= out_idx; i++) {\n" +
        "          seq_idx[axis_] = i;\n" +
        "          const auto next = get_value<CapType>(op_, seq_idx);\n" +
        "          MATX_LOOP_UNROLL\n" +
        "          for (index_t lane = 0; lane < EPT; lane++) {\n" +
        "            const MathType next_s = get_lane_scalar(next, lane);\n" +
        "            const MathType prev_s = get_lane_scalar(prev, lane);\n" +
        "            const MathType delta = next_s - prev_s;\n" +
        "            MathType delta_mod = static_cast<MathType>(scalar_internal_fmod(delta + half_period_, period_));\n" +
        "            if (delta_mod < static_cast<MathType>(0)) {\n" +
        "              delta_mod += period_;\n" +
        "            }\n" +
        "            delta_mod -= half_period_;\n" +
        "            if (delta_mod == neg_half_period && delta > static_cast<MathType>(0)) {\n" +
        "              delta_mod = half_period_;\n" +
        "            }\n" +
        "            MathType phase_correction = delta_mod - delta;\n" +
        "            if (cuda::std::abs(delta) < discont_) {\n" +
        "              phase_correction = static_cast<MathType>(0);\n" +
        "            }\n" +
        "            correction[static_cast<size_t>(lane)] += phase_correction;\n" +
        "          }\n" +
        "          prev = next;\n" +
        "        }\n" +
        "      }\n" +
        "      if constexpr (CapType::ept == ElementsPerThread::ONE) {\n" +
        "        return static_cast<value_type>(get_lane_scalar(cur, 0) + correction[0]);\n" +
        "      }\n" +
        "      else {\n" +
        "        Vector<value_type, EPT> out{};\n" +
        "        MATX_LOOP_UNROLL\n" +
        "        for (index_t lane = 0; lane < EPT; lane++) {\n" +
        "          out.data[lane] = static_cast<value_type>(get_lane_scalar(cur, lane) + correction[static_cast<size_t>(lane)]);\n" +
        "        }\n" +
        "        return out;\n" +
        "      }\n" +
        "    }\n" +
        "  }\n" +
        "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() { return Rank_; }\n" +
        "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const { return out_dims_[dim]; }\n" +
        "};\n"
    );
  }
#endif

  __MATX_INLINE__ UnwrapOp(const OpA &op, int axis, MathType discont, MathType period)
      : op_(op), discont_(discont), period_(period), half_period_(period / static_cast<MathType>(2)) {
    static_assert(cuda::std::is_floating_point_v<MathType>,
                  "unwrap() requires a floating-point input");
    static_assert(!is_complex_v<value_type>,
                  "unwrap() does not support complex input");

    MATX_ASSERT_STR(period_ > static_cast<MathType>(0), matxInvalidParameter,
                    "unwrap period must be positive");

    MATX_LOOP_UNROLL
    for (int i = 0; i < Rank(); i++) {
      sizes_[i] = op_.Size(i);
    }

    if constexpr (Rank() > 0) {
      axis_ = axis;
      if (axis_ < 0) {
        axis_ += Rank();
      }
      MATX_ASSERT_STR(axis_ >= 0 && axis_ < Rank(), matxInvalidDim,
                      "unwrap axis must be in range [-rank, rank-1]");
    }
    else {
      axis_ = 0;
    }

    // Match NumPy semantics: discont values smaller than period/2 are treated
    // as period/2.
    if (discont_ < half_period_) {
      discont_ = half_period_;
    }
  }

  __MATX_INLINE__ std::string str() const { return "unwrap(" + op_.str() + ")"; }

  template <typename CapType, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
    if constexpr (Rank() == 0) {
      return get_value<CapType>(op_);
    }
    else {
      constexpr index_t EPT = static_cast<index_t>(CapType::ept);
      auto get_lane_scalar = [](const auto &v, index_t lane) {
        if constexpr (CapType::ept == ElementsPerThread::ONE) {
          (void)lane;
          return static_cast<MathType>(v);
        }
        else {
          return static_cast<MathType>(v.data[lane]);
        }
      };

      cuda::std::array<index_t, Rank()> idx{indices...};
      const index_t out_idx = idx[axis_];
      const auto cur = get_value<CapType>(op_, idx);
      cuda::std::array<MathType, static_cast<size_t>(EPT)> correction{};

      if (out_idx != 0) {
        cuda::std::array<index_t, Rank()> seq_idx = idx;
        seq_idx[axis_] = 0;
        auto prev = get_value<CapType>(op_, seq_idx);
        const MathType neg_half_period = -half_period_;

        for (index_t i = 1; i <= out_idx; i++) {
          seq_idx[axis_] = i;
          const auto next = get_value<CapType>(op_, seq_idx);

          MATX_LOOP_UNROLL
          for (index_t lane = 0; lane < EPT; lane++) {
            const MathType next_s = get_lane_scalar(next, lane);
            const MathType prev_s = get_lane_scalar(prev, lane);
            const MathType delta = next_s - prev_s;

            MathType delta_mod =
                static_cast<MathType>(scalar_internal_fmod(delta + half_period_, period_));
            if (delta_mod < static_cast<MathType>(0)) {
              delta_mod += period_;
            }
            delta_mod -= half_period_;

            if (delta_mod == neg_half_period && delta > static_cast<MathType>(0)) {
              delta_mod = half_period_;
            }

            MathType phase_correction = delta_mod - delta;
            if (cuda::std::abs(delta) < discont_) {
              phase_correction = static_cast<MathType>(0);
            }

            correction[static_cast<size_t>(lane)] += phase_correction;
          }
          prev = next;
        }
      }

      if constexpr (CapType::ept == ElementsPerThread::ONE) {
        return static_cast<value_type>(get_lane_scalar(cur, 0) + correction[0]);
      }
      else {
        Vector<value_type, EPT> out{};
        MATX_LOOP_UNROLL
        for (index_t lane = 0; lane < EPT; lane++) {
          out.data[lane] = static_cast<value_type>(
              get_lane_scalar(cur, lane) + correction[static_cast<size_t>(lane)]);
        }
        return out;
      }
    }
  }

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
    return this->operator()<DefaultCapabilities>(indices...);
  }

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() {
    return OpA::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const {
    return sizes_[dim];
  }

  __MATX_INLINE__ __MATX_HOST__ int32_t DynRank() const {
    return detail::get_dyn_rank(op_);
  }

  __MATX_INLINE__ __MATX_HOST__ int32_t jit_rank() const {
    if constexpr (is_dynamic_rank_op_v<self_type>) return DynRank();
    else return Rank();
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept {
    if constexpr (is_matx_op<OpA>()) {
      op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept {
    if constexpr (is_matx_op<OpA>()) {
      op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
  }

  template <OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
    if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
      const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
      return get_jit_class_name() + "<" + op_jit_name + "," + detail::type_to_string<MathType>() + ">";
#else
      return "";
#endif
    }
    else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
      return combine_capabilities<Cap>(true, detail::get_operator_capability<Cap>(op_, in));
#else
      return false;
#endif
    }
    else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
      const auto [key, value] = get_jit_op_str();
      if (in.find(key) == in.end()) {
        in[key] = value;
      }
      detail::get_operator_capability<Cap>(op_, in);
      return true;
#else
      return false;
#endif
    }
    else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
      const auto my_cap =
          cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
      return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
    }
    else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
      return detail::get_operator_capability<Cap>(op_, in);
    }
    else {
      auto self_has_cap = capability_attributes<Cap>::default_value;
      return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
    }
  }

private:
  typename detail::base_type_t<OpA> op_;
  cuda::std::array<index_t, Rank()> sizes_;
  int axis_;
  MathType discont_;
  MathType period_;
  MathType half_period_;
};
} // namespace detail

/**
 * @brief Unwrap phase angles by correcting jumps greater than a discontinuity.
 *
 * This implementation follows NumPy's `unwrap` behavior, including support
 * for custom period and discont values.
 *
 * @tparam Op Input operator/tensor type
 * @param op Input operator
 * @param axis Axis to unwrap. Default is last axis (-1)
 * @param discont Maximum discontinuity between adjacent samples. Values lower
 * than `period / 2` are treated as `period / 2`.
 * @param period Complement period used to unwrap phase values. Default is 2*pi.
 */
template <typename Op>
__MATX_INLINE__ auto unwrap(
    const Op &op, int axis = -1,
    detail::value_promote_t<typename Op::value_type> discont =
        static_cast<detail::value_promote_t<typename Op::value_type>>(-1),
    detail::value_promote_t<typename Op::value_type> period =
        static_cast<detail::value_promote_t<typename Op::value_type>>(
            cuda::std::numbers::pi_v<detail::value_promote_t<typename Op::value_type>> * 2)) {
  MATX_NVTX_START("unwrap(" + get_type_str(op) + ")", matx::MATX_NVTX_LOG_API)
  using math_type = detail::value_promote_t<typename Op::value_type>;
  const math_type period_in = static_cast<math_type>(period);
  const math_type default_discont = period_in / static_cast<math_type>(2);
  const math_type discont_in =
      (discont < static_cast<math_type>(0)) ? default_discont
                                            : static_cast<math_type>(discont);
  return detail::UnwrapOp<Op, math_type>(op, axis, discont_in, period_in);
}

} // namespace matx
