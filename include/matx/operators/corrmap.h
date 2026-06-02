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
#include <cuda/std/cmath>
#include <cuda/std/array>

namespace matx
{

  /**
   * @brief Normalization mode for the windowed correlation map.
   *
   * Given two inputs \f$ A, B \f$ of identical shape, for each output element
   * the operator iterates over a small window \f$ W \f$ around that element
   * and combines the samples according to the selected mode.
   *
   * In the per-value equations below, \f$ a_k, b_k \f$ are the input samples
   * at the window offsets \f$ k \in W \f$, the conjugate is
   * \f$ \overline{\cdot} \f$ (the identity for real inputs), and \f$ N = |W|
   * \f$ is the number of in-bounds samples in the window.
   *
   * For the MAGNITUDE and ZNCC modes the listed result ranges are the
   * mathematical bounds. The operator does not clamp, so floating-point
   * arithmetic may produce values slightly outside the bounds in degenerate
   * windows.
   */
  enum class CorrMapNormalize {
    /**
     * @brief Raw windowed inner product. Result is unbounded, other than
     * the data type limits.
     *
     * \f[
     *   y = \sum_{k \in W} a_k \, \overline{b_k}
     * \f]
     */
    NONE,

    /**
     * @brief Energy-normalized cross-correlation: divide by the geometric
     * mean of the two windowed energies.
     *
     * For complex inputs the result is complex with magnitude in
     * \f$ [0, 1] \f$: \f$ |y| \f$ is the SAR/InSAR coherence magnitude and
     * \f$ \angle y \f$ is the interferogram phase. For real inputs the
     * result lies in \f$ [-1, 1] \f$.
     *
     * \f[
     *   y = \frac{\displaystyle \sum_{k \in W} a_k \, \overline{b_k}}
     *            {\sqrt{\displaystyle \sum_{k \in W} |a_k|^2 \;
     *                   \sum_{k \in W} |b_k|^2}}
     * \f]
     */
    MAGNITUDE,

    /**
     * @brief Zero-mean normalized cross-correlation: subtract the
     * window-local means before normalizing.
     *
     * This is the classic NCC used in image processing, pattern matching,
     * and is equivalent to the Pearson correlation coefficient. Let
     * \f$ \mu_a = \tfrac{1}{N}\sum_{k \in W} a_k \f$ and
     * \f$ \mu_b = \tfrac{1}{N}\sum_{k \in W} b_k \f$ be the window-local
     * means. Result is in \f$ [-1, 1] \f$ for real inputs; complex with
     * magnitude \f$ \le 1 \f$ for complex inputs.
     *
     * \f[
     *   y = \frac{\displaystyle \sum_{k \in W} (a_k - \mu_a)\,
     *                                  \overline{(b_k - \mu_b)}}
     *            {\sqrt{\displaystyle \sum_{k \in W} |a_k - \mu_a|^2 \;
     *                   \sum_{k \in W} |b_k - \mu_b|^2}}
     * \f]
     */
    ZNCC
  };

  namespace detail {

    // Scalar sqrt that dispatches half precision to matx::sqrt's half overload
    // (in matx/core/half.h). cuda::std::sqrt is ambiguous for matxFp16 /
    // matxBf16 (and the raw __half / __nv_bfloat16) because their implicit
    // conversions match multiple of its built-in overloads.
    template <typename T>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T corrmap_scalar_sqrt(T x)
    {
      if constexpr (is_matx_half_v<T> || is_half_v<T>) {
        return matx::sqrt(x);
      } else {
        return cuda::std::sqrt(x);
      }
    }

    // Per-pixel windowed correlation map.
    //
    // WindowRank == 1: the window slides along the last input dimension.
    //                  All other input dims are batch.
    // WindowRank == 2: the window slides along the last two input dimensions.
    //                  All other input dims are batch.
    //
    // Mode is a compile-time template parameter so the normalization formula
    // collapses to a single branch and the mean accumulators are elided when
    // they're not needed.
    //
    // The output shape is always identical to the input shape.
    template <typename OpA, typename OpB, int WindowRank, CorrMapNormalize Mode>
    class CorrMapOp : public BaseOp<CorrMapOp<OpA, OpB, WindowRank, Mode>>
    {
      private:
        static_assert(WindowRank == 1 || WindowRank == 2,
                      "corrmap supports 1-D or 2-D windows only");

        mutable typename detail::base_type_t<OpA> a_;
        mutable typename detail::base_type_t<OpB> b_;
        cuda::std::array<index_t, WindowRank> window_;

        static constexpr int32_t out_rank = OpA::Rank();
        cuda::std::array<index_t, out_rank> out_dims_;

      public:
        using matxop = bool;

      private:
        // Decide the output precision and whether the output is complex by
        // separating the two questions:
        //   * inner_type = common_type of the two real precisions (so e.g.
        //     float + double -> double).
        //   * any_complex = true if either input is complex.
        using inner_a_ = typename inner_op_type_t<typename OpA::value_type>::type;
        using inner_b_ = typename inner_op_type_t<typename OpB::value_type>::type;
        static constexpr bool any_complex_ =
          is_complex_v<typename OpA::value_type> ||
          is_complex_v<typename OpB::value_type>;

        // Reject integer inputs. Normalized modes (MAGNITUDE / ZNCC) would
        // silently truncate the final division to 0 for integer inner types,
        // producing meaningless results. Users wanting integer inputs must
        // cast explicitly (e.g. corrmap(as_float(A), as_float(B), w)).
        template <typename T>
        static constexpr bool is_supported_inner_v =
          cuda::std::is_same_v<T, float> ||
          cuda::std::is_same_v<T, double> ||
          is_matx_half_v<T> ||
          is_half_v<T>;
        static_assert(is_supported_inner_v<inner_a_>,
                      "corrmap() requires floating-point (or complex floating-point) input types");
        static_assert(is_supported_inner_v<inner_b_>,
                      "corrmap() requires floating-point (or complex floating-point) input types");

      public:
        // Real scalar type for energy accumulators and normalization.
        using inner_type = cuda::std::common_type_t<inner_a_, inner_b_>;

        // Output type:
        //   complex<float>  + complex<double> -> complex<double>
        //   float           + double          -> double
        //   float           + complex<double> -> complex<double>
        //   matxFp16Complex + matxFp16        -> matxFp16Complex
        //
        // complex_from_scalar_t maps the inner scalar to the right complex
        // wrapper (cuda::std::complex<T> for float/double, matxHalfComplex<T>
        // for half types). This is what keeps matxFp16Complex outputs from
        // becoming the ill-formed cuda::std::complex<matxFp16>.
        using value_type = cuda::std::conditional_t<
          any_complex_,
          complex_from_scalar_t<inner_type>,
          inner_type>;

        using self_type = CorrMapOp<OpA, OpB, WindowRank, Mode>;

        // Propagate dynamic tensor marker through expression tree
        using dynamic_tensor_expr = cuda::std::bool_constant<
          is_dynamic_tensor_v<OpA> || is_dynamic_rank_op_v<OpA> ||
          is_dynamic_tensor_v<OpB> || is_dynamic_rank_op_v<OpB>>;

#ifdef MATX_EN_JIT
        // JIT storage: same operands the host operator() reads. The window
        // sizes are baked into the generated type as a constexpr array (see
        // get_jit_class_name / get_jit_op_str), so they are not stored here.
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<OpB>> b_;
          cuda::std::array<index_t, out_rank> out_dims_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(a_),
                             detail::to_jit_storage(b_),
                             out_dims_};
        }

        // The class name encodes WindowRank, Mode, and the window dimensions so
        // each (rank, mode, window) combination gets its own NVRTC-cached class.
        // Baking the window sizes into the type (rather than passing them as
        // runtime storage) lets the generated window loops use compile-time
        // bounds and unroll, matching how slice() bakes its sizes/dims/starts.
        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string win_str;
          for (int d = 0; d < WindowRank; d++) {
            win_str += std::format("_{}", window_[d]);
          }
          return std::format("JITCorrMap_w{}_m{}{}", WindowRank,
                             static_cast<int>(Mode), win_str);
        }

        // Emit the operator's struct definition for NVRTC compilation. The
        // body mirrors the host operator() exactly -- the `if constexpr`
        // branches on Mode, WindowRank, and is_cplx collapse at JIT
        // compile time the same way they do on the host side.
        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          return cuda::std::make_tuple(
            func_name,
            std::format(
              "template <typename OpA, typename OpB> struct {0} {{\n"
              "  using matxop = bool;\n"
              "  static constexpr int32_t out_rank = OpA::Rank();\n"
              "  static constexpr int WindowRank = {1};\n"
              "  static constexpr int Mode = {2};  // NONE={3}, MAGNITUDE={4}, ZNCC={5}\n"
              "  using inner_a_ = typename inner_op_type_t<typename OpA::value_type>::type;\n"
              "  using inner_b_ = typename inner_op_type_t<typename OpB::value_type>::type;\n"
              "  static constexpr bool any_complex_ =\n"
              "    is_complex_v<typename OpA::value_type> ||\n"
              "    is_complex_v<typename OpB::value_type>;\n"
              "  using inner_type = cuda::std::common_type_t<inner_a_, inner_b_>;\n"
              "  using value_type = cuda::std::conditional_t<\n"
              "    any_complex_,\n"
              "    complex_from_scalar_t<inner_type>,\n"
              "    inner_type>;\n"
              "\n"
              "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n"
              "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpB>> b_;\n"
              "  static constexpr cuda::std::array<index_t, WindowRank> window_ = {{ {6} }};\n"
              "  cuda::std::array<index_t, out_rank> out_dims_;\n"
              "\n"
              "  // Scalar sqrt: dispatches half precision through float to avoid\n"
              "  // cuda::std::sqrt overload ambiguity for the half types.\n"
              "  template <typename T>\n"
              "  static __MATX_INLINE__ __MATX_DEVICE__ T scalar_sqrt(T x) {{\n"
              "    if constexpr (is_matx_half_v<T> || is_half_v<T>) {{\n"
              "      return static_cast<T>(cuda::std::sqrt(static_cast<float>(x)));\n"
              "    }} else {{\n"
              "      return cuda::std::sqrt(x);\n"
              "    }}\n"
              "  }}\n"
              "\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const {{\n"
              "    if constexpr (CapType::ept != ElementsPerThread::ONE) {{\n"
              "      return Vector<value_type, static_cast<size_t>(CapType::ept)>();\n"
              "    }} else {{\n"
              "      constexpr int rank = static_cast<int>(sizeof...(Is));\n"
              "      cuda::std::array<index_t, rank> out_idx{{static_cast<index_t>(indices)...}};\n"
              "\n"
              "      cuda::std::array<index_t, WindowRank> win_dim_size;\n"
              "      cuda::std::array<index_t, WindowRank> win_start;\n"
              "      for (int d = 0; d < WindowRank; d++) {{\n"
              "        win_dim_size[d] = a_.Size(rank - WindowRank + d);\n"
              "        const index_t center = out_idx[rank - WindowRank + d];\n"
              "        win_start[d] = center - (window_[d] / 2);\n"
              "      }}\n"
              "\n"
              "      constexpr bool is_cplx = is_complex_v<value_type>;\n"
              "      constexpr bool need_energies = (Mode == {4});  // MAGNITUDE\n"
              "      constexpr bool use_welford   = (Mode == {5});  // ZNCC\n"
              "\n"
              "      [[maybe_unused]] value_type sum_ab{{}};\n"
              "      [[maybe_unused]] inner_type sum_aa{{0}};\n"
              "      [[maybe_unused]] inner_type sum_bb{{0}};\n"
              "      [[maybe_unused]] value_type mean_a{{}}, mean_b{{}}, C{{}};\n"
              "      [[maybe_unused]] inner_type M2_a{{0}}, M2_b{{0}};\n"
              "      index_t n_valid = 0;\n"
              "      cuda::std::array<index_t, rank> idx = out_idx;\n"
              "\n"
              "      auto accumulate = [&](const cuda::std::array<index_t, rank> &read_idx) {{\n"
              "        const value_type a_val = static_cast<value_type>(detail::get_value<CapType>(a_, read_idx));\n"
              "        const value_type b_val = static_cast<value_type>(detail::get_value<CapType>(b_, read_idx));\n"
              "        n_valid++;\n"
              "        if constexpr (use_welford) {{\n"
              "          const inner_type n = static_cast<inner_type>(n_valid);\n"
              "          if constexpr (is_cplx) {{\n"
              "            const inner_type da_r = a_val.real() - mean_a.real();\n"
              "            const inner_type da_i = a_val.imag() - mean_a.imag();\n"
              "            const inner_type db_r = b_val.real() - mean_b.real();\n"
              "            const inner_type db_i = b_val.imag() - mean_b.imag();\n"
              "            mean_a = value_type{{mean_a.real() + da_r / n, mean_a.imag() + da_i / n}};\n"
              "            mean_b = value_type{{mean_b.real() + db_r / n, mean_b.imag() + db_i / n}};\n"
              "            const inner_type an_r = a_val.real() - mean_a.real();\n"
              "            const inner_type an_i = a_val.imag() - mean_a.imag();\n"
              "            const inner_type bn_r = b_val.real() - mean_b.real();\n"
              "            const inner_type bn_i = b_val.imag() - mean_b.imag();\n"
              "            C = value_type{{C.real() + da_r * bn_r + da_i * bn_i,\n"
              "                            C.imag() + da_i * bn_r - da_r * bn_i}};\n"
              "            M2_a += da_r * an_r + da_i * an_i;\n"
              "            M2_b += db_r * bn_r + db_i * bn_i;\n"
              "          }} else {{\n"
              "            const inner_type da = a_val - mean_a;\n"
              "            const inner_type db = b_val - mean_b;\n"
              "            mean_a += da / n;\n"
              "            mean_b += db / n;\n"
              "            C    += da * (b_val - mean_b);\n"
              "            M2_a += da * (a_val - mean_a);\n"
              "            M2_b += db * (b_val - mean_b);\n"
              "          }}\n"
              "        }} else {{\n"
              "          if constexpr (is_cplx) {{\n"
              "            const inner_type ar = a_val.real();\n"
              "            const inner_type ai = a_val.imag();\n"
              "            const inner_type br = b_val.real();\n"
              "            const inner_type bi = b_val.imag();\n"
              "            sum_ab += value_type{{ar * br + ai * bi, ai * br - ar * bi}};\n"
              "            if constexpr (need_energies) {{\n"
              "              sum_aa += ar * ar + ai * ai;\n"
              "              sum_bb += br * br + bi * bi;\n"
              "            }}\n"
              "          }} else {{\n"
              "            sum_ab += a_val * b_val;\n"
              "            if constexpr (need_energies) {{\n"
              "              sum_aa += a_val * a_val;\n"
              "              sum_bb += b_val * b_val;\n"
              "            }}\n"
              "          }}\n"
              "        }}\n"
              "      }};\n"
              "\n"
              "      if constexpr (WindowRank == 1) {{\n"
              "        for (index_t dr = 0; dr < window_[0]; dr++) {{\n"
              "          const index_t r = win_start[0] + dr;\n"
              "          if (r < 0 || r >= win_dim_size[0]) continue;\n"
              "          idx[rank - 1] = r;\n"
              "          accumulate(idx);\n"
              "        }}\n"
              "      }} else {{\n"
              "        for (index_t dr = 0; dr < window_[0]; dr++) {{\n"
              "          const index_t r = win_start[0] + dr;\n"
              "          if (r < 0 || r >= win_dim_size[0]) continue;\n"
              "          idx[rank - 2] = r;\n"
              "          for (index_t dc = 0; dc < window_[1]; dc++) {{\n"
              "            const index_t c = win_start[1] + dc;\n"
              "            if (c < 0 || c >= win_dim_size[1]) continue;\n"
              "            idx[rank - 1] = c;\n"
              "            accumulate(idx);\n"
              "          }}\n"
              "        }}\n"
              "      }}\n"
              "\n"
              "      if (n_valid == 0) return value_type{{}};\n"
              "\n"
              "      if constexpr (Mode == {3}) {{  // NONE\n"
              "        return sum_ab;\n"
              "      }} else if constexpr (Mode == {4}) {{  // MAGNITUDE\n"
              "        const inner_type denom = scalar_sqrt(sum_aa * sum_bb);\n"
              "        if (denom == inner_type{{0}}) return value_type{{}};\n"
              "        if constexpr (is_cplx) {{\n"
              "          return value_type{{sum_ab.real() / denom, sum_ab.imag() / denom}};\n"
              "        }} else {{\n"
              "          return sum_ab / denom;\n"
              "        }}\n"
              "      }} else {{\n"
              "        const inner_type denom = scalar_sqrt(\n"
              "          cuda::std::max(M2_a, inner_type{{0}}) *\n"
              "          cuda::std::max(M2_b, inner_type{{0}}));\n"
              "        if (denom == inner_type{{0}}) return value_type{{}};\n"
              "        if constexpr (is_cplx) {{\n"
              "          return value_type{{C.real() / denom, C.imag() / denom}};\n"
              "        }} else {{\n"
              "          return C / denom;\n"
              "        }}\n"
              "      }}\n"
              "    }}\n"
              "  }}\n"
              "\n"
              "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return out_rank; }}\n"
              "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
              "}};\n",
              func_name,
              WindowRank,
              static_cast<int>(Mode),
              static_cast<int>(CorrMapNormalize::NONE),
              static_cast<int>(CorrMapNormalize::MAGNITUDE),
              static_cast<int>(CorrMapNormalize::ZNCC),
              detail::array_to_string(window_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "corrmap()"; }

        __MATX_INLINE__ CorrMapOp(const OpA &A, const OpB &B,
                                  const cuda::std::array<index_t, WindowRank> &window)
          : a_(A), b_(B), window_(window)
        {
          MATX_STATIC_ASSERT_STR(OpA::Rank() >= WindowRank,
            matxInvalidDim,
            "corrmap() input rank must be >= window rank.");
          MATX_STATIC_ASSERT_STR(OpA::Rank() == OpB::Rank(),
            matxInvalidDim,
            "corrmap() requires both inputs to have the same rank.");

          for (int d = 0; d < WindowRank; d++) {
            MATX_ASSERT_STR(window[d] >= 1, matxInvalidParameter,
              "corrmap() window dimensions must be >= 1.");
          }

          for (int32_t i = 0; i < OpA::Rank(); i++) {
            MATX_ASSERT_STR(a_.Size(i) == b_.Size(i), matxInvalidSize,
              "corrmap() inputs must have matching shapes in every dimension.");
            out_dims_[i] = a_.Size(i);
          }
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
        {
          // corrmap only supports one element per thread. The ELEMENTS_PER_THREAD
          // capability below advertises this, and combine_capabilities() forces
          // the framework to pick EPT==ONE for any expression containing this
          // operator. The EPT > 1 branch here is an unreachable placeholder
          // that satisfies template instantiation during capability probing.
          if constexpr (CapType::ept != ElementsPerThread::ONE) {
            return Vector<value_type, static_cast<size_t>(CapType::ept)>();
          } else {
            constexpr int rank = static_cast<int>(sizeof...(Is));
            cuda::std::array<index_t, rank> out_idx{static_cast<index_t>(indices)...};

            // Sizes of the input dims that the window slides over (the last
            // WindowRank dims).
            cuda::std::array<index_t, WindowRank> win_dim_size;
            cuda::std::array<index_t, WindowRank> win_start;
            for (int d = 0; d < WindowRank; d++) {
              win_dim_size[d] = a_.Size(rank - WindowRank + d);
              const index_t center = out_idx[rank - WindowRank + d];
              // Floor-center convention: offsets span [-(w/2), w-1-(w/2)].
              win_start[d] = center - (window_[d] / 2);
            }

            constexpr bool is_cplx = is_complex_v<value_type>;
            // NONE: just sum_ab.
            // MAGNITUDE: sum_ab + energies. No mean subtraction, so no
            //   cancellation risk -- a simple accumulator is fine.
            // ZNCC: Welford's online algorithm for the centered cross-
            //   product and centered energies. Avoids the
            //   single-pass centered-identity cancellation that would
            //   otherwise turn near-equal sum_ab and n*ma*mb into noise.
            constexpr bool need_energies = (Mode == CorrMapNormalize::MAGNITUDE);
            constexpr bool use_welford   = (Mode == CorrMapNormalize::ZNCC);

            // Simple-sum accumulators for NONE and MAGNITUDE.
            [[maybe_unused]] value_type sum_ab{};
            [[maybe_unused]] inner_type sum_aa{0};
            [[maybe_unused]] inner_type sum_bb{0};

            // Welford state for ZNCC: running means, centered cross-product
            // C = sum (a-mu_a) conj(b-mu_b), and centered energies
            // M2_a = sum |a-mu_a|^2, M2_b = sum |b-mu_b|^2.
            [[maybe_unused]] value_type mean_a{}, mean_b{}, C{};
            [[maybe_unused]] inner_type M2_a{0}, M2_b{0};

            index_t n_valid = 0;
            cuda::std::array<index_t, rank> idx = out_idx;

            // Inline lambda: read one sample, update accumulators.
            auto accumulate = [&](const cuda::std::array<index_t, rank> &read_idx) {
              const value_type a_val = static_cast<value_type>(get_value<CapType>(a_, read_idx));
              const value_type b_val = static_cast<value_type>(get_value<CapType>(b_, read_idx));
              n_valid++;

              if constexpr (use_welford) {
                // Welford recurrence: deltas computed against the OLD running
                // mean, then mean is updated, then M2 / C use the NEW mean.
                // This keeps every accumulated quantity well-conditioned and
                // dodges the catastrophic cancellation of the closed-form
                // centered identity on highly correlated inputs.
                const inner_type n = static_cast<inner_type>(n_valid);
                if constexpr (is_cplx) {
                  const inner_type da_r = a_val.real() - mean_a.real();
                  const inner_type da_i = a_val.imag() - mean_a.imag();
                  const inner_type db_r = b_val.real() - mean_b.real();
                  const inner_type db_i = b_val.imag() - mean_b.imag();
                  // Update means: mean += delta / n.
                  mean_a = value_type{mean_a.real() + da_r / n,
                                      mean_a.imag() + da_i / n};
                  mean_b = value_type{mean_b.real() + db_r / n,
                                      mean_b.imag() + db_i / n};
                  // Centered values against the NEW means.
                  const inner_type an_r = a_val.real() - mean_a.real();
                  const inner_type an_i = a_val.imag() - mean_a.imag();
                  const inner_type bn_r = b_val.real() - mean_b.real();
                  const inner_type bn_i = b_val.imag() - mean_b.imag();
                  // C += delta_a_old * conj(b - mean_b_new).
                  C = value_type{C.real() + da_r * bn_r + da_i * bn_i,
                                 C.imag() + da_i * bn_r - da_r * bn_i};
                  // M2_a += Re(delta_a_old * conj(a - mean_a_new))
                  //       = da_r * an_r + da_i * an_i  (the imaginary part
                  //       cancels for the variance update).
                  M2_a += da_r * an_r + da_i * an_i;
                  M2_b += db_r * bn_r + db_i * bn_i;
                } else {
                  const inner_type da = a_val - mean_a;
                  const inner_type db = b_val - mean_b;
                  mean_a += da / n;
                  mean_b += db / n;
                  C    += da * (b_val - mean_b);
                  M2_a += da * (a_val - mean_a);
                  M2_b += db * (b_val - mean_b);
                }
              } else {
                // NONE / MAGNITUDE: straight accumulators.
                if constexpr (is_cplx) {
                  const inner_type ar = a_val.real();
                  const inner_type ai = a_val.imag();
                  const inner_type br = b_val.real();
                  const inner_type bi = b_val.imag();
                  // a * conj(b)
                  sum_ab += value_type{ar * br + ai * bi, ai * br - ar * bi};
                  if constexpr (need_energies) {
                    sum_aa += ar * ar + ai * ai;
                    sum_bb += br * br + bi * bi;
                  }
                } else {
                  sum_ab += a_val * b_val;
                  if constexpr (need_energies) {
                    sum_aa += a_val * a_val;
                    sum_bb += b_val * b_val;
                  }
                }
              }
            };

            if constexpr (WindowRank == 1) {
              for (index_t dr = 0; dr < window_[0]; dr++) {
                const index_t r = win_start[0] + dr;
                if (r < 0 || r >= win_dim_size[0]) continue;
                idx[rank - 1] = r;
                accumulate(idx);
              }
            } else { // WindowRank == 2
              for (index_t dr = 0; dr < window_[0]; dr++) {
                const index_t r = win_start[0] + dr;
                if (r < 0 || r >= win_dim_size[0]) continue;
                idx[rank - 2] = r;
                for (index_t dc = 0; dc < window_[1]; dc++) {
                  const index_t c = win_start[1] + dc;
                  if (c < 0 || c >= win_dim_size[1]) continue;
                  idx[rank - 1] = c;
                  accumulate(idx);
                }
              }
            }

            if (n_valid == 0) {
              return value_type{};
            }

            if constexpr (Mode == CorrMapNormalize::NONE) {
              return sum_ab;
            }
            else if constexpr (Mode == CorrMapNormalize::MAGNITUDE) {
              const inner_type denom = corrmap_scalar_sqrt(sum_aa * sum_bb);
              if (denom == inner_type{0}) {
                return value_type{};
              }
              if constexpr (is_cplx) {
                return value_type{sum_ab.real() / denom, sum_ab.imag() / denom};
              } else {
                return sum_ab / denom;
              }
            }
            else { // Mode == CorrMapNormalize::ZNCC
              // Welford state already holds the centered cross-product (C)
              // and the centered energies (M2_a, M2_b). Both M2 terms are
              // non-negative by construction; the cuda::std::max guard is
              // defensive against accumulated rounding pushing them slightly
              // below zero in lower precisions.
              const inner_type denom = corrmap_scalar_sqrt(
                cuda::std::max(M2_a, inner_type{0}) *
                cuda::std::max(M2_b, inner_type{0}));
              if (denom == inner_type{0}) {
                return value_type{};
              }
              if constexpr (is_cplx) {
                return value_type{C.real() / denom, C.imag() / denom};
              } else {
                return C / denom;
              }
            }
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto a_jit_name = detail::get_operator_capability<Cap>(a_, in);
            const auto b_jit_name = detail::get_operator_capability<Cap>(b_, in);
            return std::format("{}<{},{}>", get_jit_class_name(), a_jit_name, b_jit_name);
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return combine_capabilities<Cap>(true,
              detail::get_operator_capability<Cap>(a_, in),
              detail::get_operator_capability<Cap>(b_, in));
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
            detail::get_operator_capability<Cap>(a_, in);
            detail::get_operator_capability<Cap>(b_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{
              ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(
              my_cap,
              detail::get_operator_capability<Cap>(a_, in),
              detail::get_operator_capability<Cap>(b_, in));
          }
          else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            // Each output element reads a neighborhood of input samples
            // (the window). An in-place expression like
            // (A = corrmap(A, B, ...)).run(...) would have a thread writing
            // to A(r, c) race with neighbor threads still reading A(r+/-k,
            // c+/-k), corrupting the output. Mark this as input/output
            // non-pointwise so the alias detector flags such expressions.
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(
              detail::get_operator_capability<Cap>(a_, in_copy),
              detail::get_operator_capability<Cap>(b_, in_copy));
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap,
              detail::get_operator_capability<Cap>(a_, in),
              detail::get_operator_capability<Cap>(b_, in));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return out_rank;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
          if constexpr (is_matx_op<OpB>()) {
            b_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
          if constexpr (is_matx_op<OpB>()) {
            b_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  } // namespace detail

  /**
   * @brief Per-sample 1-D windowed correlation map.
   *
   * Slides a 1-D window of length \f$ w \f$ along the last input dimension;
   * all other dimensions are treated as independent batches. Input and
   * output have the same shape.
   *
   * For an output sample at index \f$ n \f$ along the windowed dimension,
   * let \f$ W \f$ denote the set of in-bounds window offsets
   * \f[
   *   W = \bigl\{\, k \in \mathbb{Z} \;:\;
   *               -\lfloor w/2 \rfloor \le k \le w - 1 - \lfloor w/2 \rfloor,
   *               \ \text{and } n + k \text{ is in-bounds}\bigr\}.
   * \f]
   * This is the floor-center indexing convention. For odd \f$ w \f$ the
   * window is exactly centered on \f$ n \f$; for even \f$ w \f$ it is
   * asymmetric by half a sample. The window is cropped at the input
   * boundary: out-of-bounds offsets are skipped and the normalization
   * uses only the in-bounds samples (means and energies are computed
   * over the cropped window, not over a zero-padded one).
   *
   * For the selected normalization mode the operator computes
   *
   * \f[
   *   y_n = f_{\text{Mode}}\!\left(
   *           \{a_k : k \in W\},\ \{b_k : k \in W\}\right),
   *   \qquad a_k = A(\ldots, n + k),\ b_k = B(\ldots, n + k),
   * \f]
   *
   * where the function \f$ f_{\text{Mode}} \f$ is given in
   * \ref CorrMapNormalize.
   *
   * Output element type:
   *   - complex if either input is complex
   *   - real otherwise
   *   - precision is the greater of the inputs (e.g. `float + double ->
   *     double`, `complex<float> + complex<double> -> complex<double>`,
   *     `complex<float> + double -> complex<double>`)
   *
   * @tparam Mode Normalization mode (compile-time). Defaults to
   *              CorrMapNormalize::MAGNITUDE.
   * @tparam OpA Type of input operator A
   * @tparam OpB Type of input operator B
   * @param A Input operator A
   * @param B Input operator B (same shape as A)
   * @param window Window length \f$ w \ge 1 \f$
   * @return corrmap operator producing a tensor with the same shape as A
   */
  template <CorrMapNormalize Mode = CorrMapNormalize::MAGNITUDE,
            typename OpA, typename OpB>
  __MATX_INLINE__ auto corrmap(const OpA &A, const OpB &B,
                               index_t window)
  {
    return detail::CorrMapOp<OpA, OpB, 1, Mode>(
        A, B, cuda::std::array<index_t, 1>{window});
  }

  /**
   * @brief Per-pixel 2-D windowed correlation map.
   *
   * Slides a 2-D window of shape \f$ w_r \times w_c \f$ over the last two
   * input dimensions; all other dimensions are treated as independent
   * batches. Input and output have the same shape.
   *
   * For an output pixel at row \f$ r \f$, column \f$ c \f$, let \f$ W \f$
   * denote the set of in-bounds window offsets
   * \f[
   *   W = \bigl\{\, (i, j) \in \mathbb{Z}^2 \;:\;
   *               -\lfloor w_r/2 \rfloor \le i \le w_r - 1 - \lfloor w_r/2 \rfloor,\
   *               -\lfloor w_c/2 \rfloor \le j \le w_c - 1 - \lfloor w_c/2 \rfloor,\
   *               \ \text{and } (r + i, c + j) \text{ is in-bounds}\bigr\}.
   * \f]
   * This is the floor-center indexing convention. For odd window sizes the
   * window is exactly centered on \f$ (r, c) \f$; for even sizes it is
   * asymmetric by half a pixel, which introduces a half-pixel registration
   * offset relative to the output grid. The window is cropped at the input
   * boundary: out-of-bounds offsets are skipped and the normalization uses
   * only the in-bounds samples (means and energies are computed over the
   * cropped window, not over a zero-padded one).
   *
   * For the selected normalization mode the operator computes
   *
   * \f[
   *   y_{r,c} = f_{\text{Mode}}\!\left(
   *               \{a_{i,j} : (i,j) \in W\},\
   *               \{b_{i,j} : (i,j) \in W\}\right),
   *   \quad a_{i,j} = A(\ldots, r + i,\ c + j),\
   *         b_{i,j} = B(\ldots, r + i,\ c + j),
   * \f]
   *
   * where the function \f$ f_{\text{Mode}} \f$ is given in
   * \ref CorrMapNormalize. With complex inputs and the MAGNITUDE mode the
   * result is the complex interferometric coherence: take `abs(...)` for
   * the coherence magnitude or `angle(...)` for the interferogram phase.
   *
   * Output element type:
   *   - complex if either input is complex
   *   - real otherwise
   *   - precision is the greater of the inputs (e.g. `float + double ->
   *     double`, `complex<float> + complex<double> -> complex<double>`,
   *     `complex<float> + double -> complex<double>`)
   *
   * @tparam Mode Normalization mode (compile-time). Defaults to
   *              CorrMapNormalize::MAGNITUDE (interferometric coherence for
   *              complex inputs).
   * @tparam OpA Type of input operator A
   * @tparam OpB Type of input operator B
   * @param A Input operator A
   * @param B Input operator B (same shape as A)
   * @param window Window dimensions `{rows, cols}`, each \f$ \ge 1 \f$
   * @return corrmap operator producing a tensor with the same shape as A
   */
  template <CorrMapNormalize Mode = CorrMapNormalize::MAGNITUDE,
            typename OpA, typename OpB>
  __MATX_INLINE__ auto corrmap(const OpA &A, const OpB &B,
                               const cuda::std::array<index_t, 2> &window)
  {
    return detail::CorrMapOp<OpA, OpB, 2, Mode>(A, B, window);
  }

} // end namespace matx
