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

#include "matx/generators/generator1d.h"
#include "matx/core/log.h"
#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <type_traits>
#include <cuda/std/array>
#include <cuda/std/numbers>

namespace matx
{
  /**
   * @brief Parameters for a Taylor window
   *
   * Aggregate with designated-initializer support, e.g.
   * `taylor<0>({N}, {.nbar = 6, .sll = 40})`. Members that are not named keep
   * their defaults. Designators must appear in declaration order.
   */
  struct TaylorWindowParams {
    /**
     * Number of nearly constant-level sidelobes adjacent to the mainlobe. Must be in [1, MaxNbar]. A taper
     * that falls off monotonically needs nbar >= 2*A^2 + 0.5, where A = acosh(10^(sll/20)) / pi, and an
     * nbar far above that bound makes the window peak at its edges.
     */
    index_t nbar = 4;
    /**
     * Desired suppression of the sidelobe level in dB relative to the mainlobe. Must be positive and below
     * about 6165 dB, so that 10^(sll/20) is finite. Below about 15 dB, the window has negative samples or peaks at its edges.
     */
    double sll = 30.0;
    /**
     * When true, scale the window so that its center value is 1; other values are at most 1 only when the
     * taper falls off monotonically (see nbar and sll). When false, the window is not rescaled, and a
     * symmetric window's mean (DC gain) is 1.
     */
    bool norm = true;
    /** When true, generate a symmetric window for filter design. When false, generate a periodic window for spectral analysis. */
    bool sym = true;
  };

  namespace detail {
    template <typename T, int MaxNbar> class Taylor : public BaseOp<Taylor<T, MaxNbar>> {
      static_assert(cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, double>,
                    "taylor() supports float and double window types");
      static_assert(MaxNbar >= 1, "taylor() MaxNbar must be at least 1");

      private:
        // Ordered so the scalars pack tightly ahead of the coefficients
        index_t size_;
        [[maybe_unused]] double sll_;   // Only needed to generate JIT code
        T theta_scale_;                 // pi / len, so each sample needs a multiply instead of a division
        int32_t nbar_;                  // Checked against MaxNbar before construction
        [[maybe_unused]] bool norm_;    // Only needed to generate JIT code
        bool sym_;                      // When false, sample the first size_ points of a (size_ + 1)-point symmetric window
        // The window is coeffs_[0] + sum_{m=1}^{nbar-1} coeffs_[m] * cos(m * theta), where coeffs_[0] is the
        // normalization scale and coeffs_[m] = 2 * scale * F_m
        cuda::std::array<T, MaxNbar> coeffs_;

        // pi / len in double precision, where len is the length of the symmetric window the samples come from
        static double ThetaScale(index_t size, bool sym) {
          const index_t len = sym ? size : size + 1;
          return len > 0 ? cuda::std::numbers::pi / static_cast<double>(len) : 0.0;
        }

        // Cosine sequence coefficients in double precision
        static cuda::std::array<double, MaxNbar> ComputeCoeffs(int32_t nbar, double sll, bool norm) {
          const double A = std::acosh(std::pow(10.0, sll / 20.0)) / cuda::std::numbers::pi;
          const double A2 = A * A;
          const double nbar_half = static_cast<double>(nbar) - 0.5;
          const double s2 = static_cast<double>(nbar * nbar) / (A2 + nbar_half * nbar_half);

          cuda::std::array<double, MaxNbar> c{};
          double fm_sum = 0.0;
          for (int32_t m = 1; m < nbar; m++) {
            const double m2 = static_cast<double>(m * m);
            double numer = (m % 2 == 1) ? 1.0 : -1.0;
            double denom = 2.0;
            for (int32_t n = 1; n < nbar; n++) {
              const double n_half = static_cast<double>(n) - 0.5;
              numer *= 1.0 - m2 / s2 / (A2 + n_half * n_half);
              if (n != m) {
                denom *= 1.0 - m2 / static_cast<double>(n * n);
              }
            }
            c[m] = numer / denom;
            fm_sum += c[m];
          }

          // Every cosine term is 1 at the window center, so the center value is 1 + 2 * sum(F_m) for any length
          const double scale = norm ? 1.0 / (1.0 + 2.0 * fm_sum) : 1.0;
          c[0] = scale;
          for (int32_t m = 1; m < nbar; m++) {
            c[m] *= 2.0 * scale;
          }
          return c;
        }

      public:
        using value_type = T;
        using matxop = bool;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          // No runtime members - size and coefficients are made constexpr
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITTaylor_size{}_sym{}_nbar{}_sll{}_norm{}", size_, sym_ ? 1 : 0, nbar_,
                             number_to_symbol(sll_), norm_ ? 1 : 0);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();

          const auto c = ComputeCoeffs(nbar_, sll_, norm_);
          std::string coeffs_init = "{ ";
          for (int32_t m = 0; m < nbar_; m++) {
            coeffs_init += std::format("static_cast<T>({})", c[m]);
            if (m < nbar_ - 1) {
              coeffs_init += ", ";
            }
          }
          coeffs_init += " }";

          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T> struct {} {{\n"
                "  using value_type = T;\n"
                "  using matxop = bool;\n"
                "  constexpr static index_t size_ = {};\n"
                "  constexpr static bool sym_ = {};\n"
                "  constexpr static index_t len_ = sym_ ? size_ : size_ + 1;\n"
                "  constexpr static T theta_scale_ = static_cast<T>({});\n"
                "  constexpr static int32_t nbar_ = {};\n"
                "  constexpr static cuda::std::array<T, nbar_> coeffs_ = {};\n"
                "  template <typename CapType>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(index_t i) const\n"
                "  {{\n"
                "    return detail::ApplyGeneratorVecFunc<CapType, T>([](index_t idx) {{\n"
                "      if (size_ == 1) {{ return T(1); }}\n"
                "      const T ct = cuda::std::cos(theta_scale_ * static_cast<T>(2 * idx - len_ + 1));\n"
                "      T b1 = T(0);\n"
                "      T b2 = T(0);\n"
                "      for (int32_t m = nbar_ - 1; m >= 1; m--) {{\n"
                "        const T b0 = coeffs_[m] + T(2) * ct * b1 - b2;\n"
                "        b2 = b1;\n"
                "        b1 = b0;\n"
                "      }}\n"
                "      return coeffs_[0] + b1 * ct - b2;\n"
                "    }}, i);\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return 1; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const {{ return size_; }}\n"
                "}};\n",
                func_name, size_, sym_, ThetaScale(size_, sym_), nbar_, coeffs_init)
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "taylor"; }

        inline Taylor(index_t size, const TaylorWindowParams &params) :
          size_(size),
          sll_(params.sll),
          theta_scale_(static_cast<T>(ThetaScale(size, params.sym))),
          nbar_(static_cast<int32_t>(params.nbar)),
          norm_(params.norm),
          sym_(params.sym)
        {
          MATX_LOG_TRACE("Taylor constructor: size={}, nbar={}, sll={}, norm={}, sym={}",
                         size, params.nbar, params.sll, params.norm, params.sym);
          const auto c = ComputeCoeffs(nbar_, sll_, norm_);
          for (int m = 0; m < MaxNbar; m++) {
            coeffs_[m] = static_cast<T>(c[m]);
          }
        }

        template <typename CapType>
        inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
        {
          return detail::ApplyGeneratorVecFunc<CapType, T>([this](index_t idx) {
            if (size_ == 1) { return T(1); }
            const index_t len = sym_ ? size_ : size_ + 1;
            // theta = 2 * pi * (idx - len/2 + 1/2) / len, formed from an exact integer numerator
            const T ct = cuda::std::cos(theta_scale_ * static_cast<T>(2 * idx - len + 1));
            // Clenshaw recurrence evaluates the cosine series with a single cosine
            T b1 = T(0);
            T b2 = T(0);
            for (int32_t m = nbar_ - 1; m >= 1; m--) {
              const T b0 = coeffs_[m] + T(2) * ct * b1 - b2;
              b2 = b1;
              b1 = b0;
            }
            return coeffs_[0] + b1 * ct - b2;
          }, i);
        }

        inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
        {
          return this->operator()<DefaultCapabilities>(i);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            return get_jit_class_name() + "<" + type_to_string<T>() + ">";
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return true;
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
            return true;
#else
            return false;
#endif
          }
          else {
            return capability_attributes<Cap>::default_value;
          }
        }

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size([[maybe_unused]] int dim) const
        {
          return size_;
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }
    };
  }

  /**
   * Creates a Taylor window operator of shape s with the window applied along
   * the specified dimension
   *
   * The Taylor window approximates the Dolph-Chebyshev window's constant
   * sidelobe level for the nbar sidelobes nearest the mainlobe, then tapers
   * the remaining sidelobes. It is widely used for SAR image formation and
   * array tapering.
   *
   * Each sample is evaluated from a cosine series with nbar terms. The
   * series coefficients are stored in a fixed array of MaxNbar entries, so
   * params.nbar must not exceed MaxNbar. A taper without edge peaking needs
   * nbar >= 2*A^2 + 0.5, where A = acosh(10^(sll/20)) / pi. The default
   * MaxNbar of 24 satisfies this for sidelobe levels up to about 87 dB.
   * Other parameters are not restricted: sll below about 15 dB,
   * or nbar far above that bound, gives a window that peaks at its edges or
   * has negative samples.
   *
   * Like other separable windows, a 2D window is the product of two 1D
   * windows, e.g. `taylor<0>({R, C}) * taylor<1>({R, C})`.
   *
   * @tparam Dim
   *   Dimension to create window over
   * @tparam T
   *   Data type of the window (float or double)
   * @tparam MaxNbar
   *   Largest nbar this window can hold
   * @tparam ShapeType
   *   Type of the shape, deduced from s
   *
   * @param s
   *   The shape of the tensor
   * @param params
   *   Window parameters
   *
   * @throws matxInvalidParameter if params.nbar is outside [1, MaxNbar], or params.sll is not positive
   *   or is so large that 10^(sll/20) overflows (above about 6165 dB)
   *
   * Returns values for a Taylor window across the selected dimension.
   */
  template <int Dim, typename T = float, int MaxNbar = 24, typename ShapeType>
    requires (!cuda::std::is_array_v<remove_cvref_t<ShapeType>>)
  inline auto taylor(ShapeType &&s, TaylorWindowParams params = {})
  {
    constexpr int RANK = cuda::std::tuple_size<cuda::std::decay_t<ShapeType>>::value;
    static_assert(RANK > Dim);
    // Checked unconditionally because nbar indexes the fixed-size coefficient array
    if (params.nbar < 1 || params.nbar > MaxNbar) {
      MATX_THROW(matxInvalidParameter,
          std::format("taylor: nbar ({}) must be in [1, {}]; raise the MaxNbar template argument for larger values",
                      params.nbar, MaxNbar));
    }
    // A larger sll overflows 10^(sll/20) and makes every coefficient NaN
    if (!(params.sll > 0.0) || !std::isfinite(std::pow(10.0, params.sll / 20.0))) {
      MATX_THROW(matxInvalidParameter, "taylor: sll must be positive and below about 6165 dB, so that 10^(sll/20) is finite");
    }
    detail::Taylor<T, MaxNbar> h(*(s.begin() + Dim), params);
    return detail::matxGenerator1D_t<detail::Taylor<T, MaxNbar>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
  }

  /**
   * Creates a Taylor window operator of C-array shape s with the window
   * applied along the specified dimension
   *
   * @tparam Dim
   *   Dimension to create window over
   * @tparam T
   *   Data type of the window (float or double)
   * @tparam MaxNbar
   *   Largest nbar this window can hold
   * @tparam RANK
   *   The RANK of the shape, can be deduced from shape
   *
   * @param s
   *   C array representing shape of the tensor
   * @param params
   *   Window parameters
   *
   * @throws matxInvalidParameter if params.nbar is outside [1, MaxNbar], or params.sll is not positive
   *   or is so large that 10^(sll/20) overflows (above about 6165 dB)
   *
   * Returns values for a Taylor window across the selected dimension.
   */
  template <int Dim, typename T = float, int MaxNbar = 24, int RANK>
  inline auto taylor(const index_t (&s)[RANK], TaylorWindowParams params = {})
  {
    return taylor<Dim, T, MaxNbar>(detail::to_array(s), params);
  }
} // end namespace matx
