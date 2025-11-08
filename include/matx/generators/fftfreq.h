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

#include "matx/generators/generator1d.h"
#include "matx/core/log.h"

namespace matx
{
  namespace detail {
    template <class T> class FFTFreqOp : public BaseOp<FFTFreqOp<T>> {
      private:
        index_t n_;
        float d_;

      public:
        using value_type = T;
        using matxop = bool;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          // No runtime members - all become constexpr
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITFFTFreq_n{}_d{}", n_, d_);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T> struct {} {{\n"
                "  using value_type = T;\n"
                "  using matxop = bool;\n"
                "  constexpr static index_t n_ = {};\n"
                "  constexpr static float d_ = {};\n"
                "  template <typename CapType>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(index_t idx) const\n"
                "  {{\n"
                "    return detail::ApplyGeneratorVecFunc<CapType, T>([](index_t i) {{\n"
                "      index_t offset = i >= (n_+1)/2 ? -n_ : 0;\n"
                "      return static_cast<T>(i + offset) / (d_*(T)n_);\n"
                "    }}, idx);\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return 1; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const {{ return n_; }}\n"
                "}};\n",
                func_name, n_, d_)
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "fftfreq"; }

        inline FFTFreqOp(index_t n, float d = 1.0)
        {
          n_ = n;
          d_ = d;
          MATX_LOG_TRACE("FFTFreqOp constructor: n={}, d={}", n, d);
        }

        template <typename CapType>
        __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ auto operator()(index_t idx) const {
          return detail::ApplyGeneratorVecFunc<CapType, T>([this](index_t i) {
            index_t offset = i >= (n_+1)/2 ? -n_ : 0;
            return static_cast<T>(i + offset) / (d_*(T)n_);
          }, idx);
        }

        __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ auto operator()(index_t idx) const {
          return this->operator()<DefaultCapabilities>(idx);
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
          return n_;
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }
    };
  }


  /**
   * @brief Return FFT sample frequencies
   *
   * Returns the bin centers in cycles/unit of the sampling frequency known by the user.
   *
   * @tparam T Type of output
   * @param n Number of elements
   * @param d Sample spacing (defaults to 1.0)
   * @return Operator with sampling frequencies
   */
  template <typename T = float>
    inline auto fftfreq(index_t n, float d = 1.0)
    {
      detail::FFTFreqOp<T> l(n, d);
      cuda::std::array<index_t, 1> s{n};
      return detail::matxGenerator1D_t<detail::FFTFreqOp<T>, 0, decltype(s)>(std::move(s), l);
    }
} // end namespace matx
