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

namespace matx {

namespace detail {
  template<typename OpA, int WHICH>
  class FrexpOp : public BaseOp<FrexpOp<OpA, WHICH>>
  {
    private:
      mutable typename detail::base_type_t<OpA> a_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;

#ifdef MATX_EN_JIT
      struct JIT_Storage {
        typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;
      };

      JIT_Storage ToJITStorage() const {
        return JIT_Storage{detail::to_jit_storage(a_)};
      }

      __MATX_INLINE__ std::string get_jit_class_name() const {
        return std::format("JITFrexp_which{}", WHICH);
      }

      __MATX_INLINE__ auto get_jit_op_str() const {
        std::string func_name = get_jit_class_name();
        cuda::std::array<index_t, Rank()> out_dims_;
        for (int i = 0; i < Rank(); ++i) {
          out_dims_[i] = Size(i);
        }
        
        return cuda::std::make_tuple(
          func_name,
          std::format("template <typename OpA> struct {} {{\n"
              "  using value_type = typename OpA::value_type;\n"
              "  using matxop = bool;\n"
              "  constexpr static int WHICH_ = {};\n"
              "  constexpr static int Rank_ = {};\n"
              "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
              "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const {{\n"
              "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
              "      const auto val = get_value<CapType>(a_, indices...);\n"
              "      int rexp;\n"
              "      if constexpr (is_cuda_complex_v<value_type>) {{\n"
              "        if constexpr (cuda::std::is_same_v<float, typename value_type::value_type>) {{\n"
              "          if constexpr (WHICH_ == 0) {{\n"
              "            return cuda::std::frexpf(val.real(), &rexp);\n"
              "          }} else if constexpr (WHICH_ == 1) {{\n"
              "            cuda::std::frexpf(val.real(), &rexp);\n"
              "            return rexp;\n"
              "          }} else if constexpr (WHICH_ == 2) {{\n"
              "            return cuda::std::frexpf(val.imag(), &rexp);\n"
              "          }} else {{\n"
              "            cuda::std::frexpf(val.imag(), &rexp);\n"
              "            return rexp;\n"
              "          }}\n"
              "        }} else {{\n"
              "          if constexpr (WHICH_ == 0) {{\n"
              "            return cuda::std::frexp(val.real(), &rexp);\n"
              "          }} else if constexpr (WHICH_ == 1) {{\n"
              "            cuda::std::frexp(val.real(), &rexp);\n"
              "            return rexp;\n"
              "          }} else if constexpr (WHICH_ == 2) {{\n"
              "            return cuda::std::frexp(val.imag(), &rexp);\n"
              "          }} else {{\n"
              "            cuda::std::frexp(val.imag(), &rexp);\n"
              "            return rexp;\n"
              "          }}\n"
              "        }}\n"
              "      }} else {{\n"
              "        if constexpr (cuda::std::is_same_v<float, value_type>) {{\n"
              "          const float frac = cuda::std::frexpf(val, &rexp);\n"
              "          if constexpr (WHICH_ == 0) {{\n"
              "            return frac;\n"
              "          }} else {{\n"
              "            return rexp;\n"
              "          }}\n"
              "        }} else {{\n"
              "          const double frac = cuda::std::frexp(val, &rexp);\n"
              "          if constexpr (WHICH_ == 0) {{\n"
              "            return frac;\n"
              "          }} else {{\n"
              "            return rexp;\n"
              "          }}\n"
              "        }}\n"
              "      }}\n"
              "    }} else {{\n"
              "      return Vector<value_type, static_cast<size_t>(CapType::ept)>{{}};\n"
              "    }}\n"
              "  }}\n"
              "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
              "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
              "}};\n",
              func_name, WHICH, Rank(), detail::array_to_string(out_dims_))
        );
      }
#endif

      __MATX_INLINE__ std::string str() const { return "frexp()"; }
      __MATX_INLINE__ FrexpOp(const OpA &a) : a_(a) {
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
        static_assert(std::is_floating_point_v<value_type> ||
                      is_cuda_complex_v<value_type>, "frexp() must take a floating point input");

      };

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
      {
        auto get_scalar = [](const auto &x){
          [[maybe_unused]] int rexp;        
          if constexpr (is_cuda_complex_v<value_type>) {
            if constexpr (std::is_same_v<float, typename value_type::value_type>) {
              if constexpr (WHICH == 0) { // real fractional
                const auto frac = cuda::std::frexpf(x.real(), &rexp);
                return frac;
              } else if constexpr (WHICH == 1) { // real exponent
                [[maybe_unused]] const auto frac = cuda::std::frexpf(x.real(), &rexp);
                return rexp;
              } else if constexpr (WHICH == 2) { // imag fractional
                const auto frac = cuda::std::frexpf(x.imag(), &rexp);
                return frac;
              } else if constexpr (WHICH == 3) { // imag exponent
                [[maybe_unused]] const auto frac = cuda::std::frexpf(x.imag(), &rexp);
                return rexp;
              }
            }
            else {
              if constexpr (WHICH == 0) { // real fractional
                const auto frac = cuda::std::frexp(x.real(), &rexp);
                return frac;
              } else if constexpr (WHICH == 1) { // real exponent
                [[maybe_unused]] const auto frac = cuda::std::frexp(x.real(), &rexp);
                return rexp;
              } else if constexpr (WHICH == 2) { // imag fractional
                const auto frac = cuda::std::frexp(x.imag(), &rexp);
                return frac;
              } else if constexpr (WHICH == 3) { // imag exponent
                [[maybe_unused]] const auto frac = cuda::std::frexp(x.imag(), &rexp);
                return rexp;
              }
            }
          }
          else {
            if constexpr (std::is_same_v<float, value_type>) {
              [[maybe_unused]] const float frac = cuda::std::frexpf(x, &rexp);
              if constexpr (WHICH == 0) { // fractional
                return frac;
              } else if constexpr (WHICH == 1) { // exponent
                return rexp;
              }
            }
            else {
              [[maybe_unused]] const double frac = cuda::std::frexp(x, &rexp);
              if constexpr (WHICH == 0) { // fractional
                return frac;
              } else if constexpr (WHICH == 1) { // exponent
                return rexp;
              }
            }
          }
        };

        const auto val = get_value<CapType>(a_, indices...);
        if constexpr (CapType::ept == ElementsPerThread::ONE) {
          return get_scalar(val);
        } else {
          Vector<remove_cvref_t<decltype(get_scalar(val.data[0]))>, static_cast<size_t>(CapType::ept)> out;
          MATX_LOOP_UNROLL
          for (index_t i = 0; i < static_cast<index_t>(CapType::ept); i++) {
            out.data[i] = get_scalar(val.data[i]);
          }
          return out;
        }
      }
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
      {
        return this->operator()<DefaultCapabilities>(indices...);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
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

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
        if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
          const auto op_jit_name = detail::get_operator_capability<Cap>(a_, in);
          return std::format("{}<{}>", get_jit_class_name(), op_jit_name);
#else
          return "";
#endif
        }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return combine_capabilities<Cap>(true, detail::get_operator_capability<Cap>(a_, in));
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
          return true;
#else
          return false;
#endif
        }
        else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
          return detail::get_operator_capability<Cap>(a_, in);
        }
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
        }
      }

  };
}

template<typename OpA>
__MATX_INLINE__ auto frexp(const OpA &a) {
  return cuda::std::tuple{
          detail::FrexpOp<OpA, 0>(a),
          detail::FrexpOp<OpA, 1>(a)
  };
}

template<typename OpA>
__MATX_INLINE__ auto frexpc(const OpA &a) {
  return cuda::std::tuple{
          detail::FrexpOp<OpA, 0>(a),
          detail::FrexpOp<OpA, 1>(a),
          detail::FrexpOp<OpA, 2>(a),
          detail::FrexpOp<OpA, 3>(a)
  };
}

};

