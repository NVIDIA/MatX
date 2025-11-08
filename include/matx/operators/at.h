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

#include "matx/operators/base_operator.h"

namespace matx
{

  /**
   * Returns the current tensor index for the given dimension.
   */
  namespace detail {
    template <typename Op, typename... Is>
    class AtOp : public BaseOp<AtOp<Op, Is...>>
    {
      private:
        mutable typename detail::base_type_t<Op> op_;
        cuda::std::array<index_t, sizeof...(Is)> idx_;

      public:
        using matxop = bool;
        using value_type = typename Op::value_type;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<Op>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string idx_str;
          for (int32_t i = 0; i < static_cast<int32_t>(sizeof...(Is)); i++) {
            idx_str += std::to_string(idx_[i]);
            if (i < static_cast<int32_t>(sizeof...(Is)) - 1) idx_str += "_";
          }
          return std::format("JITAt_idx{}", idx_str);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename Op> struct {} {{\n"
                "  using value_type = typename Op::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int NumIdx = {};\n"
                "  constexpr static cuda::std::array<index_t, NumIdx> idx_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<Op>> op_;\n"
                "  template <typename CapType, typename... Is2>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()([[maybe_unused]] Is2... indices) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      return op_.template operator()<CapType>(idx_);\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<size_t>(CapType::ept)>();\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return 0; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const {{ return index_t(0); }}\n"
                "}};\n",
                func_name, sizeof...(Is), detail::array_to_string(idx_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "at()"; }
        __MATX_INLINE__ AtOp(const Op &op, Is... is) : op_(op), idx_{is...} {
          MATX_LOG_TRACE("{} constructor: num_indices={}", str(), sizeof...(Is));
        };

        template <typename CapType, typename... Is2>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()([[maybe_unused]] Is2... indices) const
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            return op_.template operator()<CapType>(idx_);
          }
          else {
            return Vector<value_type, static_cast<size_t>(CapType::ept)>();
          }
        }

        template <typename... Is2>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()([[maybe_unused]] Is2... indices) const
        {
          return this->operator()<DefaultCapabilities>(idx_);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
            return std::format("{}<{}>", get_jit_class_name(), op_jit_name);
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
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(op_, in);
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return 0;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          return index_t(0);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<Op>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<Op>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }


#ifndef DOXYGEN_ONLY
  template <typename Op, typename... Is>
    requires ((std::is_integral_v<Is>) && ...)
#else
  template <typename Op, typename... Is>
#endif
  __MATX_INLINE__ auto at(const Op &op, Is... indices) {
    return detail::AtOp(op, indices...);
  }
} // end namespace matx
