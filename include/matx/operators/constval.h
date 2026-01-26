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

#include "matx/core/utils.h"

namespace matx
{
  namespace detail {
    template <typename T, typename ShapeType> class ConstVal : public BaseOp<ConstVal<T,ShapeType>> {
      static constexpr int RANK = cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value;

      private:
      ShapeType s_;
      T v_;

      public:
      // dummy type to signal this is a matxop
      using matxop = bool;
      using value_type = T;

#ifdef MATX_EN_JIT
      struct JIT_Storage {
        // No runtime members - v_ becomes constexpr
      };

      JIT_Storage ToJITStorage() const {
        return JIT_Storage{};
      }

      __MATX_INLINE__ std::string get_jit_class_name() const {
        // Convert the numeric value to a valid C++ symbol name
        std::string val_str = detail::number_to_symbol(v_);
        return std::format("JITConstVal_val{}_rank{}", val_str, Rank() == matxNoRank ? "No" : std::to_string(Rank()));
      }

      __MATX_INLINE__ auto get_jit_op_str() const {
        std::string func_name = get_jit_class_name();
        std::string dims_array_str;
        std::string size_func_str;
        
        if constexpr (!is_noshape_v<ShapeType>) {
          cuda::std::array<index_t, static_cast<size_t>(RANK)> out_dims_;
          for (int i = 0; i < RANK; ++i) {
            out_dims_[i] = Size(i);
          }
          dims_array_str = std::format("constexpr static cuda::std::array<index_t, {}> out_dims_ = {{ {} }};\n  ", 
                                        RANK, detail::array_to_string(out_dims_));
          size_func_str = std::format("constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n  ");
        } else {
          dims_array_str = "";
          size_func_str = std::format("constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return 0; }}\n  ");
        }
        
        // Format the value for code generation
        std::string val_init_str;
        if constexpr (is_complex_v<T>) {
          // For complex numbers, use constructor syntax: T{real, imag}
          val_init_str = std::format("T{{{}, {}}}", v_.real(), v_.imag());
        } else {
          val_init_str = std::format("{}", v_);
        }
        
        return cuda::std::make_tuple(
          func_name,
          std::format("template <typename T> struct {} {{\n"
              "  using value_type = T;\n"
              "  using matxop = bool;\n"
              "  {}"
              "  constexpr static T v_ = {};\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is...) const\n"
              "  {{\n"
              "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
              "      return v_;\n"
              "    }} else {{\n"
              "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{v_}};\n"
              "    }}\n"
              "  }}\n"
              "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return {}; }}\n"
              "  {}"
              "}};\n",
              func_name, dims_array_str, val_init_str, Rank(), size_func_str)
        );
      }
#endif

      __MATX_INLINE__ std::string str() const { return  "constval"; }
      ConstVal(ShapeType &&s, T val) : s_(std::forward<ShapeType>(s)), v_(val){};

      template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is...) const { 
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            return v_;
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{v_};
          }
        }

      template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is...) const { 
          return this->operator()<DefaultCapabilities>();
        }

      constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const {
        if constexpr (!is_noshape_v<ShapeType>) {
          return *(s_.begin() + dim);
        }
        else {
          return index_t(0);
        }
      }
      static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { 
        if constexpr (!is_noshape_v<ShapeType>) {
          return RANK;
        }
        else {
          return matxNoRank;
        }
      }

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
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
    };
  }
} // end namespace matx
