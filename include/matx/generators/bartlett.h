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
#include <type_traits>

namespace matx
{
  namespace detail {
    template <typename T> class Bartlett : public BaseOp<Bartlett<T>> {
      private:
        index_t size_;

      public:
        using value_type = T;
        using matxop = bool;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          // No runtime members - size_ is made constexpr
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return "JITBartlett_size" + std::to_string(size_);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          
          return cuda::std::make_tuple(
            func_name,
            std::string("template <typename T> struct " + func_name + " {\n") +
                "  using value_type = T;\n" +
                "  using matxop = bool;\n" +
                "  constexpr static index_t size_ = " + std::to_string(size_) + ";\n" +
                "  template <typename CapType>\n" +
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(index_t i) const\n" +
                "  {\n" +
                "    return detail::ApplyGeneratorVecFunc<CapType, T>([](index_t idx) { return 1 - cuda::std::abs(((2*T(idx))/(T(size_ - 1))) - 1); }, i);\n" +
                "  }\n" +
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() { return 1; }\n" +
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const { return size_; }\n" +
                "};\n"
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "bartlett"; }

        inline __MATX_HOST__ __MATX_DEVICE__ Bartlett(index_t size) : size_(size){
#ifndef __CUDA_ARCH__
          MATX_LOG_TRACE("Bartlett constructor: size={}", size);
#endif
        };

        template <typename CapType>
        inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
        {
          return detail::ApplyGeneratorVecFunc<CapType, T>([this](index_t idx) { return 1 - cuda::std::abs(((2*T(idx))/(T(size_ - 1))) - 1); }, i);
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
   * Creates a Bartlett window operator of shape s with the
   * window applies along the x, y, z, or w dimension
   *
   * @tparam T
   *   Data type
   * @tparam Dim
   *   Dimension to create window over
   * @tparam RANK
   *   The RANK of the shape
   *
   * @param s
   *   The shape of the tensor
   *
   * Returns values for a Bartlett window across the selected dimension.
   */
  template <int Dim, typename ShapeType, typename T = float>
    requires (!cuda::std::is_array_v<remove_cvref_t<ShapeType>>)
  inline auto bartlett(ShapeType &&s)
             {
               constexpr int RANK = cuda::std::tuple_size<std::decay_t<ShapeType>>::value;
               static_assert(RANK > Dim);
               detail::Bartlett<T> h( *(s.begin() + Dim));
               return detail::matxGenerator1D_t<detail::Bartlett<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
             }

  /**
   * Creates a Bartlett window operator of shape s with the
   * window applies along the x, y, z, or w dimension
   *
   * @tparam T
   *   Data type
   * @tparam Dim
   *   Dimension to create window over
   * @tparam RANK
   *   The RANK of the shape
   *
   * @param s
   *   The C array shape of the tensor
   *
   * Returns values for a Bartlett window across the selected dimension.
   */
  template <int Dim, int RANK, typename T = float>
    inline auto bartlett(const index_t (&s)[RANK])
    {
      for (int i = 0; i < RANK; i++) {
        MATX_ASSERT_STR(s[i] > 0, matxInvalidSize, "All dimensions must be greater than 0");
      }
      return bartlett<Dim>(detail::to_array(s));
    }

} // end namespace matx
