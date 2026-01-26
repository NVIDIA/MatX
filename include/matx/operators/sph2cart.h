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
#include "matx/operators/base_operator.h"

namespace matx
{
  /**
   * Operator that computes the transform from spherical to cartesian indexing
   */
  namespace detail {
    template <typename T1, typename T2, typename T3, int WHICH>
      class Sph2CartOp : public BaseOp<Sph2CartOp<T1, T2, T3, WHICH>>
    {
      private:
        mutable typename detail::base_type_t<T1> theta_;
        mutable typename detail::base_type_t<T2> phi_;
        mutable typename detail::base_type_t<T3> r_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> theta_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> phi_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<T3>> r_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(theta_), detail::to_jit_storage(phi_), detail::to_jit_storage(r_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITSph2Cart_which{}", WHICH);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, Rank()> out_dims_;
          for (int i = 0; i < Rank(); ++i) {
            out_dims_[i] = Size(i);
          }
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T1, typename T2, typename T3> struct {} {{\n"
                "  using value_type = typename T1::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int WHICH_ = {};\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> theta_;\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> phi_;\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T3>> r_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      auto theta = get_value<CapType>(theta_, indices...);\n"
                "      auto phi = get_value<CapType>(phi_, indices...);\n"
                "      auto r = get_value<CapType>(r_, indices...);\n"
                "      if constexpr (WHICH_ == 0) {{\n"
                "        return r * (scalar_internal_cos(phi) * scalar_internal_cos(theta));\n"
                "      }} else if constexpr (WHICH_ == 1) {{\n"
                "        return r * (scalar_internal_cos(phi) * scalar_internal_sin(theta));\n"
                "      }} else {{\n"
                "        return r * scalar_internal_sin(phi);\n"
                "      }}\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ auto Size(int dim) const {{ return out_dims_[dim]; }}\n"
                "}};\n",
                func_name, WHICH, Rank(), detail::array_to_string(out_dims_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "sph2cart(" + get_type_str(theta_) +
          "," + get_type_str(phi_) + "," + get_type_str(r_) + ")"; }

        __MATX_INLINE__ Sph2CartOp(const T1 &theta, const T2 &phi, const T3 &r) : theta_(theta), phi_(phi), r_(r)
      {
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
        MATX_ASSERT_COMPATIBLE_OP_SIZES(theta);
        MATX_ASSERT_COMPATIBLE_OP_SIZES(phi);
        MATX_ASSERT_COMPATIBLE_OP_SIZES(r);
      }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            [[maybe_unused]] auto theta = get_value<CapType>(theta_, indices...);
            [[maybe_unused]] auto phi = get_value<CapType>(phi_, indices...);
            auto r = get_value<CapType>(r_, indices...);

            if constexpr (WHICH==0) { // X
              return r * (scalar_internal_cos(phi) * scalar_internal_cos(theta));
            } else if constexpr (WHICH==1) { // Y
              return r * (scalar_internal_cos(phi) * scalar_internal_sin(theta));
            } else {  // Z
              return r * scalar_internal_sin(phi);
            }
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            theta_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            phi_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T3>()) {
            r_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            theta_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            phi_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T3>()) {
            r_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return matx_max(get_rank<T1>(), get_rank<T2>(), get_rank<T3>());
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
					index_t size1 = get_expanded_size<Rank()>(theta_, dim);
					index_t size2 = get_expanded_size<Rank()>(phi_, dim);
					index_t size3 = get_expanded_size<Rank()>(r_, dim);
					return detail::matx_max(size1, size2, size3);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto theta_jit_name = detail::get_operator_capability<Cap>(theta_, in);
            const auto phi_jit_name = detail::get_operator_capability<Cap>(phi_, in);
            const auto r_jit_name = detail::get_operator_capability<Cap>(r_, in);
            return std::format("{}<{},{},{}>", get_jit_class_name(), theta_jit_name, phi_jit_name, r_jit_name);
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return combine_capabilities<Cap>(true, 
              detail::get_operator_capability<Cap>(theta_, in),
              detail::get_operator_capability<Cap>(phi_, in),
              detail::get_operator_capability<Cap>(r_, in));
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
            detail::get_operator_capability<Cap>(theta_, in);
            detail::get_operator_capability<Cap>(phi_, in);
            detail::get_operator_capability<Cap>(r_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(theta_, in) +
                   detail::get_operator_capability<Cap>(phi_, in) +
                   detail::get_operator_capability<Cap>(r_, in);
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(
              my_cap,
              detail::get_operator_capability<Cap>(theta_, in),
              detail::get_operator_capability<Cap>(phi_, in),
              detail::get_operator_capability<Cap>(r_, in)
            );
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap,
              detail::get_operator_capability<Cap>(theta_, in),
              detail::get_operator_capability<Cap>(phi_, in),
              detail::get_operator_capability<Cap>(r_, in)
            );
          }
        }
    };
  }
  /**
   * Operator to compute cartesian coordiantes from spherical coordinates
   * TODO
   * @tparam T1
   *   Operator type defining theta
   *
   * @tparam T2
   *   Operator type defining phi
   *
   * @tparam T3
   *   Operator type defining radius
   *
   * @param theta
   *   Operator defining theta
   *
   * @param phi
   *   Operator defining phi
   *
   * @param r
   *   Operator defining radius
   *
   * @returns
   *   Tuple of operators for x, y, and z.
   */
  template <typename T1, typename T2, typename T3>
    auto __MATX_INLINE__ sph2cart(const T1 &theta, const T2 &phi, const T3 &r)
    {
      return cuda::std::tuple{
        detail::Sph2CartOp<T1,T2,T3,0>(theta, phi, r),
        detail::Sph2CartOp<T1,T2,T3,1>(theta, phi, r),
        detail::Sph2CartOp<T1,T2,T3,2>(theta, phi, r)};
    };

} // end namespace matx
