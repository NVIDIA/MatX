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
   * Operator that computes the transform from cartesian to spherical indexing
   */
  namespace detail {
    template <typename T1, typename T2, typename T3, int WHICH>
      class Cart2SphOp : public BaseOp<Cart2SphOp<T1, T2, T3, WHICH>>
    {
      private:
        mutable typename detail::base_type_t<T1> x_;
        mutable typename detail::base_type_t<T2> y_;
        mutable typename detail::base_type_t<T3> z_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> x_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> y_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<T3>> z_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(x_), detail::to_jit_storage(y_), detail::to_jit_storage(z_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITCart2Sph_which{}", WHICH);
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
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> x_;\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> y_;\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T3>> z_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      auto x = get_value<CapType>(x_, indices...);\n"
                "      auto y = get_value<CapType>(y_, indices...);\n"
                "      auto z = get_value<CapType>(z_, indices...);\n"
                "      if constexpr (WHICH_ == 0) {{\n"
                "        return scalar_internal_atan2(y, x);\n"
                "      }} else if constexpr (WHICH_ == 1) {{\n"
                "        return scalar_internal_atan2(z, scalar_internal_sqrt(x * x + y * y));\n"
                "      }} else {{\n"
                "        return scalar_internal_sqrt(x * x + y * y + z * z);\n"
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

        __MATX_INLINE__ std::string str() const { return "cart2sph(" + get_type_str(x_) +
          "," + get_type_str(y_) + "," + get_type_str(z_) + ")"; }

        __MATX_INLINE__ Cart2SphOp(const T1 &x, const T2 &y, const T3 &z) : x_(x), y_(y), z_(z)
      {
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
        MATX_ASSERT_COMPATIBLE_OP_SIZES(x);
        MATX_ASSERT_COMPATIBLE_OP_SIZES(y);
        MATX_ASSERT_COMPATIBLE_OP_SIZES(z);
      }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            auto x = get_value<CapType>(x_, indices...);
            auto y = get_value<CapType>(y_, indices...);
            [[maybe_unused]] auto z = get_value<CapType>(z_, indices...);

            if constexpr (WHICH==0) { // theta
              return scalar_internal_atan2(y, x);
            } else if constexpr (WHICH==1) { // phi
              return scalar_internal_atan2(z, scalar_internal_sqrt(x * x + y * y));
            } else {  // r
              return scalar_internal_sqrt(x * x + y * y + z * z);
            }
          }
          else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto x_jit_name = detail::get_operator_capability<Cap>(x_, in);
            const auto y_jit_name = detail::get_operator_capability<Cap>(y_, in);
            const auto z_jit_name = detail::get_operator_capability<Cap>(z_, in);
            return std::format("{}<{},{},{}>", get_jit_class_name(), x_jit_name, y_jit_name, z_jit_name);
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return combine_capabilities<Cap>(true, 
              detail::get_operator_capability<Cap>(x_, in),
              detail::get_operator_capability<Cap>(y_, in),
              detail::get_operator_capability<Cap>(z_, in));
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
            detail::get_operator_capability<Cap>(x_, in);
            detail::get_operator_capability<Cap>(y_, in);
            detail::get_operator_capability<Cap>(z_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(x_, in) +
                   detail::get_operator_capability<Cap>(y_, in) +
                   detail::get_operator_capability<Cap>(z_, in);
          }
          // No specific capabilities enforced
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, 
              detail::get_operator_capability<Cap>(x_, in), 
              detail::get_operator_capability<Cap>(y_, in), 
              detail::get_operator_capability<Cap>(z_, in));
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, 
              detail::get_operator_capability<Cap>(x_, in), 
              detail::get_operator_capability<Cap>(y_, in), 
              detail::get_operator_capability<Cap>(z_, in));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return matx_max(get_rank<T1>(), get_rank<T2>(), get_rank<T3>());
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
					index_t size1 = get_expanded_size<Rank()>(x_, dim);
					index_t size2 = get_expanded_size<Rank()>(y_, dim);
					index_t size3 = get_expanded_size<Rank()>(z_, dim);
					return detail::matx_max(size1, size2, size3);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            x_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            y_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T3>()) {
            z_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            x_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            y_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T3>()) {
            z_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }
  /**
   * Operator to compute spherical cooridantes based on cartisian coordinates
   * @tparam T1
   *   Operator type defining x
   *
   * @tparam T2
   *   Operator type defining y
   *
   * @tparam T3
   *   Operator type defining z
   *
   * @param x
   *   Operator defining x
   *
   * @param y
   *   Operator defining y
   *
   * @param z
   *   Operator defining z
   *
   * @returns
   *   Tuple of operators for theta, phi, and r.
   */
  template <typename T1, typename T2, typename T3>
    auto __MATX_INLINE__ cart2sph(const T1 &x, const T2 &y, const T3 &z)
    {
      return cuda::std::tuple{
        detail::Cart2SphOp<T1, T2, T3, 0>(x, y, z),
        detail::Cart2SphOp<T1, T2, T3, 1>(x, y, z),
        detail::Cart2SphOp<T1, T2, T3, 2>(x, y, z)};
    };

} // end namespace matx
