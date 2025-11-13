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
   * Legendre polynomial
   *
   * Calculates the terms of the legendre polyimial(n,m) evaluated
   * at the input X
   */
  namespace detail {
    template <typename T1, typename T2, typename T3>
      class LegendreOp : public BaseOp<LegendreOp<T1,T2,T3>>
    {
      private:
        mutable typename detail::base_type_t<T1> n_;
        mutable typename detail::base_type_t<T2> m_;
        mutable typename detail::base_type_t<T3> in_;

        cuda::std::array<int,2> axis_;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> n_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> m_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<T3>> in_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(n_), detail::to_jit_storage(m_), detail::to_jit_storage(in_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITLegendre_axis{}_{}", axis_[0], axis_[1]);
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
                "  using value_type = typename T3::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static cuda::std::array<int,2> axis_ = {{ {}, {} }};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> n_;\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> m_;\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T3>> in_;\n"
                "  template <typename TypeParam>\n"
                "  static __MATX_INLINE__ __MATX_DEVICE__ TypeParam legendre_calc(int n, int m, TypeParam x) {{\n"
                "    if (m > n) return 0;\n"
                "    TypeParam a = cuda::std::sqrt(TypeParam(1)-x*x);\n"
                "    TypeParam d1 = 1, d0;\n"
                "    for(int i=0; i < m; i++) {{\n"
                "      d0 = d1;\n"
                "      d1 = -TypeParam(2*i+1)*a*d0;\n"
                "    }}\n"
                "    TypeParam p0, p1 = 0, p2 = d1;\n"
                "    for(int l=m; l<n; l++) {{\n"
                "      p0 = p1;\n"
                "      p1 = p2;\n"
                "      p2 = (TypeParam(2*l+1) * x * p1 - TypeParam(l+m)*p0)/(TypeParam(l-m+1));\n"
                "    }}\n"
                "    return p2;\n"
                "  }}\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      cuda::std::array<index_t, Rank_> inds{{indices...}};\n"
                "      cuda::std::array<index_t, {}> xinds;\n"
                "      int axis1 = axis_[0];\n"
                "      int axis2 = axis_[1];\n"
                "      index_t nind = inds[axis1];\n"
                "      int n = get_value<CapType>(n_, nind);\n"
                "      index_t mind = inds[axis2];\n"
                "      int m = get_value<CapType>(m_, mind);\n"
                "      if(axis1>axis2) {{\n"
                "        int tmp = axis1; axis1 = axis2; axis2 = tmp;\n"
                "      }}\n"
                "      int idx = 0;\n"
                "      for(int i = 0; i < Rank_; i++) {{\n"
                "        index_t ind = inds[i];\n"
                "        if(i != axis_[0] && i != axis_[1]) {{\n"
                "          xinds[idx++] = ind;\n"
                "        }}\n"
                "      }}\n"
                "      auto x = get_value<CapType>(in_, xinds);\n"
                "      if constexpr (is_complex_half_v<value_type>) {{\n"
                "        return static_cast<value_type>(legendre_calc(n, m, cuda::std::complex<float>(x)));\n"
                "      }} else if constexpr (is_matx_half_v<value_type>) {{\n"
                "        return static_cast<value_type>(legendre_calc(n, m, float(x)));\n"
                "      }} else {{\n"
                "        return legendre_calc(n, m, x);\n"
                "      }}\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<int>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
                "}};\n",
                func_name, Rank(), axis_[0], axis_[1], detail::array_to_string(out_dims_), T3::Rank())
          );
        }
#endif

        template<class TypeParam>
          static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ TypeParam legendre(int n, int m, TypeParam x) {
            if (m > n ) return 0;

            TypeParam a = cuda::std::sqrt(TypeParam(1)-x*x);
            // first we will move along diagonal

            // initialize registers
            TypeParam d1 = 1, d0;

            for(int i=0; i < m; i++) {
              // advance diagonal (shift)
              d0 = d1;
              // compute next term using recurrence relationship
              d1 = -TypeParam(2*i+1)*a*d0;
            }

            // next we will move to the right till we get to the correct entry

            // initialize registers
            TypeParam p0, p1 = 0, p2 = d1;

            for(int l=m; l<n; l++) {
              // advance one step (shift)
              p0 = p1;
              p1 = p2;

              // Compute next term using recurrence relationship
              p2 = (TypeParam(2*l+1) * x * p1 - TypeParam(l+m)*p0)/(TypeParam(l-m+1));
            }
            return p2;
          }

      public:
        using matxop = bool;
        using value_type = typename T3::value_type;

        __MATX_INLINE__ std::string str() const { return "legendre(" + get_type_str(n_) + "," + get_type_str(m_) + "," + get_type_str(in_) + ")"; }

        __MATX_INLINE__ LegendreOp(const T1 &n, const T2 &m, const T3 &in, cuda::std::array<int,2> axis) : n_(n), m_(m), in_(in), axis_(axis) {
          MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
          static_assert(get_rank<T1>() <= 1, "legendre op:  n must be a scalar, rank 0 or 1 operator");
          static_assert(get_rank<T2>() <= 1, "legendre op:  m must be a scalar, rank 0 or 1 operator");
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            cuda::std::array<index_t, Rank()> inds{indices...};
            cuda::std::array<index_t, T3::Rank()> xinds{};

            int axis1 = axis_[0];
            int axis2 = axis_[1];

            // compute n
            index_t nind = inds[axis1];
            int n = get_value<DefaultCapabilities>(n_, nind);
            
            // compute m 
            index_t mind = inds[axis2];
            int m = get_value<DefaultCapabilities>(m_, mind);
            
            if(axis1>axis2) 
              cuda::std::swap(axis1, axis2);

            // compute indices for x
            int idx = 0;
            for(int i = 0; i < Rank(); i++) {
              index_t ind = inds[i];
              if(i != axis1 && i != axis2) {
                xinds[idx++] = ind;
              }
            }

            auto lret = [](auto ln, auto lm, auto lx) {
              if constexpr (is_complex_half_v<value_type>) {
                return static_cast<value_type>(legendre(ln, lm, cuda::std::complex<float>(lx)));
              } else if constexpr (is_matx_half_v<value_type>) {
                return static_cast<value_type>(legendre(ln, lm, float(lx)));
              } else {
                return legendre(ln, lm, lx);
              }
            };

            auto x = get_value<DefaultCapabilities>(in_, xinds);
            if constexpr (CapType::ept != ElementsPerThread::ONE) {
              Vector<value_type, static_cast<int>(CapType::ept)> ret;
              MATX_LOOP_UNROLL
              for (int e = 0; e < static_cast<int>(CapType::ept); ++e) {
                ret.data[e] = lret(GetVectorVal(n, e), GetVectorVal(m, e), GetVectorVal(x, e));
              }

              return ret;
            }
            else {
              return lret(n, m, x);
            }
          } else {
            return Vector<value_type, static_cast<int>(CapType::ept)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto n_jit_name = detail::get_operator_capability<Cap>(n_, in);
            const auto m_jit_name = detail::get_operator_capability<Cap>(m_, in);
            const auto in_jit_name = detail::get_operator_capability<Cap>(in_, in);
            return std::format("{}<{},{},{}>", get_jit_class_name(), n_jit_name, m_jit_name, in_jit_name);
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return combine_capabilities<Cap>(true, 
              detail::get_operator_capability<Cap>(n_, in),
              detail::get_operator_capability<Cap>(m_, in),
              detail::get_operator_capability<Cap>(in_, in));
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
            detail::get_operator_capability<Cap>(n_, in);
            detail::get_operator_capability<Cap>(m_, in);
            detail::get_operator_capability<Cap>(in_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(n_, in) +
                   detail::get_operator_capability<Cap>(m_, in) +
                   detail::get_operator_capability<Cap>(in_, in);
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(
              my_cap,
              detail::get_operator_capability<Cap>(n_, in),
              detail::get_operator_capability<Cap>(m_, in),
              detail::get_operator_capability<Cap>(in_, in)
            );
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap,
              detail::get_operator_capability<Cap>(n_, in),
              detail::get_operator_capability<Cap>(m_, in),
              detail::get_operator_capability<Cap>(in_, in)
            );
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            n_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            m_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T3>()) {
            in_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            n_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            m_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T3>()) {
            in_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T3>() + 2;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          int axis1 = axis_[0];
          int axis2 = axis_[1];
          if(dim==axis1) {
            return get_size(n_,0);
          } else if (dim==axis2) {
            return get_size(m_,0);
          } else {
            int d = dim;
            if(dim>axis1)
              d--;
            if(dim>axis2)
              d--;
            return get_size(in_, d);
          }
        }
    };
  }

  /**
   * Legendre polynomial operator
   *
   * constructs the legendre polynomial coefficients evaluated at the input operator
   *
   * @tparam T1
   *   Input Operator
   * @tparam m
   *   The degree operator
   * @param in
   *   Operator that computes the location to evaluate the lengrande polynomial
   * @param n
   *   order of the polynomial produced
   * @param m
   *   operator specifing which degrees to output
   *
   * @returns
   *   New operator with Rank+1 and size of last dimension = order.
   */
  template <typename T1, typename T2, typename T3>
    auto __MATX_INLINE__ legendre(const T1 &n, const T2 &m, const T3 &in)
    {
      int axis[2] = {0,1};
      return detail::LegendreOp<T1,T2,T3>(n, m, in, detail::to_array(axis));
    };

  /**
   * Legendre polynomial operator
   *
   * constructs the legendre polynomial coefficients evaluated at the input operator.
   * This version of the API produces all n+1 coefficients
   *
   * @tparam T1
   *   Input Operator
   * @param in
   *   Operator that computes the location to evaluate the lengrande polynomial
   * @param n
   *   order of the polynomial produced
   * @param m
   *   operator specifing which degrees to output
   * @param axis
   *   The axis to write the polynomial coeffients into the output tensor
   *
   * @returns
   *   New operator with Rank+1 and size of last dimension = order.
   */
  template <typename T1, typename T2, typename T3>
  auto __MATX_INLINE__ legendre(const T1 &n, const T2 &m, const T3 &in, cuda::std::array<int, 2> axis)
  {
    return detail::LegendreOp<T1,T2,T3>(n, m, in, axis);
  };

  template <typename T1, typename T2, typename T3>
  auto __MATX_INLINE__ legendre(const T1 &n, const T2 &m, const T3 &in, int (&axis)[2])
  {
    return detail::LegendreOp<T1,T2,T3>(n, m, in, detail::to_array(axis));
  };

} // end namespace matx
