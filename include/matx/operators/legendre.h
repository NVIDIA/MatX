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
        typename detail::base_type_t<T1> n_;
        typename detail::base_type_t<T2> m_;
        typename detail::base_type_t<T3> in_;

        cuda::std::array<int,2> axis_;

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
          static_assert(get_rank<T1>() <= 1, "legendre op:  n must be a scalar, rank 0 or 1 operator");
          static_assert(get_rank<T2>() <= 1, "legendre op:  m must be a scalar, rank 0 or 1 operator");
        }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ value_type operator()(Is... indices) const 
        {
          cuda::std::array<index_t, Rank()> inds{indices...};
          cuda::std::array<index_t, T3::Rank()> xinds{};
          
          int axis1 = axis_[0];
          int axis2 = axis_[1];
          
          // compute n
          index_t nind = inds[axis1];
          int n = get_value(n_, nind);
          
          // compute m 
          index_t mind = inds[axis2];
          int m = get_value(m_, mind);
          
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

          auto x = cuda::std::apply(in_, xinds);

          value_type ret;

          // if we are half precision up cast to float
          if constexpr (is_complex_half_v<value_type>) {
            ret = static_cast<value_type>(legendre(n, m, cuda::std::complex<float>(x)));
          } else if constexpr (is_matx_half_v<value_type>) {
            ret = static_cast<value_type>(legendre(n, m, float(x)));
          } else {
            ret = legendre(n, m, x);
          }
          
          return ret;
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
