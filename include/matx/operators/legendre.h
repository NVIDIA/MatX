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
   * Calculates the Nth order legendre polynomial evaluated
   * at the input X
   */
  namespace detail {
    template <typename T1, typename T2>
      class LegendreOp : public BaseOp<LegendreOp<T1,T2>>
    {
      private:
        typename base_type<T1>::type in_;
        T2 m_;
        int order_;
        int axis_;

        template<class TypeParam>
          static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ TypeParam legendre(int n, int m, TypeParam x) {
            if (m > n ) return 0;

            TypeParam a = cuda::std::sqrt(TypeParam(1)-x*x);
            // first we will move move along diagonal

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
        using scalar_type = typename T1::scalar_type;

        __MATX_INLINE__ std::string str() const { return "legendre(" + in_.str() + ")"; }

        __MATX_INLINE__ LegendreOp(int order, T2 m, const T1 in, const int axis) : in_(in), m_(m), order_(order), axis_(axis) {
          static_assert(get_rank<T2>() <= 1, "legendre op:  m must be a scalar or rank 0 or 1 operator");
        }

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
          {
            std::array<index_t, Rank()> inds{indices...};
            std::array<index_t, T1::Rank()> xinds;

            [[maybe_unused]] int mind;
            int m;
            
            // if T2 has rank > 0 we need to compute the thread index for m
            if constexpr (get_rank<T2>()  > 0) { 
              mind = inds[axis_];
              m = get_value(m_, mind);
            } else {
              m = get_value(m_);
            }
            
            // compute m 

            if constexpr (get_rank<T2>()  <= 0) { 
              // scalar output so just fill indices
              xinds = inds;
            } else {
              // vector output so we need to delete the m axis from input
              // fill indices before axis
              for(int i = 0 ; i < axis_; i++) {
                xinds[i] = inds[i];
              }

              // fill indices after axis
              for(int i = axis_; i <  T1::Rank(); i++) {
                xinds[i] = inds[i+1];
              }
            }

            auto x = mapply(in_, xinds);

            scalar_type ret;


            // if we are half precision up cast to float
            if constexpr (is_complex_half_v<scalar_type>) {
              ret = static_cast<scalar_type>(legendre(order_, m, cuda::std::complex<float>(x)));
            } else if constexpr (is_matx_half_v<scalar_type>) {
              ret = static_cast<scalar_type>(legendre(order_, m, float(x)));
            } else {
              ret = legendre(order_, m, x);
            }

            return ret;

          }    

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          if constexpr (detail::get_rank<T2>() <= 0) {
            // if m is a scalar or a rank 0 tensor rank will not increase
            return detail::get_rank<T1>();
          } else {
            return detail::get_rank<T1>()+1;
          }
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          // if m is rank0 or scalar just return input size
          if constexpr (get_rank<T2>() <= 0) {
            return get_size(in_, dim);
          } else {
            if(dim == axis_) {
              // requesting size of polygon coefficients.  Return size of m.
              return m_.Size(0);
            } else {
              int d = dim; 

              // remove axis dim
              if (dim > axis_) {
                d--;
              }

              return get_size(in_, d);
            }
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
   * @param axis
   *   The axis to write the polynomial coeffients into the output tensor
   *
   * @returns
   *   New operator with Rank+1 and size of last dimension = order.
   */
  template <typename T1, typename T2>
    auto __MATX_INLINE__ legendre(int n, T2 m, const T1 in, int axis = -1)
    {
      if(axis == -1) axis = in.Rank();

      return detail::LegendreOp<T1,T2>(n, m, in, axis);
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
   * @param axis
   *   The axis to write the polynomial coeffients into the output tensor
   *
   * @returns
   *   New operator with Rank+1 and size of last dimension = order.
   */
  template <typename T1>
    auto __MATX_INLINE__ legendre(int n, const T1 in, int axis = -1)
    {
      if(axis == -1) axis = in.Rank();

      auto m = range<0, 1, int>({n+1}, 0, 1);

      return detail::LegendreOp<T1,decltype(m)>(n, m, in, axis);
    };
  
} // end namespace matx
