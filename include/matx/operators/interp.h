////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

namespace matx {

  // Base interpolation method tag
  struct InterpMethodBase {};

  // Specific interpolation method tags
  struct InterpMethodLinear : public InterpMethodBase {};
  struct InterpMethodNearest : public InterpMethodBase {};
  struct InterpMethodNext : public InterpMethodBase {};
  struct InterpMethodPrev : public InterpMethodBase {};
  struct InterpMethodSpline : public InterpMethodBase {};

  namespace detail {
  template <typename OpX, typename OpV, typename OpXQ, typename Method>
  class InterpolateOp : public BaseOp<InterpolateOp<OpX, OpV, OpXQ, Method>> {
    public:
      using matxop = bool;
      using domain_type = typename OpX::value_type;
      using value_type = typename OpV::value_type;
      using method_type = Method;

    protected:
      typename detail::base_type_t<OpX> x_;    // Sample points
      typename detail::base_type_t<OpV> v_;    // Values at sample points
      typename detail::base_type_t<OpXQ> xq_;  // Query points

      mutable detail::tensor_impl_t<value_type, 1> m_; // Derivatives at sample points (spline only)
      mutable value_type *ptr_m_ = nullptr;

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) searchsorted(const domain_type x_query) const
      {        
        // Binary search to find the interval containing the query point

        // if x_query < x(0), idx_low = n, idx_high = 0
        // if x_query > x(n-1), idx_low = n-1, idx_high = n
        // else x(idx_low) <= x_query <= x(idx_high)

        index_t idx_low, idx_high, idx_mid;
        domain_type x_low, x_high, x_mid;

        idx_low = 0;
        idx_high = x_.Size(0) - 1;
        x_low = x_(idx_low);
        if (x_query < x_low) {
          idx_low = x_.Size(0);
          idx_high = 0;
          return cuda::std::make_tuple(idx_low, idx_high);
        } else if (x_query == x_low) {
          return cuda::std::make_tuple(idx_low, idx_low);
        }
        x_high = x_(idx_high);
        if (x_query > x_high) {
          idx_low = x_.Size(0) - 1;
          idx_high = x_.Size(0);
          return cuda::std::make_tuple(idx_low, idx_high);
        } else if (x_query == x_high) {
          return cuda::std::make_tuple(idx_high, idx_high);
        }

        // Find the interval containing the query point
        while (idx_high - idx_low > 1) {
          idx_mid = (idx_low + idx_high) / 2;
          x_mid = x_(idx_mid);
          if (x_query == x_mid) {
            return cuda::std::make_tuple(idx_mid, idx_mid);
          } else if (x_query < x_mid) {
            idx_high = idx_mid;
            x_high = x_mid;
          } else {
            idx_low = idx_mid;
            x_low = x_mid;
          }
        }
        return cuda::std::make_tuple(idx_low, idx_high);
      }
      
      // Linear interpolation implementation
      template <typename M = Method,
                std::enable_if_t<std::is_same_v<M, InterpMethodLinear>, bool> = true>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ 
      value_type interpolate(const domain_type x_query, index_t idx_low, index_t idx_high) const {
        value_type v;

        if (idx_high == 0 || idx_low == idx_high) { // x_query <= x(0) or x_query == x(idx_low) == x(idx_high)
          v = v_(idx_high);
        } else if (idx_low == x_.Size(0) - 1) { // x_query > x(n-1)
          v = v_(idx_low);
        } else {
          domain_type x_low = x_(idx_low);
          domain_type x_high = x_(idx_high);
          value_type v_low = v_(idx_low);
          value_type v_high = v_(idx_high);
          v = v_low + (x_query - x_low) * (v_high - v_low) / (x_high - x_low);
        }
        return v;
      }

      // Nearest neighbor interpolation implementation
      template <typename M = Method,
                std::enable_if_t<std::is_same_v<M, InterpMethodNearest>, bool> = true>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ 
      value_type interpolate(const domain_type x_query, index_t idx_low, index_t idx_high) const {
        if (idx_low == x_.Size(0)) { // x_query < x(0)
          idx_low = 0;
        } else if (idx_high == x_.Size(0)) { // x_query > x(n-1)
          idx_high = x_.Size(0) - 1;
        }
        domain_type x_low = x_(idx_low);
        domain_type x_high = x_(idx_high);
        index_t idx_nearest = (x_query - x_low < x_high - x_query) ? idx_low : idx_high;
        value_type v = v_(idx_nearest);
        return v;
      }


      // Next value interpolation implementation
      template <typename M = Method,
                std::enable_if_t<std::is_same_v<M, InterpMethodNext>, bool> = true>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ 
      value_type interpolate([[maybe_unused]] const domain_type x_query, [[maybe_unused]] index_t idx_low, index_t idx_high) const {
        if (idx_high == x_.Size(0)) { // x_query > x(n-1)
          idx_high = x_.Size(0) - 1;
        }
        value_type v = v_(idx_high);
        return v;
      }

      // Previous value interpolation implementation
      template <typename M = Method,
                std::enable_if_t<std::is_same_v<M, InterpMethodPrev>, bool> = true>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ 
      value_type interpolate([[maybe_unused]] const domain_type x_query, index_t idx_low, [[maybe_unused]] index_t idx_high) const {
        if (idx_low == x_.Size(0)) { // x_query < x(0)
          idx_low = 0;
        }
        value_type v = v_(idx_low);
        return v;
      }

      // Spline interpolation implementation
      template <typename M = Method,
                std::enable_if_t<std::is_same_v<M, InterpMethodSpline>, bool> = true>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ 
      value_type interpolate([[maybe_unused]] const domain_type x_query, index_t idx_low,  index_t idx_high) const {
        if (idx_low == x_.Size(0)) { // x_query < x(0)
          idx_low = 0;
          idx_high = 1;
        } else if (idx_high == x_.Size(0)) { // x_query > x(n-1)
          idx_high = x_.Size(0) - 1;
          idx_low = x_.Size(0) - 2;
        }

        // Hermite cubic interpolation 

        // sample points
        domain_type x_low = x_(idx_low);
        domain_type x_high = x_(idx_high);

        // values at the sample points
        value_type v_low = v_(idx_low);
        value_type v_high = v_(idx_high);
        value_type v_diff = v_high - v_low;


        // derivatives at the sample points
        value_type m_low = m_(idx_low);
        value_type m_high = m_(idx_high);


        value_type h = x_high - x_low;
        value_type h_low = x_query - x_low;
        value_type h_high = x_high - x_query;
        
        value_type t = h_low / h;
        value_type s = h_high / h;

        value_type v = s * v_low \
          + t * v_high \
          + (h  * (m_low * s - m_high * t) + v_diff * (t - s)) * t * s;

        return v;
      }


    public:
      __MATX_INLINE__ std::string str() const { return "interp()"; }

      __MATX_INLINE__ InterpolateOp(const OpX &x, const OpV &v, const OpXQ &xq) : 
        x_(x), 
        v_(v),
        xq_(xq)
      {
        MATX_ASSERT_STR(x_.Size(0) == v_.Size(0), matxInvalidSize, "interp: sample points and values must have the same size");
      }
      
      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() 
      { 
        return OpXQ::Rank(); 
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return xq_.Size(dim);
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept {
        // Allocate temporary storage for spline coefficients
        cuda::std::array<index_t, OpX::Rank()> m_shape{x_.Size(0)};
        detail::AllocateTempTensor(m_, std::forward<Executor>(ex), m_shape, &ptr_m_);

        (m_ = static_cast<value_type>(0)).run(ex);

        // // Create tensors for tridiagonal system
        // tensor_t<value_type, 1> d_tensor{{x_.Size(0)}};
        // tensor_t<value_type, 1> dl_tensor{{x_.Size(0)}};
        // tensor_t<value_type, 1> du_tensor{{x_.Size(0)}};
        // tensor_t<value_type, 1> b_tensor{{x_.Size(0)}};
        
        // // Fill tridiagonal system via custom operator
        // fill_tridiagonal_system(d_tensor, dl_tensor, du_tensor, b_tensor, v_, x_, xq_, ex);

        // // Solve tridiagonal system
        // solve_tridiagonal(d_tensor, dl_tensor, du_tensor, b_tensor, ex);
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, 
                                  [[maybe_unused]] Executor &&ex) const noexcept {
        matxFree(ptr_m_);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        auto x_query = xq_(indices...);
        auto [idx_low, idx_high] = searchsorted(x_query);

        return interpolate(x_query, idx_low, idx_high);
      }

    };
  } // namespace detail


/**
 * Interpolation operator with specified method
 *
 * @tparam OpX
 *   Type of sample points vector
 * @tparam OpV
 *   Type of values vector
 * @tparam OpXQ
 *   Type of query points
 * @tparam Method
 *   Interpolation method type (InterpMethodLinear, InterpMethodNearest, InterpMethodNext, InterpMethodPrev)
 * @param x
 *   Sample points (must be sorted in ascending order)
 * @param v
 *   Values at sample points
 * @param xq
 *   Query points where to interpolate
 * @returns Operator that interpolates values at query points
 */
template <typename Method = InterpMethodLinear, typename OpX, typename OpV, typename OpXQ>
auto interp(const OpX &x, const OpV &v, const OpXQ &xq) {
  static_assert(OpX::Rank() == 1, "interp: sample points must be 1D");
  static_assert(OpV::Rank() == 1, "interp: values must be 1D");
  static_assert(std::is_base_of_v<InterpMethodBase, Method>, "interp: Method must be a valid interpolation method type");
  return detail::InterpolateOp<OpX, OpV, OpXQ, Method>(x, v, xq);
}

} // namespace matx 