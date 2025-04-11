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

  enum class InterpMethod {
    INTERP_METHOD_LINEAR,
    INTERP_METHOD_NEAREST,
    INTERP_METHOD_NEXT,
    INTERP_METHOD_PREV
  };

  namespace detail {
  template <typename OpX, typename OpV, typename OpXQ>
  class InterpolateOp : public BaseOp<InterpolateOp<OpX, OpV, OpXQ>> {
    private:
      typename detail::base_type_t<OpX> x_;    // Sample points
      typename detail::base_type_t<OpV> v_;    // Values at sample points
      typename detail::base_type_t<OpXQ> xq_;  // Query points

    public:
      using matxop = bool;
      using domain_type = typename OpX::value_type;
      using value_type = typename OpV::value_type;
      InterpMethod method_;

      __MATX_INLINE__ std::string str() const { return "interp()"; }

      __MATX_INLINE__ InterpolateOp(const OpX &x, const OpV &v, const OpXQ &xq, InterpMethod method) : 
        x_(x), 
        v_(v),
        xq_(xq),
        method_(method) 
      {
        MATX_ASSERT_STR(x_.Size(0) == v_.Size(0), matxInvalidSize, "interp: sample points and values must have the same size");
      }
      

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        auto x_query = xq_(indices...);
        
        index_t idx_low, idx_high, idx_mid;
        domain_type x_low, x_high, x_mid;

        idx_low = 0;
        idx_high = x_.Size(0) - 1;
        x_low = x_(idx_low);
        if (x_query <= x_low) {
          value_type v = v_(idx_low);
          return v;
        }
        x_high = x_(idx_high);
        if (x_query >= x_high) {
          value_type v = v_(idx_high);
          return v;
        }

        // Find the interval containing the query point
        while (idx_high - idx_low > 1) {
          idx_mid = (idx_low + idx_high) / 2;
          x_mid = x_(idx_mid);
          if (x_query == x_mid) {
            value_type v = v_(idx_mid);
            return v;
          } else if (x_query < x_mid) {
            idx_high = idx_mid;
            x_high = x_mid;
          } else {
            idx_low = idx_mid;
            x_low = x_mid;
          }
        }
        if (method_ == InterpMethod::INTERP_METHOD_LINEAR)
        {
          value_type v_low = v_(idx_low);
          value_type v_high = v_(idx_high);
          return v_low + (x_query - x_low) * (v_high - v_low) / (x_high - x_low);
        }
        else if (method_ == InterpMethod::INTERP_METHOD_NEAREST)
        {
          if (x_query - x_low < x_high - x_query) {
            value_type v = v_(idx_low);
            return v;
          }
          else {
            value_type v = v_(idx_high);
            return v;
          }
        }
        else if (method_ == InterpMethod::INTERP_METHOD_NEXT)
        {
          value_type v = v_(idx_high);
          return v;
        }
        else // if (method_ == InterpMethod::INTERP_METHOD_PREV)
        {
          value_type v = v_(idx_low);
          return v;
        }
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() 
      { 
        return OpXQ::Rank(); 
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return xq_.Size(dim);
      }
    };
  } // namespace detail
/**
 * Linear interpolation operator
 *
 * @tparam OpX
 *   Type of sample points vector
 * @tparam OpV
 *   Type of values vector
 * @tparam OpXQ
 *   Type of query points
 * @param x
 *   Sample points (must be sorted in ascending order)
 * @param v
 *   Values at sample points
 * @param xq
 *   Query points where to interpolate
 * @returns Operator that interpolates values at query points
 */
template <typename OpX, typename OpV, typename OpXQ>
auto interp(const OpX &x, const OpV &v, const OpXQ &xq) {
  static_assert(OpX::Rank() == 1, "interp: sample points must be 1D");
  static_assert(OpV::Rank() == 1, "interp: values must be 1D");
  return detail::InterpolateOp<OpX, OpV, OpXQ>(x, v, xq, InterpMethod::INTERP_METHOD_LINEAR);
}

/**
 * Linear interpolation operator
 *
 * @tparam OpX
 *   Type of sample points vector
 * @tparam OpV
 *   Type of values vector
 * @tparam OpXQ
 *   Type of query points
 * @param x
 *   Sample points (must be sorted in ascending order)
 * @param v
 *   Values at sample points
 * @param xq
 *   Query points where to interpolate
 * @param method
 *   Interpolation method (INTERP_METHOD_LINEAR, INTERP_METHOD_NEAREST, INTERP_METHOD_NEXT, INTERP_METHOD_PREV)
 * @returns Operator that interpolates values at query points
 */
template <typename OpX, typename OpV, typename OpXQ>
auto interp(const OpX &x, const OpV &v, const OpXQ &xq, const InterpMethod method) {
  static_assert(OpX::Rank() == 1, "interp: sample points must be 1D");
  static_assert(OpV::Rank() == 1, "interp: values must be 1D");
  return detail::InterpolateOp<OpX, OpV, OpXQ>(x, v, xq, method);
}

} // namespace matx 