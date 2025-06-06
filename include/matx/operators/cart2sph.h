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
        typename detail::base_type_t<T1> x_;
        typename detail::base_type_t<T2> y_;
        typename detail::base_type_t<T3> z_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ std::string str() const { return "cart2sph(" + get_type_str(x_) +
          "," + get_type_str(y_) + "," + get_type_str(z_) + ")"; }

        __MATX_INLINE__ Cart2SphOp(const T1 &x, const T2 &y, const T3 &z) : x_(x), y_(y), z_(z)
      {
        MATX_ASSERT_COMPATIBLE_OP_SIZES(x);
        MATX_ASSERT_COMPATIBLE_OP_SIZES(y);
        MATX_ASSERT_COMPATIBLE_OP_SIZES(z);
      }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
        {
          if constexpr (EPT == ElementsPerThread::ONE) {
            auto x = get_value<EPT>(x_, indices...);
            auto y = get_value<EPT>(y_, indices...);
            [[maybe_unused]] auto z = get_value<EPT>(z_, indices...);

            if constexpr (WHICH==0) { // theta
              return scalar_internal_atan2(y, x);
            } else if constexpr (WHICH==1) { // phi
              return scalar_internal_atan2(z, scalar_internal_sqrt(x * x + y * y));
            } else {  // r
              return scalar_internal_sqrt(x * x + y * y + z * z);
            }
          }
          else {
            return Vector<value_type, static_cast<index_t>(EPT)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          // No specific capabilities enforced
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            return ElementsPerThread::ONE;
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, 
              detail::get_operator_capability<Cap>(x_), 
              detail::get_operator_capability<Cap>(y_), 
              detail::get_operator_capability<Cap>(z_));
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
