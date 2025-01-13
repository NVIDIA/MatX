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
  namespace detail {
    template <typename T1>
      class R2COp : public BaseOp<R2COp<T1>>
    {
      private:
        typename detail::base_type_t<T1> op_;
        index_t orig_size_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type; 

        __MATX_INLINE__ std::string str() const { return "r2c(" + op_.str() + ")"; }

        __MATX_INLINE__ R2COp(const T1 &op, index_t orig) : op_(op), orig_size_(orig) {
          static_assert(Rank() >= 1, "R2COp must have a rank 1 operator or higher");
        };

        // This version of the operator returns auto rather than decltype(auto) because we need to force the 
        // return type to be by value and not pass through references
        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          auto tup = cuda::std::make_tuple(indices...);

          // If we're on the upper part of the spectrum, return the conjugate of the first half
          if (cuda::std::get<Rank()-1>(tup) >= op_.Size(Rank()-1)) {
            cuda::std::get<Rank()-1>(tup) = orig_size_ - cuda::std::get<Rank()-1>(tup);
            const auto res = cuda::std::apply([&](auto &&...args)  {
                return this->op_.template operator()<InWidth, OutWidth>(args...);
              }, tup);
            return internal_conj(res);
          }

          return cuda::std::apply([&](auto &&...args)  {
              return this->op_.template operator()<InWidth, OutWidth>(args...);
            }, tup);
        }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return std::as_const(*this).template operator()<InWidth, OutWidth>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return (*this).template operator()<VecWidth::SCALAR, VecWidth::SCALAR>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return std::as_const(*this).template operator()<VecWidth::SCALAR, VecWidth::SCALAR>(indices...);
        }      

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
          if (dim == Rank() - 1) {
            return orig_size_;
          }
          else {
            return op_.Size(dim);
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }

  /**
   * Returns the full spectrum from an R2C transform
   *
   * cuFFT's R2C FFTs only return half the spectrum since the other half is the complex
   * conjugate of the first half. This operator returns the full spectrum from the output
   * of an R2C FFT.
   *
   * @tparam T1
   *   Type of View/Op
   * @param t
   *   View/Op to shift
   * @param orig
   *   Original size. Needed to disambiguate between integer division giving same output size
   *
   */
  template <typename T1>
    auto r2c(const T1 &t, index_t orig) { return detail::R2COp<T1>(t, orig); }

} // end namespace matx
