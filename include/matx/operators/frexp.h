////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// COpBright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above cOpBright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above cOpBright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the cOpBright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COpBRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
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

namespace detail {
  template<typename OpA, int WHICH>
  class FrexpOp : public BaseOp<FrexpOp<OpA, WHICH>>
  {
    private:
      typename detail::base_type_t<OpA> a_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;

      __MATX_INLINE__ std::string str() const { return "frexp()"; }
      __MATX_INLINE__ FrexpOp(const OpA &a) : a_(a) { 
        static_assert(std::is_floating_point_v<value_type> ||
                      is_cuda_complex_v<value_type>, "frexp() must take a floating point input");

      };

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
      {
        [[maybe_unused]] int rexp;        
        if constexpr (is_cuda_complex_v<value_type>) {
          if constexpr (std::is_same_v<float, typename value_type::value_type>) {
            if constexpr (WHICH == 0) { // real fractional
              const auto frac = cuda::std::frexpf(a_(indices...).real(), &rexp);
              return frac;
            } else if constexpr (WHICH == 1) { // real exponent
              [[maybe_unused]] const auto frac = cuda::std::frexpf(a_(indices...).real(), &rexp);
              return rexp;
            } else if constexpr (WHICH == 2) { // imag fractional
              const auto frac = cuda::std::frexpf(a_(indices...).imag(), &rexp);
              return frac;
            } else if constexpr (WHICH == 3) { // imag exponent
              [[maybe_unused]] const auto frac = cuda::std::frexpf(a_(indices...).imag(), &rexp);
              return rexp;
            }
          }
          else {
            if constexpr (WHICH == 0) { // real fractional
              const auto frac = cuda::std::frexp(a_(indices...).real(), &rexp);
              return frac;
            } else if constexpr (WHICH == 1) { // real exponent
              [[maybe_unused]] const auto frac = cuda::std::frexp(a_(indices...).real(), &rexp);
              return rexp;
            } else if constexpr (WHICH == 2) { // imag fractional
              const auto frac = cuda::std::frexp(a_(indices...).imag(), &rexp);
              return frac;
            } else if constexpr (WHICH == 3) { // imag exponent
              [[maybe_unused]] const auto frac = cuda::std::frexp(a_(indices...).imag(), &rexp);
              return rexp;
            }
          }
        }
        else {
          if constexpr (std::is_same_v<float, value_type>) {
            [[maybe_unused]] const float frac = cuda::std::frexpf(a_(indices...), &rexp);
            if constexpr (WHICH == 0) { // fractional
              return frac;
            } else if constexpr (WHICH == 1) { // exponent
              return rexp;
            }
          }
          else {
            [[maybe_unused]] const double frac = cuda::std::frexp(a_(indices...), &rexp);
            if constexpr (WHICH == 0) { // fractional
              return frac;
            } else if constexpr (WHICH == 1) { // exponent
              return rexp;
            }
          }
        }
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
      }

  };
}

template<typename OpA>
__MATX_INLINE__ auto frexp(const OpA &a) {
  return cuda::std::tuple{
          detail::FrexpOp<OpA, 0>(a),
          detail::FrexpOp<OpA, 1>(a)
  };
}

template<typename OpA>
__MATX_INLINE__ auto frexpc(const OpA &a) {
  return cuda::std::tuple{
          detail::FrexpOp<OpA, 0>(a),
          detail::FrexpOp<OpA, 1>(a),
          detail::FrexpOp<OpA, 2>(a),
          detail::FrexpOp<OpA, 3>(a)
  };
}

};

