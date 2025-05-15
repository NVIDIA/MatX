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
   * Returns the sign of an element.  -1 if negative, 0 if 0, 1 if positive.
   */
  namespace detail {
    template <typename T>
      class SignOp : public BaseOp<SignOp<T>>
    {
      private:
        typename detail::base_type_t<T> op_;

      public:
        using matxop = bool;
        using value_type = typename T::value_type;

        value_type zval_;

        __MATX_INLINE__ std::string str() const { return "sign(" + get_type_str(op_) + ")"; }
        __MATX_INLINE__ SignOp(const T &op, value_type zval) : op_(op), zval_(zval) {};

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
#ifdef __CUDA_ARCH__
        if constexpr (CapType::jit) {
          if ((threadIdx.x * CapType::ept) >= Size(Rank() - 1)) {
            return detail::GetJitSentinelValue<CapType, value_type>();
          }
        }
#endif
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            auto v = get_value<CapType>(op_,indices...);

            auto set_val = [this](auto vl) {
              if constexpr (is_complex_v<value_type> ) {
                if ( vl == value_type(0)) {
                  return zval_;
                } else {
                  return vl / abs(vl); // sign defintion for complex values
                }
              } else {  // real branch
                if( vl < 0)
                  return value_type(-1);
                else if ( vl > 0 )
                  return value_type(1);
                else
                  return zval_;
              }
            };

            if constexpr (CapType::ept == ElementsPerThread::ONE) {
              return set_val(v);
            }
            else {
              Vector<value_type, static_cast<int>(CapType::ept)> ret;
              MATX_LOOP_UNROLL
              for (int e = 0; e < static_cast<int>(CapType::ept); ++e) {
                ret.data[e] = set_val(GetVectorVal(v, e));
              }
              return ret;
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

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return op_.Size(dim);
        }
    };
  } // end namespace detail

  template <typename T>
  __MATX_INLINE__ auto sign(const T &op, typename T::value_type zval=0) {
    return detail::SignOp(op,zval);
  }

} // end namespace matx
