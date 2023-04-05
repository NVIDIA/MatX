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
#include "matx/operators/scalar_ops.h"

#define DEFINE_BINARY_OP(FUNCTION, TENSOR_OP)                        \
  template <typename I1, typename I2,                                \
            typename = typename std::enable_if_t<is_matx_op<I1>() or \
                                                 is_matx_op<I2>()>>  \
  [[nodiscard]] __MATX_INLINE__ auto FUNCTION(I1 i1, I2 i2)                   \
  {                                                                  \
    using I1Type = extract_scalar_type_t<I1>;                        \
    using I2Type = extract_scalar_type_t<I2>;                        \
    using Op = TENSOR_OP<I1Type, I2Type>;                            \
    const typename detail::base_type<I1>::type &base1 = i1;       \
    const typename detail::base_type<I2>::type &base2 = i2;       \
    return detail::matxBinaryOp(base1, base2, Op());              \
  }

namespace matx
{
  /**
   * @brief Utility operator for multiplying scalars by a complex value
   * 
   * @tparam T Complex type
   * @tparam S Scalar type
   * @param n Scalar value
   * @param c Complex value
   * @return Product result
   */
  template <typename T, typename S>
    __MATX_INLINE__
    typename std::enable_if_t<!std::is_same_v<T, S> && std::is_arithmetic_v<S>,
             cuda::std::complex<T>>
               __MATX_HOST__ __MATX_DEVICE__ operator*(const cuda::std::complex<T> &c, S n)
               {
                 return c * T(n);
               }

  /**
   * @brief Utility operator for multiplying scalars by a complex value
   * 
   * @tparam T Complex type
   * @tparam S Scalar type
   * @param n Scalar value
   * @param c Complex value
   * @return Product result
   */
  template <typename T, typename S>
    __MATX_INLINE__
    typename std::enable_if_t<!std::is_same_v<T, S> && std::is_arithmetic_v<S>,
             cuda::std::complex<T>>
               __MATX_HOST__ __MATX_DEVICE__ operator*(S n, const cuda::std::complex<T> &c)
               {
                 return T(n) * c;
               }


  namespace detail {
    template <class I1, class I2, class Op>
      class matxBinaryOp : public BaseOp<matxBinaryOp<I1,I2,Op>>
    {
      private:
        typename base_type<I1>::type in1_;
        typename base_type<I2>::type in2_;
        typename base_type<Op>::type op_;

      public:
        // dummy type to signal this is a matxop
        using matxop = bool;
        using scalar_type = typename Op::scalar_type;
        using self_type = matxBinaryOp<I1, I2, Op>;

      __MATX_INLINE__ const std::string str() const {
        return op_.str(get_type_str(in1_), get_type_str(in2_));
      }

        __MATX_INLINE__ matxBinaryOp(I1 in1, I2 in2, Op op) : in1_(in1), in2_(in2), op_(op)
      {
        if constexpr (Rank() > 0)
        {
          ASSERT_COMPATIBLE_OP_SIZES(in1_);
          ASSERT_COMPATIBLE_OP_SIZES(in2_);
        }
      }

      template <typename... Is, std::enable_if_t<std::conjunction_v<std::is_integral<Is>...>, bool> = true>
      __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ auto operator()(Is... indices) const
      {
        auto i1 = get_value(in1_, indices...);
        auto i2 = get_value(in2_, indices...);
        return op_(i1, i2);
      }

      template <typename ArrayType, std::enable_if_t<is_std_array_v<ArrayType>, bool> = true>
      __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const auto operator()(const ArrayType &idx) const noexcept
      {
        return mapply([&](auto &&...args)  {
            return this->operator()(args...);
          }, idx);      
      }        


      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return detail::matx_max(detail::get_rank<I1>(), detail::get_rank<I2>());
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
      {
        index_t size1 = detail::get_expanded_size<Rank()>(in1_, dim);
        index_t size2 = detail::get_expanded_size<Rank()>(in2_, dim);
        return detail::matx_max(size1,size2);
      }
    };
  }


#ifdef DOXYGEN_ONLY
  /**
   * Add two operators or tensors
   * @param t
   *   Tensor or operator input
   * @param t2
   *   RHS second tensor or operator input
   */
  Op operator+(Op t, Op t2) {}

  /**
   * Subtract two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS second tensor or operator input
   */
  Op operator-(Op t, Op t2) {}

  /**
   * Multiply two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS second tensor or operator input
   */
  Op operator*(Op t, Op t2) {}

  /**
   * Multiply two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS second tensor or operator input
   */
  Op mul(Op t, Op t2) {}

  /**
   * Divide two operators or tensors
   * @param t
   *   LHS tensor numerator
   * @param t2
   *   RHS tensor or operator denominator
   */
  Op operator/(Op t, Op t2) {}

  /**
   * Modulo two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS second tensor or operator modulus
   */
  Op operator%(Op t, Op t2) {}

  /**
   * Modulo two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS second tensor or operator modulus
   */
  Op fmod(Op t, Op t2) {}


  /**
   * Compute the t^t2 of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator power
   */
  Op pow(Op t, Op t2) {}

  /**
   * Compute max(t, t2) of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op max(Op t, Op t2) {}

  /**
   * Compute min(t, t2) of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op min(Op t, Op t2) {}

  /**
   * Compute t < t2 of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator<(Op t, Op t2) {}

  /**
   * Compute t > t2 of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator>(Op t, Op t2) {}

  /**
   * Compute t <= t2 of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator<=(Op t, Op t2) {}

  /**
   * Compute t >= t2 of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator>=(Op t, Op t2) {}

  /**
   * Compute t == t2 of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator==(Op t, Op t2) {}

  /**
   * Compute t != t2 of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator!=(Op t, Op t2) {}

  /**
   * Compute t && t2 (logical AND) of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator&&(Op t, Op t2) {}

  /**
   * Compute t || t2 (logical OR) of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator||(Op t, Op t2) {}

  /**
   * Compute the arctangent of two inputs
   * @param t
   *   X value of input
   * @param t2
   *   Y value of input
   */
  Op atan2(Op t, Op t2) {}  
#else
  DEFINE_BINARY_OP(operator+, detail::AddOp);
  DEFINE_BINARY_OP(operator-, detail::SubOp);
  DEFINE_BINARY_OP(operator*, detail::MulOp);
  DEFINE_BINARY_OP(mul, detail::MulOp);
  DEFINE_BINARY_OP(operator/, detail::DivOp);
  DEFINE_BINARY_OP(operator%, detail::ModOp);
  DEFINE_BINARY_OP(fmod, detail::FModOp);
  DEFINE_BINARY_OP(operator|, detail::OrOp);
  DEFINE_BINARY_OP(operator&, detail::AndOp);
  DEFINE_BINARY_OP(operator^, detail::XorOp);
  DEFINE_BINARY_OP(pow, detail::PowOp);
  DEFINE_BINARY_OP(max, detail::MaxOp);
  DEFINE_BINARY_OP(atan2, detail::Atan2Op);
  DEFINE_BINARY_OP(min, detail::MinOp);
  DEFINE_BINARY_OP(operator<, detail::LTOp);
  DEFINE_BINARY_OP(operator>, detail::GTOp);
  DEFINE_BINARY_OP(operator<=, detail::LTEOp);
  DEFINE_BINARY_OP(operator>=, detail::GTEOp);
  DEFINE_BINARY_OP(operator==, detail::EQOp);
  DEFINE_BINARY_OP(operator!=, detail::NEOp);
  DEFINE_BINARY_OP(operator&&, detail::AndAndOp);
  DEFINE_BINARY_OP(operator||, detail::OrOrOp);
  DEFINE_UNARY_OP(operator!, detail::NotOp);
#endif

} // end namespace matx
