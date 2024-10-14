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
#include "matx/operators/scalar_ops.h"
#include "matx/operators/base_operator.h"

#define DEFINE_UNARY_OP(FUNCTION, TENSOR_OP)                        \
  template <typename I1,                                            \
            typename = typename std::enable_if_t<is_matx_op<I1>()>> \
  [[nodiscard]] __MATX_INLINE__ auto FUNCTION(const I1 &i1)                         \
  {                                                                 \
    using I1Type = extract_value_type_t<I1>;                       \
    using Op = TENSOR_OP<I1Type>;                                   \
    const typename detail::base_type_t<I1> &base = i1;          \
    return detail::matxUnaryOp(base, Op());                           \
  }


namespace matx
{

  namespace detail {
  template <class I1, class Op>
  class matxUnaryOp :  public BaseOp<matxUnaryOp<I1,Op>>
  {
  private:
    typename detail::base_type_t<I1> in1_;
    typename detail::base_type_t<Op> op_;
    cuda::std::array<index_t, detail::get_rank<I1>()> size_;

  public:
    // dummy type to signal this is a matxop
    using matxop = bool;
    using value_type = typename Op::value_type;
    using matx_width = bool; ///< Signal we can do vector types from this operator
    using self_type = matxUnaryOp<I1, Op>;

    __MATX_INLINE__ const std::string str() const {
      return op_.str() + "(" + get_type_str(in1_) + ")";
    }

    __MATX_INLINE__ matxUnaryOp(const I1 &in1, const Op &op) : in1_(in1), op_(op) {
      if constexpr (Rank() > 0) {
        for (int32_t i = 0; i < Rank(); i++) {
          size_[i] = get_size(in1_, i);
        }
      }
    }

    VecWidth GetMaxWidth() const {
      return GetOpWidth(in1_);
    }

    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(const cuda::std::array<index_t, detail::get_rank<I1>()> &idx) const noexcept
    {
      return cuda::std::apply([&](auto &&...args)  {
          return this->operator()<VecWidth::SCALAR, VecWidth::SCALAR>(args...);
        }, idx);
    }    

    template <VecWidth InWidth, VecWidth OutWidth>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(const cuda::std::array<index_t, detail::get_rank<I1>()> &idx) const noexcept
    {
      return cuda::std::apply([&](auto &&...args)  {
          return this->operator()<InWidth, OutWidth>(args...);
        }, idx);      
    }  

    template <typename... Is, std::enable_if_t<std::conjunction_v<std::is_integral<Is>...>, bool> = true>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
    {
      auto i1 = get_value<VecWidth::SCALAR, VecWidth::SCALAR>(in1_, indices...);
      return op_.template operator()<VecWidth::SCALAR, VecWidth::SCALAR>(i1);
    }

    template <VecWidth InWidth, VecWidth OutWidth, typename... Is, std::enable_if_t<std::conjunction_v<std::is_integral<Is>...>, bool> = true>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
    {
      auto i1 = get_value<InWidth, OutWidth>(in1_, indices...);
      return op_.template operator()<InWidth, OutWidth>(i1);
    }

    static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
    {
      return detail::get_rank<I1>();
    }

    constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
    {
      return size_[dim];
    }

    template <typename ShapeType, typename Executor>
    __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
    {
      in1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }

    template <typename ShapeType, typename Executor>
    __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept  
    {
      in1_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
  };
  }


#ifdef DOXYGEN_ONLY
  /**
 * Compute the square root of each value in a tensor.
 * @param t
 *   Tensor or operator input
 */
  Op sqrt(Op t) {}

  /**
 * Compute the square root of each value in a tensor.
 * @param t
 *   Tensor or operator input
 */
  Op rsqrt(Op t) {}

  /**
 * Compute e^x of each value in a tensor.
 * @param t
 *   Tensor or operator input
 */
  Op exp(Op t) {}

  /**
 * Compute e^(jx) of each value in a tensor where j is sqrt(-1).
 * @param t
 *   Tensor or operator input
 */
  Op expj(Op t) {}

  /**
 * Compute log base 10 of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op log10(Op t) {}

  /**
 * Compute log base 2 of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op log2(Op t) {}

  /**
 * Compute log base e (natural log) of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op log(Op t) {}

  /**
 * Compute log base e (natural log) of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op loge(Op t) {}

  /**
 * Compute the complex conjugate of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op conj(Op t) {}

  /**
 * Compute absolute value of every element in the tensor. For complex numbers
 * this returns the magnitude, or sqrt(x^2+y^2)
 * @param t
 *   Tensor or operator input
 */
  Op abs(Op t) {}

  /**
 * Compute squared absolute value of every element in the tensor. For complex numbers
 * this returns the squared magnitude, or real(t)^2 + imag(t)^2. For real numbers
 * this returns the squared value, or t*t.
 * @param t
 *   Tensor or operator input
 */
  Op abs2(Op t) {}

  /**
 * Compute the sine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op sin(Op t) {}

  /**
 * Compute cosine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op cos(Op t) {}

  /**
 * Compute the tangent of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op tan(Op t) {}

  /**
 * Compute the hyperbolic sine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op sinh(Op t) {}

  /**
 * Compute hyperbolic cosine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op cosh(Op t) {}

  /**
 * Compute the hyperbolic tangent of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op tanh(Op t) {}

  /**
 * Compute the arcsine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op asin(Op t) {}

  /**
 * Compute arccosine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op acos(Op t) {}

  /**
 * Compute the arctangent of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op atan(Op t) {}

  /**
 * Compute the hyperbolic arcsine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op asinh(Op t) {}

  /**
 * Compute hyperbolic arccosine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op acosh(Op t) {}

  /**
 * Compute hyperbolic the arctangent of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op atanh(Op t) {}

  /**
 * Compute the angle of a complex number.
 * @param t
 *   Tensor or operator input
 */
  Op angle(Op t) {}

  /**
 * Compute the principal value of the arctangent of y/x for complex numbers
 * @param t
 *   Tensor or operator input
 */
  Op atan2(Op t) {}

  /**
 * Compute the floor of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op floor(Op t) {}

  /**
 * Compute the ceiling of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op ceil(Op t) {}

  /**
 * Round every element in the tensor
 * @param t
 *   Tensor or operator input
 */
  Op round(Op t) {}

  /**
 * Compute !t (logical NOT) of input tensor or operator
 * @param t
 *   LHS tensor or operator input
 */
  Op operator!(Op t) {}

 /**
 * Negate input tensor or operator
 * @param t
 *   LHS tensor or operator input
 */
  Op operator-(Op t) {}

 /**
 * Return real components of an operator
 * @param t
 *   Input operator
 */
  Op real(Op t) {}  

 /**
 * Return imaginary components of an operator
 * @param t
 *   Input operator
 */
  Op imag(Op t) {}  

 /**
 * Returns a truth value if operator value is NaN
 * @param t
 *   Input operator
 */
  Op isnan(Op t) {}  

 /**
 * Returns a truth value if operator value is infinite
  * @param x
 *   Input operator
 */
  Op isinf( Op x) {}  

 /**
 * Returns values from the standard normal cumulative distribution function
  * @param x
 *   Input operator
 */
  Op normcdf( Op x) {}  

#else
  DEFINE_UNARY_OP(sqrt, detail::SqrtOp);
  DEFINE_UNARY_OP(csqrt, detail::CsqrtOp);
  DEFINE_UNARY_OP(rsqrt, detail::RSqrtOp);
  DEFINE_UNARY_OP(exp, detail::ExpOp);
  DEFINE_UNARY_OP(expj, detail::ExpjOp);
  DEFINE_UNARY_OP(log10, detail::Log10Op);
  DEFINE_UNARY_OP(log2, detail::Log2Op);
  DEFINE_UNARY_OP(log, detail::LogOp);
  DEFINE_UNARY_OP(loge, detail::LogOp);
#if 0
  DEFINE_UNARY_OP(conj, detail::ConjOp);
#else
  // implementing without a macro so we can optimize conj(real)
  template <typename I1,                        
            typename = typename std::enable_if_t<is_matx_op<I1>()>> 
  [[nodiscard]] __MATX_INLINE__ auto conj(I1 i1) {
    using I1Type = extract_value_type_t<I1>;
    if constexpr (is_complex_v<I1Type>) {
      using Op = detail::ConjOp<I1Type>;
      const typename detail::base_type_t<I1> &base = i1;
      return detail::matxUnaryOp(base, Op());
    } else {
      // real type conj is a no-op so return original op.
      return i1;
    }
  }
#endif
  DEFINE_UNARY_OP(abs, detail::AbsOp);
  DEFINE_UNARY_OP(abs2, detail::Abs2Op);
  DEFINE_UNARY_OP(sin, detail::SinOp);
  DEFINE_UNARY_OP(cos, detail::CosOp);
  DEFINE_UNARY_OP(tan, detail::TanOp);
  DEFINE_UNARY_OP(asin, detail::AsinOp);
  DEFINE_UNARY_OP(acos, detail::AcosOp);
  DEFINE_UNARY_OP(atan, detail::AtanOp);
  DEFINE_UNARY_OP(sinh, detail::SinhOp);
  DEFINE_UNARY_OP(cosh, detail::CoshOp);
  DEFINE_UNARY_OP(tanh, detail::TanhOp);
  DEFINE_UNARY_OP(asinh, detail::AsinhOp);
  DEFINE_UNARY_OP(acosh, detail::AcoshOp);
  DEFINE_UNARY_OP(atanh, detail::AtanhOp);
  DEFINE_UNARY_OP(angle, detail::AngleOp);
  DEFINE_UNARY_OP(floor, detail::FloorOp);
  DEFINE_UNARY_OP(ceil, detail::CeilOp);
  DEFINE_UNARY_OP(round, detail::RoundOp);
  DEFINE_UNARY_OP(normcdf, detail::NormCdfOp);
#if 0
  DEFINE_UNARY_OP(real, detail::RealOp);
#else
  // implementing without a macro so we can optimize away real on a real operator
  template <typename I1,                        
            typename = typename std::enable_if_t<is_matx_op<I1>()>> 
  [[nodiscard]] __MATX_INLINE__ auto real(I1 i1) {
    using I1Type = extract_value_type_t<I1>;
    if constexpr (is_complex_v<I1Type>) {
      using Op = detail::RealOp<I1Type>;
      const typename detail::base_type_t<I1> &base = i1;
      return detail::matxUnaryOp(base, Op());
    } else {
      // already real just return i1
      return i1;
    }
  }
#endif
  DEFINE_UNARY_OP(imag, detail::ImagOp);  
  DEFINE_UNARY_OP(operator-, detail::SubNegOp );
  DEFINE_UNARY_OP(isnan, detail::IsNanOp);
  DEFINE_UNARY_OP(isinf, detail::IsInfOp);
#endif

} // end namespace matx
