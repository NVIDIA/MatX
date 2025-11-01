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

#define MATX_DEFINE_BINARY_OP(FUNCTION, TENSOR_OP)                        \
  template <typename I1, typename I2>                                \
    requires (is_matx_op_c<I1> || is_matx_op_c<I2>)                 \
  [[nodiscard]] __MATX_INLINE__ auto FUNCTION(const I1 &i1, const I2 &i2)                   \
  {                                                                  \
    using I1Type = extract_value_type_t<I1>;                        \
    using I2Type = extract_value_type_t<I2>;                        \
    using Op = TENSOR_OP<I1Type, I2Type>;                            \
    const typename detail::base_type_t<I1> &base1 = i1;       \
    const typename detail::base_type_t<I2> &base2 = i2;       \
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
    requires (!std::is_same_v<T, S> && std::is_arithmetic_v<S>)
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ 
  auto operator*(const cuda::std::complex<T> &c, S n) -> cuda::std::complex<T>
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
    requires (!std::is_same_v<T, S> && std::is_arithmetic_v<S>)
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ 
  auto operator*(S n, const cuda::std::complex<T> &c) -> cuda::std::complex<T>
  {
    return T(n) * c;
  }


  namespace detail {
    template <class I1, class I2, class Op>
      class matxBinaryOp : public BaseOp<matxBinaryOp<I1,I2,Op>>
    {
      private:
        mutable typename detail::base_type_t<I1> in1_;
        mutable typename detail::base_type_t<I2> in2_;
        typename detail::base_type_t<Op> op_;

      public:
        // dummy type to signal this is a matxop
        using matxop = bool;
        using value_type = typename Op::value_type;
        using self_type = matxBinaryOp<I1, I2, Op>;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<I1>> in1_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<I2>> in2_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<Op>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(in1_), detail::to_jit_storage(in2_), detail::to_jit_storage(op_)};
        }        
#endif        

      __MATX_INLINE__ const std::string str() const {
        return op_.str(get_type_str(in1_), get_type_str(in2_));
      }


      __MATX_INLINE__ matxBinaryOp(const I1 &in1, const I2 &in2, const Op &op) : in1_(in1), in2_(in2), op_(op)
      {
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
        if constexpr (Rank() > 0)
        {
          MATX_ASSERT_COMPATIBLE_OP_SIZES(in1_);
          MATX_ASSERT_COMPATIBLE_OP_SIZES(in2_);
        }
      }

      template <typename CapType, typename... Is>
        requires (cuda::std::conjunction_v<cuda::std::is_integral<Is>...>)
      __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ decltype(auto) operator()(Is... indices) const
      {
        const auto &lhs = in1_;
        const auto &rhs = in2_;
        const auto i1 = get_value<CapType>(lhs, indices...);
        const auto i2 = get_value<CapType>(rhs, indices...);

        return op_.template operator()<CapType>(i1, i2);
      }

      template <typename... Is>
        requires (cuda::std::conjunction_v<cuda::std::is_integral<Is>...>)
      __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ decltype(auto) operator()(Is... indices) const
      {
        return this->template operator()<DefaultCapabilities>(indices...);
      }      

      template <typename CapType, typename ArrayType>
        requires is_std_array_c<ArrayType>
      __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto operator()(const ArrayType &idx) const noexcept
      {
        return cuda::std::apply([&](auto &&...args)  {
            return this->operator()<CapType>(args...);
          }, idx);
      }

#ifdef MATX_EN_JIT
      __MATX_INLINE__ std::string get_jit_class_name() const {
        std::string symbol_name = "JITBinOp";
        return symbol_name;
      }

      __MATX_INLINE__ auto get_jit_op_str() const {
        cuda::std::array<index_t, Rank()> out_dims_;
        for (int i = 0; i < Rank(); ++i) {
          out_dims_[i] = Size(i);
        }
        const std::string func_name = get_jit_class_name();
        return cuda::std::make_tuple(
           func_name, 
           std::string("template <typename I1, typename I2, typename Op> struct " + func_name + "  {\n") + 
               "  using value_type = typename Op::value_type;\n" +
               "  using matxop = bool;\n" +
               "  constexpr static cuda::std::array<index_t, " + std::to_string(Rank()) + "> out_dims_ = { " + 
               detail::array_to_string(out_dims_) + " };\n" +
               "  typename detail::inner_storage_or_self_t<detail::base_type_t<I1>> in1_;\n" +
               "  typename detail::inner_storage_or_self_t<detail::base_type_t<I2>> in2_;\n" +
               "  typename detail::inner_storage_or_self_t<detail::base_type_t<Op>> op_;\n" +
               "  template <typename CapType, typename... Is>\n" +
               "  __MATX_INLINE__ __MATX_DEVICE__  decltype(auto) operator()(Is... indices) const\n" +
               "  {\n" +
               "    if ((threadIdx.x * static_cast<int>(CapType::ept)) > Size(Rank() - 1)) {\n" +
               "      return detail::GetJitSentinelValue<CapType, value_type>();\n" +
               "    }\n" +
               "    auto i1 = get_value<CapType>(in1_, indices...);\n" + 
               "    auto i2 = get_value<CapType>(in2_, indices...);\n" +
               "    return op_.template operator()<CapType>(i1, i2);\n" + 
               "  }\n" +
               "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n" +
               "  {\n" +
               "    return detail::matx_max(detail::get_rank<I1>(), detail::get_rank<I2>());\n" +
               "  }\n" +
               "  constexpr __MATX_INLINE__  __MATX_DEVICE__ index_t Size(int dim) const\n" +
               "  {\n" +
               "    return out_dims_[dim];\n " +
               "  }\n" +          
               "};\n"
        );
      }      
#endif

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
        if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
          // No need to use combine_capabilities here since we're just returning a string.
          const auto lhs_jit_name = detail::get_operator_capability<Cap>(in1_, in);
          const auto rhs_jit_name = detail::get_operator_capability<Cap>(in2_, in);
          const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);

          return get_jit_class_name() + "<" + lhs_jit_name + "," + rhs_jit_name + "," + op_jit_name + ">";
#else
          return "";
#endif
        }
        else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
          // Get the key/value pair from get_jit_op_str()
          const auto [key, value] = get_jit_op_str();
          
          // Insert into the map if the key doesn't exist
          if (in.find(key) == in.end()) {
            in[key] = value;
          }
          
          // Also handle child operators
          detail::get_operator_capability<Cap>(in1_, in);
          detail::get_operator_capability<Cap>(in2_, in);
          detail::get_operator_capability<Cap>(op_, in);
          
          // Always return true for now
          return true;
#else
          return false;
#endif
        }    
        else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
          // The dynamic shmem size is the sum of the dynamic shmem sizes of the two operands.
          return 
            detail::get_operator_capability<Cap>(in1_, in) +
            detail::get_operator_capability<Cap>(in2_, in);
        }   
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, 
                                        detail::get_operator_capability<Cap>(in1_, in),
                                        detail::get_operator_capability<Cap>(in2_, in));
        }
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

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<I1>()) {
          in1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<I2>()) {
          in2_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<I1>()) {
          in1_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<I2>()) {
          in2_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
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
   * Compute element-wise max(t, t2) of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op max(Op t, Op t2) {}

  /**
   * Compute element-wise min(t, t2) of two operators or tensors
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
   * Compute t & t2 (bitwise AND) of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator&(Op t, Op t2) {}

  /**
   * Compute t | t2 (bitwise OR) of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator|(Op t, Op t2) {}

  /**
   * Compute t ^ t2 (bitwise XOR) of two operators or tensors
   * @param t
   *   LHS tensor or operator input
   * @param t2
   *   RHS tensor or operator input
   */
  Op operator^(Op t, Op t2) {}

  /**
   * Compute the arctangent of two inputs
   * @param t
   *   X value of input
   * @param t2
   *   Y value of input
   */
  Op atan2(Op t, Op t2) {}
#else
  MATX_DEFINE_BINARY_OP(operator+, detail::AddOp);
  MATX_DEFINE_BINARY_OP(operator-, detail::SubOp);
  MATX_DEFINE_BINARY_OP(operator*, detail::MulOp);
  MATX_DEFINE_BINARY_OP(mul, detail::MulOp);
  MATX_DEFINE_BINARY_OP(operator/, detail::DivOp);
  MATX_DEFINE_BINARY_OP(operator%, detail::ModOp);
  MATX_DEFINE_BINARY_OP(fmod, detail::FModOp);
  MATX_DEFINE_BINARY_OP(operator|, detail::OrOp);
  MATX_DEFINE_BINARY_OP(operator&, detail::AndOp);
  MATX_DEFINE_BINARY_OP(operator^, detail::XorOp);
  MATX_DEFINE_BINARY_OP(pow, detail::PowOp);
  MATX_DEFINE_BINARY_OP(max, detail::MaximumOp);
  MATX_DEFINE_BINARY_OP(atan2, detail::Atan2Op);
  MATX_DEFINE_BINARY_OP(min, detail::MinimumOp);
  MATX_DEFINE_BINARY_OP(operator<, detail::LTOp);
  MATX_DEFINE_BINARY_OP(operator>, detail::GTOp);
  MATX_DEFINE_BINARY_OP(operator<=, detail::LTEOp);
  MATX_DEFINE_BINARY_OP(operator>=, detail::GTEOp);
  MATX_DEFINE_BINARY_OP(operator==, detail::EQOp);
  MATX_DEFINE_BINARY_OP(operator!=, detail::NEOp);
  MATX_DEFINE_BINARY_OP(operator&&, detail::AndAndOp);
  MATX_DEFINE_BINARY_OP(operator||, detail::OrOrOp);
  MATX_DEFINE_UNARY_OP(operator!, detail::NotOp);
#endif

} // end namespace matx
