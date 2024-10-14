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

  template<typename T> __MATX_INLINE__ std::string as_type_str() { return "as_type"; }
  template<> __MATX_INLINE__ std::string as_type_str<float>() { return "as_float"; }
  template<> __MATX_INLINE__ std::string as_type_str<double>() { return "as_double"; }
  template<> __MATX_INLINE__ std::string as_type_str<cuda::std::complex<double>>() { return "cuda::std::complex<double>"; }
  template<> __MATX_INLINE__ std::string as_type_str<cuda::std::complex<float>>() { return "cuda::std::complex<float>"; }
  template<> __MATX_INLINE__ std::string as_type_str<int32_t>() { return "as_int32_t"; }
  template<> __MATX_INLINE__ std::string as_type_str<uint32_t>() { return "as_uint32_t"; }
  template<> __MATX_INLINE__ std::string as_type_str<int16_t>() { return "as_int16_t"; }
  template<> __MATX_INLINE__ std::string as_type_str<uint16_t>() { return "as_uint16_t"; }
  template<> __MATX_INLINE__ std::string as_type_str<int8_t>() { return "as_int8_t"; }
  template<> __MATX_INLINE__ std::string as_type_str<uint8_t>() { return "as_uint8_t"; }

  /**
   * Casts the element of the tensor to a specified type
   *
   * Useful when performing type conversions inside of larger expressions
   *
   */
  namespace detail {
    template <typename T, typename NewType>
      class CastOp : public BaseOp<CastOp<T, NewType>>
    {
      private:
        typename detail::base_type_t<T> op_;

      public:
        using matxop = bool;
        using value_type = NewType;

	      __MATX_INLINE__ std::string str() const { return as_type_str<NewType>() + "(" + op_.str() + ")"; }
        __MATX_INLINE__ CastOp(const T &op) : op_(op){};  

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return static_cast<NewType>(op_.template operator()<VecWidth::SCALAR, VecWidth::SCALAR>(indices...));
        }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return static_cast<NewType>(op_.template operator()<InWidth, OutWidth>(indices...));
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) 
        {
          return static_cast<NewType>(op_.template operator()<VecWidth::SCALAR, VecWidth::SCALAR>(indices...));
        }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return static_cast<NewType>(op_.template operator()<InWidth, OutWidth>(indices...));
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

    template <typename T1, typename T2, typename NewType>
      class ComplexCastOp : public BaseOp<ComplexCastOp<T1, T2, NewType>>
    {
      private:
        typename detail::base_type_t<T1> real_op_;
        typename detail::base_type_t<T2> imag_op_;

      public:
        using matxop = bool;
        using value_type = NewType;
        static_assert(!is_complex_v<T1> && !is_complex_half_v<T1>, "T1 input operator cannot be complex");
        static_assert(!is_complex_v<T2> && !is_complex_half_v<T2>, "T2 input operator cannot be complex");
        static_assert(is_complex_v<NewType> || is_complex_half_v<NewType>, "ComplexCastOp output type should be complex");

	      __MATX_INLINE__ std::string str() const { return as_type_str<NewType>() + "(" + real_op_.str() + "," + imag_op_.str() + ")"; }
        __MATX_INLINE__ ComplexCastOp(T1 real_op, T2 imag_op) : real_op_(real_op), imag_op_(imag_op) {
          static_assert(detail::get_rank<T1>() == detail::get_rank<T2>(), "rank of real and imaginary operators must match");
          if (real_op_.Shape() != imag_op_.Shape()) {
            MATX_THROW(matxInvalidSize, "ComplexCastOp: sizes of input operators must match in all dimensions");
          }
        };

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
        {
          using inner_type = typename inner_op_type_t<NewType>::type;
          return NewType(static_cast<inner_type>(real_op_(indices...)),static_cast<inner_type>(imag_op_(indices...)));
        }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          using inner_type = typename inner_op_type_t<NewType>::type;
          return NewType(static_cast<inner_type>(real_op_(indices...)),static_cast<inner_type>(imag_op_(indices...)));
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            real_op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            imag_op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            real_op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
          if constexpr (is_matx_op<T2>()) {
            imag_op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          // ctor static_assert verifies that detail::get_rank<T>() == detail::get_rank<U>()
          return detail::get_rank<T1>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          // ctor verifies that per dimensions sizes of real_op_ and imag_op_ match
          return real_op_.Size(dim);
        }
    };
  }

  /**
   * @brief Helper function to cast an input operator to a different type
   * 
   * @tparam T Input type
   * @tparam NewType Casted type
   * @param t Input operator
   * @return Operator output casted to NewType 
   */
  template <typename NewType, typename T>
    auto __MATX_INLINE__ as_type(T t)
    {
      if constexpr (std::is_same_v<NewType, typename T::value_type>) {
        // optimized path when type is the same to avoid creating unecessary operators
        return t;
      } else {
        return detail::CastOp<T, NewType>(t);
      }
    };   

  /**
   * @brief Helper function to cast a pair of input operators to a complex type.
   *
   * @tparam T1 Input type for the real components of the complex type
   * @tparam T2 Input type for the imaginary components of the complex type
   * @tparam NewType Casted type (must be complex)
   * @param t1 Input operator of type T1
   * @param t2 Input operator of type T2
   * @return Operator output casted to NewType (must be complex)
   */
  template <typename NewType, typename T1, typename T2>
    auto __MATX_INLINE__ as_complex_type(const T1 &t1, const T2 &t2)
    {
      return detail::ComplexCastOp<T1, T2, NewType>(t1, t2);
    };

  /**
   * @brief Helper function to cast an input operator to an int
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to int 
   */
  template <typename T>
    auto __MATX_INLINE__ as_int(const T &t)
    {
      return as_type<int>(t);
    };   

  /**
   * @brief Helper function to cast an input operator to an float
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to float 
   */
  template <typename T>
    auto __MATX_INLINE__ as_float(const T &t)
    {
      return as_type<float>(t);
    };   

  /**
   * @brief Helper function to cast an input operator to a cuda::std::complex<float>
   *
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to cuda::std::complex<float>
   */
  template <typename T>
    auto __MATX_INLINE__ as_complex_float(const T &t)
    {
      return as_type<cuda::std::complex<float>>(t);
    };

  /**
   * @brief Helper function to cast an input operator to a cuda::std::complex<float>
   *
   * @tparam T1 Input type for real components of the complex output type
   * @tparam T2 Input type for imaginary components of the complex output type
   * @param t1 Input operator for real components of the complex output type
   * @param t2 Input operator for imaginary components of the complex output type
   * @return Operator output casted to cuda::std::complex<float>
   */
  template <typename T1, typename T2>
    auto __MATX_INLINE__ as_complex_float(const T1 &t1, const T2 &t2)
    {
      return as_complex_type<cuda::std::complex<float>>(t1, t2);
    };

  /**
   * @brief Helper function to cast an input operator to a cuda::std::complex<double>
   *
   * @tparam T1 Input type for real components of the complex output type
   * @tparam T2 Input type for imaginary components of the complex output type
   * @param t1 Input operator for real components of the complex output type
   * @param t2 Input operator for imaginary components of the complex output type
   * @return Operator output casted to cuda::std::complex<double>
   */
  template <typename T1, typename T2>
    auto __MATX_INLINE__ as_complex_double(const T1 &t1, const T2 &t2)
    {
      return as_complex_type<cuda::std::complex<double>>(t1, t2);
    };

  /**
   * @brief Helper function to cast an input operator to an double
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to double 
   */
  template <typename T>
    auto __MATX_INLINE__ as_double(const T &t)
    {
      return as_type<double>(t);
    };   

  /**
   * @brief Helper function to cast an input operator to a cuda::std::complex<double>
   *
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to cuda::std::complex<double>
   */
  template <typename T>
    auto __MATX_INLINE__ as_complex_double(const T &t)
    {
      return as_type<cuda::std::complex<double>>(t);
    };

  /**
   * @brief Helper function to cast an input operator to an uint32_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to uint32_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_uint32(const T &t)
    {
      return as_type<uint32_t>(t);
    };   

  /**
   * @brief Helper function to cast an input operator to an int32_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to int32_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_int32(const T &t)
    {
      return as_type<int32_t>(t);
    }; 

  /**
   * @brief Helper function to cast an input operator to an int16_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to int16_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_int16(const T &t)
    {
      return as_type<int16_t>(t);
    }; 

  /**
   * @brief Helper function to cast an input operator to an uint16_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to uint16_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_uint16(const T &t)
    {
      return as_type<uint16_t>(t);
    }; 

  /**
   * @brief Helper function to cast an input operator to an int8_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to int8_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_int8(const T &t)
    {
      return as_type<int8_t>(t);
    }; 

  /**
   * @brief Helper function to cast an input operator to an uint8_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to uint8_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_uint8(const T &t)
    {
      return as_type<uint8_t>(t);
    }; 

} // end namespace matx
