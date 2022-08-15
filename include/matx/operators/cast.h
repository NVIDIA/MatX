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
        typename base_type<T>::type op_;

      public:
        using matxop = bool;
        using scalar_type = NewType;

        __MATX_INLINE__ CastOp(T op) : op_(op){};  

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
          {
            return static_cast<NewType>(op_(indices...));     
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
      return detail::CastOp<T, NewType>(t);
    };   


  /**
   * @brief Helper function to cast an input operator to an int
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to int 
   */
  template <typename T>
    auto __MATX_INLINE__ as_int(T t)
    {
      return detail::CastOp<T, int>(t);
    };   

  /**
   * @brief Helper function to cast an input operator to an float
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to float 
   */
  template <typename T>
    auto __MATX_INLINE__ as_float(T t)
    {
      return detail::CastOp<T, float>(t);
    };   

  /**
   * @brief Helper function to cast an input operator to an double
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to double 
   */
  template <typename T>
    auto __MATX_INLINE__ as_double(T t)
    {
      return detail::CastOp<T, double>(t);
    };   

  /**
   * @brief Helper function to cast an input operator to an uint32_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to uint32_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_uint32(T t)
    {
      return detail::CastOp<T, uint32_t>(t);
    };   

  /**
   * @brief Helper function to cast an input operator to an int32_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to int32_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_int32(T t)
    {
      return detail::CastOp<T, int32_t>(t);
    }; 

  /**
   * @brief Helper function to cast an input operator to an int16_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to int16_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_int16(T t)
    {
      return detail::CastOp<T, int16_t>(t);
    }; 

  /**
   * @brief Helper function to cast an input operator to an uint16_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to uint16_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_uint16(T t)
    {
      return detail::CastOp<T, uint16_t>(t);
    }; 

  /**
   * @brief Helper function to cast an input operator to an int8_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to int8_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_int8(T t)
    {
      return detail::CastOp<T, int8_t>(t);
    }; 

  /**
   * @brief Helper function to cast an input operator to an uint8_t
   * 
   * @tparam T Input type
   * @param t Input operator
   * @return Operator output casted to uint8_t 
   */
  template <typename T>
    auto __MATX_INLINE__ as_uint8(T t)
    {
      return detail::CastOp<T, uint8_t>(t);
    }; 

} // end namespace matx
