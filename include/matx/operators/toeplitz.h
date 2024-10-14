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
   * Construct a Toeplitz matrix
   */
  namespace detail {
    template <typename T1, typename T2>
      class TopelitzOp : public BaseOp<TopelitzOp<T1, T2>>
    {
      private:
        typename detail::base_type_t<T1> op1_;
        typename detail::base_type_t<T2> op2_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ std::string str() const { 
          std::string top1;
          if constexpr (is_matx_op<T1>()) {
            top1 = op1_.str();
          }
          else {
            top1 = "array";
          }

          std::string top2;
          if constexpr (is_matx_op<T2>()) {
            top2 = op2_.str();
          }
          else {
            top2 = "array";
          }        

          return "toeplitz(" + top1 + "," + top2 + ")"; 
        }

          __MATX_INLINE__ TopelitzOp(const T1 &op1, const T2 &op2) : op1_(op1), op2_(op2)
        {
          if constexpr (is_matx_op<T1>()) {
            static_assert(T1::Rank() == 1, "toeplitz() operator input rank must be 1");
          }

          if constexpr (!std::is_same_v<T2, EmptyOp>) {
            static_assert(std::is_same_v<typename T1::value_type, 
                                        typename T2::value_type>, "Input types to toeplitz() must match");
            if constexpr (is_matx_op<T2>()) {
              static_assert(T2::Rank() == 1, "toeplitz() operator input rank must be 1");
            }
          }        
        }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(index_t i, index_t j) const
        {
          if (j > i) {
            if constexpr (is_matx_op<T2>()) {
              auto val = op2_(j - i);
              return val;
            }
            else {
              auto val = op2_[j - i];
              return val;
            }
          }
          else {
            if constexpr (is_matx_op<T1>()) {
              auto val = op1_(i - j);
              return val;
            }
            else {
              auto val = op1_[i - j];
              return val;
            }          
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            op2_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }          
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op1_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<T2>()) {
            op2_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          } 
        }          

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return 2;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          if constexpr (is_matx_op<T1>()) {
            return (dim == 0) ? op1_.Size(0) : op2_.Size(0);
          }
          else {
            return (dim == 0) ? op1_.size() : op2_.size();
          }
        }
    };
  }

  /**
   * Toeplitz operator
   *
   * The toeplitz operator constructs a toeplitz matrix
   *
   * @tparam T
   *   Type of data
   * @tparam D
   *   Length of array
   * @param c
   *   Operator or view for first input
   *
   * @returns
   *   New operator of the kronecker product
   */
  template <typename T, int D>
  auto __MATX_INLINE__ toeplitz(const T (&c)[D])
  {
    const auto op = detail::to_array(c);
    const auto op2 = op;
    if constexpr (is_complex_v<T>) {
      cuda::std::transform(op2.begin(), op2.end(), [](T val){ return _internal_conj(val); } );
    }
    return detail::TopelitzOp(op, op2);
  };

  /**
   * Toeplitz operator
   *
   * The toeplitz operator constructs a toeplitz matrix
   *
   * @tparam T1
   *   Type of first input
   * @param c
   *   Operator or view for first input
   *
   * @returns
   *   New operator of the kronecker product
   */
  template <typename Op, std::enable_if_t<!std::is_array_v<typename remove_cvref<Op>::type>, bool> = true>
  auto __MATX_INLINE__ toeplitz(const Op &c)
  {
    if constexpr (is_complex_v<typename Op::value_type>) {
      return detail::TopelitzOp(c, conj(c));
    }
    else {
      return detail::TopelitzOp(c, c);
    }
  };  

  /**
   * Toeplitz operator
   *
   * The toeplitz operator constructs a toeplitz matrix
   *
   * @tparam T
   *   Type of matrix
   * @tparam D1
   *   Length of c
   * @tparam D2
   *   Length of r
   * @param c
   *   First column of the matrix
   * @param r
   *   First row of the matrix
   *
   * @returns
   *   New operator of the kronecker product
   */
  template <typename T, int D1, int D2>
  auto __MATX_INLINE__ toeplitz(const T (&c)[D1], const T (&r)[D2])
  {
    const auto cop = detail::to_array(c);
    const auto rop = detail::to_array(r);    
    return detail::TopelitzOp(cop, rop);
  };

  /**
   * Toeplitz operator
   *
   * The toeplitz operator constructs a toeplitz matrix
   *
   * @tparam COp
   *   Operator type of c
   * @tparam ROp
   *   Operator type of r 
   * @param c
   *   First column of the matrix
   * @param r
   *   First row of the matrix
   *
   * @returns
   *   New operator of the kronecker product
   */
  template <typename COp, typename ROp, std::enable_if_t< !std::is_array_v<typename remove_cvref<COp>::type> && 
                                                          !std::is_array_v<typename remove_cvref<ROp>::type>, 
                                                          bool> = true>
  auto __MATX_INLINE__ toeplitz(const COp &c, const ROp &r)
  {
    return detail::TopelitzOp<COp, ROp>(c, r);
  };    
} // end namespace matx
