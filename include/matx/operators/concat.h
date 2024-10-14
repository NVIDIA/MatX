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
   * ConcatOp operators
   *
   * Class for concatening operators along a single dimension. Sizes of the operators not
   * being concatenated must be the same, and the new operator has dimensions equal to the original
   * operator on non-index dimension, and the sum of sizes along the index dimension.
   */
  namespace detail {  
    template <typename... Ts>
      class ConcatOp : public BaseOp<ConcatOp<Ts...>>
    {
      using first_type = cuda::std::tuple_element_t<0, cuda::std::tuple<Ts...>>;
      using first_value_type = typename first_type::value_type;
      using self_type = ConcatOp<Ts...>;

      static constexpr int RANK = first_type::Rank();

      public:
      using matxop = bool;
      using matxoplvalue = bool;
      using shape_type = index_t;

      // Scalar type of operation
      using value_type = first_value_type;

      template <int I = -1>
        __MATX_INLINE__ std::string get_str() const {
          if constexpr (I==-1) return "concat(" + get_str<I+1>();
          else if constexpr (I < sizeof...(Ts)-1) return cuda::std::get<I>(ops_).str() + "," + get_str<I+1>();
          else if constexpr (I == sizeof...(Ts)-1) return cuda::std::get<I>(ops_).str() + ")";
          else return "";
        }

      __MATX_INLINE__ std::string str() const {
        return get_str<-1>();
      }

      __MATX_INLINE__ ConcatOp(int axis, const Ts&... ts) : ops_(ts...), axis_(axis)
      {
        static_assert(RANK > 0, "Cannot concatenate rank-0 tensors");
        static_assert(sizeof...(Ts) > 1, "Must have more than one tensor to concatenate");
        static_assert((... && (RANK == ts.Rank())), "concatenated ops must have the same rank");

        for (int32_t i = 0; i < RANK; i++) {
          if(i == axis_) {
            size_ = (ts.Size(i) + ...);
          } else {
            MATX_ASSERT_STR(((ts.Size(i) == pp_get<0>(ts).Size(i)) && ...)
                , matxInvalidSize, "concatenate operators must have the same size in non-axis dimension");
          }
        }
      }

      template <int I = 0, int N>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto GetVal(cuda::std::array<index_t,RANK> &indices) const {

        if constexpr ( I == N ) {
          // This should never happen
          return value_type(-9999);
          // returning this to satisfy lvalue requirements
        } else {
          const auto &op = cuda::std::get<I>(ops_);
          auto idx = indices[axis_];
          auto size = op.Size(axis_);
          // If in range of this operator
          if(idx < size) {
            // evaluate operator
            return cuda::std::apply(op, indices);
          } else {
            // otherwise remove this operator and recurse
            indices[axis_] -= size;
            return GetVal<I+1, N>(indices);
          }
        }
      }
      
      template <int I = 0, int N>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) GetVal(cuda::std::array<index_t,RANK> &indices) {

        if constexpr ( I == N ) {
          // This should never happen
          // returning this to satisfy lvalue requirements
          auto &op = cuda::std::get<I-1>(ops_);
          return cuda::std::apply(op, indices);
        } else {
          auto &op = cuda::std::get<I>(ops_);
          auto idx = indices[axis_];
          auto size = op.Size(axis_);
          // If in range of this operator
          if(idx < size) {
            // evaluate operator
            return cuda::std::apply(op, indices);
          } else {
            // otherwise remove this operator and recurse
            indices[axis_] -= size;
            return GetVal<I+1, N>(indices);
          }
        }
      }

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        cuda::std::array<index_t, sizeof...(Is)> indices = {{is...}};
        return GetVal<0, sizeof...(Ts)>(indices);
      }
      
      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        cuda::std::array<index_t, sizeof...(Is)> indices = {{is...}};
        return GetVal<0, sizeof...(Ts)>(indices);
      }


      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() noexcept
      {
        return RANK;
      }

      constexpr index_t __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const noexcept
      {
        if(dim==axis_)
          return size_;
        else
          return cuda::std::get<0>(ops_).Size(dim);
      }

      ~ConcatOp() = default;
      ConcatOp(const ConcatOp &rhs) = default;
      __MATX_INLINE__ auto operator=(const self_type &rhs) { 
        return set(*this, rhs); 
      }        

      template<typename R> 
      __MATX_INLINE__ auto operator=(const R &rhs) { 
        if constexpr (is_matx_transform_op<R>()) {
          return mtie(*this, rhs);
        }
        else {          
          return set(*this, rhs); 
        }
      }

      template <int I, typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (I < sizeof...(Ts)-1) {
          if constexpr (is_matx_op<cuda::std::tuple_element_t<I,cuda::std::tuple<Ts...>>>()) {
            cuda::std::get<I>(ops_).PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
            PreRun<I+1, ShapeType, Executor>(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        } else if constexpr (I == sizeof...(Ts)-1) {
          if constexpr (is_matx_op<cuda::std::tuple_element_t<I,cuda::std::tuple<Ts...>>>()) {
            cuda::std::get<I>(ops_).PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
            // This was the last ops_ element, so stop recursion
          }
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        PreRun<0, ShapeType, Executor>(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
      }

      template <int I, typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (I < sizeof...(Ts)-1) {
          if constexpr (is_matx_op<cuda::std::tuple_element_t<I,cuda::std::tuple<Ts...>>>()) {
            cuda::std::get<I>(ops_).PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
            PostRun<I+1, ShapeType, Executor>(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        } else if constexpr (I == sizeof...(Ts)-1) {
          if constexpr (is_matx_op<cuda::std::tuple_element_t<I,cuda::std::tuple<Ts...>>>()) {
            cuda::std::get<I>(ops_).PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
            // This was the last ops_ element, so stop recursion
          }
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        PostRun<0, ShapeType, Executor>(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
      }

      private:
      cuda::std::tuple<typename detail::base_type_t<Ts> ...> ops_;
      index_t size_;    
      int axis_;
    }; // end class ConcatOp
  } // end namespace detail

  /**
   * @brief ConcatOp multiple operators along a dimension
   * 
   * @tparam Dim dimension to concatenate
   * @tparam Ts operator types
   * @param axis axis to operate along
   * @param ts operators
   * @return concatenated operator 
   */
  template <typename... Ts>
    __MATX_INLINE__ __MATX_HOST__  auto concat(int axis, const Ts&... ts)
    {
      [[maybe_unused]] const auto first = detail::pp_get<0>(ts...);
      MATX_ASSERT_STR(axis <= first.Rank(),matxInvalidDim, "concat must take an axis less than the rank of the operators");

      return detail::ConcatOp<Ts...>{axis, ts...};
    }  
} // end namespace matx
