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
        static_assert((... && (RANK == Ts::Rank())), "concatenated ops must have the same rank");

        for (int32_t i = 0; i < RANK; i++) {
          if(i == axis_) {
            size_ = (ts.Size(i) + ...);
          } else {
            MATX_ASSERT_STR(((ts.Size(i) == pp_get<0>(ts).Size(i)) && ...)
                , matxInvalidSize, "concatenate operators must have the same size in non-axis dimension");
          }
        }
      }


      // Non-const path returns references where available (used for LHS writes)
      template <typename CapType, int I = 0, int N>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(cuda::std::array<index_t,RANK> &indices) {
        if constexpr ( I == N ) {
          // This should never happen, but we return a fake value from the first tuple element anyways
          auto &op = cuda::std::get<0>(ops_);
          return cuda::std::apply([&](auto &&...call_args) -> decltype(auto) { return op.template operator()<CapType>(call_args...); }, indices);
        } else {
          auto &op = cuda::std::get<I>(ops_);
          auto idx = indices[axis_];
          auto size = op.Size(axis_);
          // If in range of this operator
          if(idx < size) {
            // evaluate operator
            return cuda::std::apply([&](auto &&...call_args) -> decltype(auto) { return op.template operator()<CapType>(call_args...); }, indices);
          } else {
            // otherwise remove this operator and recurse
            indices[axis_] -= size;
            return get_impl<CapType, I+1, N>(indices);
          }
        }
      }

      // Const path: unify scalar return type to value_type to avoid ref/value conflicts
      template <typename CapType, int I = 0, int N>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_impl(cuda::std::array<index_t,RANK> &indices) const {
        using return_t = cuda::std::conditional_t<
            (CapType::ept == ElementsPerThread::ONE),
            value_type,
            Vector<value_type, static_cast<index_t>(CapType::ept)>>;
        if constexpr ( I == N ) {
          const auto &op = cuda::std::get<0>(ops_);
          return cuda::std::apply([&](auto &&...call_args) -> return_t {
            return op.template operator()<CapType>(call_args...);
          }, indices);
        } else {
          const auto &op = cuda::std::get<I>(ops_);
          auto idx = indices[axis_];
          auto size = op.Size(axis_);
          if(idx < size) {
            return cuda::std::apply([&](auto &&...call_args) -> return_t {
              return op.template operator()<CapType>(call_args...);
            }, indices);
          } else {
            indices[axis_] -= size;
            return get_impl<CapType, I+1, N>(indices);
          }
        }
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        if constexpr (CapType::ept == ElementsPerThread::ONE) {
          cuda::std::array<index_t, sizeof...(Is)> indices = {{is...}};
          return get_impl<CapType, 0, sizeof...(Ts)>(indices);
        }
        else {
          return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
        }
      }

      

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return this->operator()<DefaultCapabilities>(is...);
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        if constexpr (CapType::ept == ElementsPerThread::ONE) {
          cuda::std::array<index_t, sizeof...(Is)> indices = {{is...}};
          return get_impl<CapType, 0, sizeof...(Ts)>(indices);
        }
        else {
          return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
        }
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return this->operator()<DefaultCapabilities>(is...);
      }


      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType & in) const {
        if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, get_combined_ops_capability<Cap>(in, ops_));
        } else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          // static_assert(sizeof...(Ts) > 1, ...); ensures ops_ is not empty.
          return combine_capabilities<Cap>(self_has_cap, get_combined_ops_capability<Cap>(in, ops_));
        }
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
        return set(*this, rhs);
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
