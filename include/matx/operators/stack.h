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
   * StackOp operators
   *
   * Class for stacking operators along a new dimension. Ranks and Sizes of the operators not
   * being stacked must be the same. 
   */
  namespace detail {  
    template <typename... Ts>
      class StackOp : public BaseOp<StackOp<Ts...>>
    {
      using first_type = cuda::std::tuple_element_t<0, cuda::std::tuple<Ts...>>;
      using first_value_type = typename first_type::value_type;
      using self_type = StackOp<Ts...>;

      static constexpr int RANK = first_type::Rank();

      public:
      using matxop = bool;
      using matxoplvalue = bool;

      // Scalar type of operation
      using value_type = first_value_type;

      template <int I = -1>
        __MATX_INLINE__ std::string get_str() const {
          if constexpr (I==-1) return "stack(" + get_str<I+1>();
          else if constexpr (I < sizeof...(Ts)-1) return cuda::std::get<I>(ops_).str() + "," + get_str<I+1>();
          else if constexpr (I == sizeof...(Ts)-1) return cuda::std::get<I>(ops_).str() + ")";
          else return "";
        }

      __MATX_INLINE__ std::string str() const {
        return get_str<-1>();
      }

      __MATX_INLINE__ StackOp(int axis, const Ts&... ts) : ops_(ts...), axis_(axis)
      {
        static_assert(sizeof...(Ts) > 1, "Must have more than one tensor to stack");
        static_assert((... && (RANK == Ts::Rank())), "stacked ops must have the same rank");

        for (int32_t i = 0; i < RANK; i++) {
          MATX_ASSERT_STR(((ts.Size(i) == pp_get<0>(ts).Size(i)) && ...)
              , matxInvalidSize, "stacked operators must have the same size");
        }
      }

      template <typename CapType, int I = 0, int N>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) GetVal(index_t oidx, cuda::std::array<index_t,RANK> &indices) const {

        if constexpr ( I == N ) {
          const auto &op = cuda::std::get<0>(ops_);
          return get_value<CapType>(op, indices);
        } else {
          if ( I < oidx ) {
            // this is not the correct operator, recurse
            return GetVal<CapType, I+1, N>(oidx, indices);
          } else {
            // this is the correct operator, return it's value
            auto &op = cuda::std::get<I>(ops_);
            return get_value<CapType>(op, indices);
          }
        }
      }

      template <typename CapType, int I = 0, int N>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) GetVal(index_t oidx, cuda::std::array<index_t,RANK> &indices) {

        if constexpr ( I == N ) {
          // This should never happen, but we return a fake value from the first tuple element anyways
          auto &op = cuda::std::get<0>(ops_);
          return get_value<CapType>(op, indices);
        } else {
          if ( I < oidx ) {
            // this is not the correct operator, recurse
            return GetVal<CapType, I+1, N>(oidx, indices);
          } else {
            // this is the correct operator, return it's value
            auto &op = cuda::std::get<I>(ops_);
            return get_value<CapType>(op, indices);
          }
        }
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
#ifdef __CUDA_ARCH__
        if constexpr (CapType::jit) {
          if ((threadIdx.x * CapType::ept) >= Size(Rank() - 1)) {
            return detail::GetJitSentinelValue<CapType, value_type>();
          }
        }
#endif
        if constexpr (CapType::ept == ElementsPerThread::ONE) {
          cuda::std::array<index_t, RANK + 1> indices = {{is...}};
          cuda::std::array<index_t, RANK> indices_o;

          // operator index
          index_t oidx = indices[axis_];

          // removing operator axis from indices
          for(int i = 0; i < axis_; i++) {
            indices_o[i] = indices[i];
          } 
          
          for(int i = axis_; i < (int)indices_o.size(); i++) {
            indices_o[i] = indices[i+1];
          }

          return GetVal<CapType, 0, sizeof...(Ts)>(oidx, indices_o);
        } else {
          return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
        }
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return this->operator()<DefaultCapabilities>(is...);
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
      {
        return cuda::std::as_const(*this).template operator()<CapType>(indices...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
      {
        return this->operator()<DefaultCapabilities>(indices...);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() noexcept
      {
        return RANK + 1;
      }

      constexpr index_t __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const noexcept
      {
        if(dim==axis_)
          return sizeof...(Ts);
        else if (dim < axis_) {
          return cuda::std::get<0>(ops_).Size(dim);
        } else {
          // remove axis_ dim from dim.
          return cuda::std::get<0>(ops_).Size(dim-1);
        }
      }

      ~StackOp() = default;
      StackOp(const StackOp &rhs) = default;
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

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] const InType& in) const {
        if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, get_combined_ops_capability<Cap>(ops_, in));
        } else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, get_combined_ops_capability<Cap>(ops_, in));
        }
      }

      private:
      cuda::std::tuple<typename detail::base_type_t<Ts> ...> ops_;
      index_t size_;    
      int axis_;

      template <OperatorCapability Cap, size_t I = 0>
      __MATX_INLINE__ __MATX_HOST__ auto get_combined_ops_capability(const cuda::std::tuple<typename detail::base_type_t<Ts>...>& ops) const {
        if constexpr (I == sizeof...(Ts)) {
          return capability_attributes<Cap>::default_value;
        } else {
          auto current_cap = detail::get_operator_capability<Cap>(cuda::std::get<I>(ops));
          auto rest_cap = get_combined_ops_capability<Cap, I + 1>(ops);
          return combine_capabilities<Cap>(current_cap, rest_cap);
        }
      }

      template <OperatorCapability Cap, typename InType, size_t I = 0>
      __MATX_INLINE__ __MATX_HOST__ auto get_combined_ops_capability(const cuda::std::tuple<typename detail::base_type_t<Ts>...>& ops, const InType& in) const {
        if constexpr (I == sizeof...(Ts)) {
          return capability_attributes<Cap>::default_value;
        } else {
          auto current_cap = detail::get_operator_capability<Cap>(cuda::std::get<I>(ops), in);
          auto rest_cap = get_combined_ops_capability<Cap, InType, I + 1>(ops, in);
          return combine_capabilities<Cap>(current_cap, rest_cap);
        }
      }
    }; // end class StackOp
  } // end namespace detail

  /**
   * @brief StackOp multiple operators along a dimension
   * 
   * @tparam Ts operator types
   * @param axis dimension to insert new dimension
   * @param ts operators
   * @return stacked operator 
   */
  template <typename... Ts>
    __MATX_INLINE__ __MATX_HOST__  auto stack(int axis, const Ts&... ts)
    {
      auto first = detail::pp_get<0>(ts...);

      MATX_ASSERT_STR(axis <= first.Rank(),matxInvalidDim, "stack must take an axis less than or equal to the the rank of the operators");
      return detail::StackOp<Ts...>{axis, ts...};
    }  
} // end namespace matx
