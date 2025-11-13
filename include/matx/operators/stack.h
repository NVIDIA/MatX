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
#include "matx/core/utils.h"
#include "matx/operators/base_operator.h"
#include <format>

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

#ifdef MATX_EN_JIT
      struct JIT_Storage {
        cuda::std::tuple<typename detail::inner_storage_or_self_t<detail::base_type_t<Ts>>...> ops_;
      };

      JIT_Storage ToJITStorage() const {
        return JIT_Storage{cuda::std::apply([](const auto&... ops) {
          return cuda::std::make_tuple(detail::to_jit_storage(ops)...);
        }, ops_)};
      }

      template <int I = 0>
      __MATX_INLINE__ std::string get_sizes_str() const {
        if constexpr (I < sizeof...(Ts)) {
          const auto& op = cuda::std::get<I>(ops_);
          std::string sizes = "op" + std::to_string(I) + "_";
          for (int d = 0; d < RANK; d++) {
            sizes += std::to_string(op.Size(d));
            if (d < RANK - 1) sizes += "x";
          }
          if constexpr (I < sizeof...(Ts) - 1) {
            return sizes + "_" + get_sizes_str<I+1>();
          } else {
            return sizes;
          }
        } else {
          return "";
        }
      }

      __MATX_INLINE__ std::string get_jit_class_name() const {
        return std::format("JITStack_axis{}_num{}_{}", axis_, sizeof...(Ts), get_sizes_str<0>());
      }

      template <int I = 0>
      __MATX_INLINE__ std::string get_jit_type_list() const {
        if constexpr (I < sizeof...(Ts) - 1) {
          return "typename T" + std::to_string(I) + ", " + get_jit_type_list<I+1>();
        } else if constexpr (I == sizeof...(Ts) - 1) {
          return "typename T" + std::to_string(I);
        } else {
          return "";
        }
      }

      template <int I = 0>
      __MATX_INLINE__ std::string get_jit_storage_tuple_types() const {
        if constexpr (I < sizeof...(Ts) - 1) {
          return "typename detail::inner_storage_or_self_t<detail::base_type_t<T" + std::to_string(I) + ">>, " + get_jit_storage_tuple_types<I+1>();
        } else if constexpr (I == sizeof...(Ts) - 1) {
          return "typename detail::inner_storage_or_self_t<detail::base_type_t<T" + std::to_string(I) + ">>";
        } else {
          return "";
        }
      }

      __MATX_INLINE__ std::string get_jit_storage_tuple() const {
        return "cuda::std::tuple<" + get_jit_storage_tuple_types<0>() + "> ops_;\n";
      }

      __MATX_INLINE__ auto get_jit_op_str() const {
        std::string func_name = get_jit_class_name();
        cuda::std::array<index_t, RANK + 1> out_dims_;
        for (int i = 0; i < RANK + 1; i++) {
          out_dims_[i] = Size(i);
        }
        
        return cuda::std::make_tuple(
          func_name,
          std::format("template <{}> struct {} {{\n"
              "  using value_type = typename T0::value_type;\n"
              "  using matxop = bool;\n"
              "  constexpr static int RANK_ = {};\n"
              "  constexpr static cuda::std::array<index_t, RANK_+1> sizes_ = {{ {} }};\n"
              "  constexpr static int axis_ = {};\n"
              "  {}"
              "  // Const GetVal\n"
              "  template <typename CapType, int I, int N>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ auto GetVal(index_t oidx, cuda::std::array<index_t, RANK_>& indices) const {{\n"
              "    if constexpr ( I == N ) {{\n"
              "      const auto &op = cuda::std::get<0>(ops_);\n"
              "      return get_value<CapType>(op, indices);\n"
              "    }} else {{\n"
              "      if ( I < oidx ) {{\n"
              "        return GetVal<CapType, I+1, N>(oidx, indices);\n"
              "      }} else {{\n"
              "        const auto &op = cuda::std::get<I>(ops_);\n"
              "        return get_value<CapType>(op, indices);\n"
              "      }}\n"
              "    }}\n"
              "  }}\n"
              "  // Non-const GetVal for lvalue assignments\n"
              "  template <typename CapType, int I, int N>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) GetVal(index_t oidx, cuda::std::array<index_t, RANK_>& indices) {{\n"
              "    if constexpr ( I == N ) {{\n"
              "      auto &op = cuda::std::get<0>(ops_);\n"
              "      return get_value<CapType>(op, indices);\n"
              "    }} else {{\n"
              "      if ( I < oidx ) {{\n"
              "        return GetVal<CapType, I+1, N>(oidx, indices);\n"
              "      }} else {{\n"
              "        auto &op = cuda::std::get<I>(ops_);\n"
              "        return get_value<CapType>(op, indices);\n"
              "      }}\n"
              "    }}\n"
              "  }}\n"
              "  // Const operator()\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... is) const {{\n"
              "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
              "      cuda::std::array<index_t, RANK_+1> indices{{is...}};\n"
              "      cuda::std::array<index_t, RANK_> indices_o;\n"
              "      index_t oidx = indices[axis_];\n"
              "      for(int i = 0; i < axis_; i++) {{\n"
              "        indices_o[i] = indices[i];\n"
              "      }}\n"
              "      for(int i = axis_; i < (int)indices_o.size(); i++) {{\n"
              "        indices_o[i] = indices[i+1];\n"
              "      }}\n"
              "      return GetVal<CapType, 0, {}>(oidx, indices_o);\n"
              "    }} else {{\n"
              "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
              "    }}\n"
              "  }}\n"
              "  // Non-const operator()\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... is) {{\n"
              "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
              "      cuda::std::array<index_t, RANK_+1> indices{{is...}};\n"
              "      cuda::std::array<index_t, RANK_> indices_o;\n"
              "      index_t oidx = indices[axis_];\n"
              "      for(int i = 0; i < axis_; i++) {{\n"
              "        indices_o[i] = indices[i];\n"
              "      }}\n"
              "      for(int i = axis_; i < (int)indices_o.size(); i++) {{\n"
              "        indices_o[i] = indices[i+1];\n"
              "      }}\n"
              "      return GetVal<CapType, 0, {}>(oidx, indices_o);\n"
              "    }} else {{\n"
              "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
              "    }}\n"
              "  }}\n"
              "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return RANK_+1; }}\n"
              "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{\n"
              "    return sizes_[dim];\n"
              "  }}\n"
              "}};\n",
              get_jit_type_list<0>(), func_name, RANK, detail::array_to_string(out_dims_), axis_, get_jit_storage_tuple(), sizeof...(Ts), sizeof...(Ts))
        );
      }
#endif

      __MATX_INLINE__ StackOp(int axis, const Ts&... ts) : ops_(ts...), axis_(axis)
      {
        MATX_LOG_TRACE("{} constructor: axis={}, num_tensors={}", str(), axis, sizeof...(Ts));
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
        return set(*this, rhs); 
      }

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
          return get_jit_class_name() + "<" + get_jit_type_params<0>() + ">";
#else
          return "";
#endif
        }
        else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
          return combine_capabilities<Cap>(true, get_combined_ops_capability<Cap>(in, ops_));
#else
          return false;
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
          cuda::std::apply([&in](const auto&... ops) {
            (detail::get_operator_capability<Cap>(ops, in), ...);
          }, ops_);
          
          return true;
#else
          return false;
#endif
        }
        else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, get_combined_ops_capability<Cap>(in, ops_));
        } 
        else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
          auto in_copy = in;
          in_copy.permutes_input_output = true;
          return combine_capabilities<Cap>(detail::get_combined_ops_capability<Cap>(in_copy, ops_));
        }
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, get_combined_ops_capability<Cap>(in, ops_));
        }
      }

#ifdef MATX_EN_JIT
      template <int I = 0>
      __MATX_INLINE__ std::string get_jit_type_params() const {
        if constexpr (I < sizeof...(Ts)) {
          VoidCapabilityType void_type{};
          auto type_name = detail::get_operator_capability<OperatorCapability::JIT_TYPE_QUERY>(cuda::std::get<I>(ops_), void_type);
          if constexpr (I < sizeof...(Ts) - 1) {
            return type_name + "," + get_jit_type_params<I+1>();
          } else {
            return type_name;
          }
        } else {
          return "";
        }
      }
#endif

      private:
      cuda::std::tuple<typename detail::base_type_t<Ts> ...> ops_;
      index_t size_;    
      int axis_;
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
