////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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
   * ZipVecOp operator
   *
   * Class for zipping operators into a vectorized output operator. The rank and
   * dimensions of the input operators must all be the same and the output operator
   * will have the same rank and dimensions.
   */
  namespace detail {
    template <typename... Ts>
      class ZipVecOp : public BaseOp<ZipVecOp<Ts...>>
    {
      using first_type = cuda::std::tuple_element_t<0, cuda::std::tuple<Ts...>>;
      using self_type = ZipVecOp<Ts...>;

      static constexpr int RANK = first_type::Rank();

      public:
      using matxop = bool;
      using matxoplvalue = bool;

      using value_type = AggregateToVecType<typename Ts::value_type...>;

      template <int I = -1>
      __MATX_INLINE__ std::string get_str() const {
        if constexpr (I==-1) return "zipvec(" + get_str<I+1>();
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
        return std::format("JITZipVec_num{}_{}", sizeof...(Ts), get_sizes_str<0>());
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

      template <int I = 0>
      __MATX_INLINE__ std::string get_jit_value_types() const {
        if constexpr (I < sizeof...(Ts)) {
          std::string type_str = "typename T" + std::to_string(I) + "::value_type";
          if constexpr (I < sizeof...(Ts) - 1) {
            return type_str + ", " + get_jit_value_types<I+1>();
          } else {
            return type_str;
          }
        } else {
          return "";
        }
      }

      template <int I = 0>
      __MATX_INLINE__ std::string get_jit_operator_calls() const {
        if constexpr (I < sizeof...(Ts)) {
          std::string call = "static_cast<scalar_type>(cuda::std::get<" + std::to_string(I) + ">(ops_).template operator()<CapType>(cuda::std::forward<Is>(is)...))";
          if constexpr (I < sizeof...(Ts) - 1) {
            return call + ", " + get_jit_operator_calls<I+1>();
          } else {
            return call;
          }
        } else {
          return "";
        }
      }

      __MATX_INLINE__ auto get_jit_op_str() const {
        std::string func_name = get_jit_class_name();
        cuda::std::array<index_t, RANK> out_dims_;
        for (int i = 0; i < RANK; i++) {
          out_dims_[i] = Size(i);
        }
        
        std::string value_types = get_jit_value_types<0>();
        
        return cuda::std::make_tuple(
          func_name,
          std::format("template <{}> struct {} {{\n"
              "  using value_type = AggregateToVecType<{}>;\n"
              "  using matxop = bool;\n"
              "  constexpr static int RANK_ = {};\n"
              "  constexpr static cuda::std::array<index_t, RANK_> sizes_ = {{ {} }};\n"
              "  {}"
              "  // Const operator()\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... is) const {{\n"
              "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
              "      using scalar_type = typename AggregateToVec<{}>::common_type;\n"
              "      return value_type{{ {} }};\n"
              "    }} else {{\n"
              "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
              "    }}\n"
              "  }}\n"
              "  // Non-const operator()\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... is) {{\n"
              "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
              "      using scalar_type = typename AggregateToVec<{}>::common_type;\n"
              "      return value_type{{ {} }};\n"
              "    }} else {{\n"
              "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
              "    }}\n"
              "  }}\n"
              "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return RANK_; }}\n"
              "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{\n"
              "    return sizes_[dim];\n"
              "  }}\n"
              "}};\n",
              get_jit_type_list<0>(), func_name, value_types, RANK, detail::array_to_string(out_dims_), get_jit_storage_tuple(),
              value_types, get_jit_operator_calls<0>(), value_types, get_jit_operator_calls<0>())
        );
      }
#endif

      __MATX_INLINE__ ZipVecOp(const Ts&... ts) : ops_(ts...)
      {
        MATX_LOG_TRACE("{} constructor: num_ops={}, rank={}", str(), sizeof...(Ts), Rank());
        static_assert(sizeof...(Ts) > 0 && sizeof...(Ts) <= 4, "Must have between 1 and 4 operators for zipvec");
        static_assert((... && (RANK == Ts::Rank())), "zipped ops must have the same rank");
        // All ops must have the same scalar value type; that is enforced by AggregateToVecType

        for (int32_t i = 0; i < RANK; i++) {
            MATX_ASSERT_STR(((ts.Size(i) == pp_get<0>(ts).Size(i)) && ...), 
                matxInvalidSize, "zipped operators must have the same size in all dimensions");
        }
      }

      template <typename CapType, typename Ops, typename... Is>
      static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Ops&& ops, Is... is) {
        if constexpr (CapType::ept == ElementsPerThread::ONE) {
          return cuda::std::apply([&](auto&&... op) {
            using scalar_type = typename AggregateToVec<typename Ts::value_type...>::common_type;
            return value_type{ static_cast<scalar_type>(op.template operator()<CapType>(cuda::std::forward<Is>(is)...))... };
          }, cuda::std::forward<Ops>(ops));
        } else {
          return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
        }
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return get_impl<CapType>(cuda::std::as_const(ops_), cuda::std::forward<Is>(is)...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return get_impl<DefaultCapabilities>(cuda::std::as_const(ops_), cuda::std::forward<Is>(is)...);
      }

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return get_impl<CapType>(ops_, cuda::std::forward<Is>(is)...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return get_impl<DefaultCapabilities>(ops_, cuda::std::forward<Is>(is)...);
      }

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
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
          // For now, we do not support vectorization. We could support it, but it will require some
          // rework of the assumptions used in the matx::Vector class.
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, get_combined_ops_capability<Cap>(in, ops_));
        } else {
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

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() noexcept
      {
        return RANK;
      }

      constexpr index_t __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const noexcept
      {
        // All ops must have the same size in all dimensions
        return cuda::std::get<0>(ops_).Size(dim);
      }

      ~ZipVecOp() = default;
      ZipVecOp(const ZipVecOp &rhs) = default;

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
    }; // end class ZipVecOp
  } // end namespace detail

  /**
   * @brief zipvec zips multiple operators together into a vectorized operator. This allows combining multiple operators
   * that represent scalar types into an operator with vectorized types. For example, two operators of type float can
   * be combined into an operator of type float2.
   *
   * The input operators must have the same rank and size in all dimensions and the types must be compatible. This
   * is only supported for the types for which CUDA has corresponding vector types, including char, short,
   * int, long, float, and double. The integer types also support unsigned variants (uchar, ushort, etc.).
   * For these sizes, the number of input operators and the corresponding zipped vector length can be 1-4.
   *
   * The components from the input operators are accessed by the fields x, y, z, and w, respectively, in the zipped operator.
   *
   * @tparam Ts input operator types
   * @param ts input operators
   * @return zipped operator
   */
  template <typename... Ts>
  __MATX_INLINE__ __MATX_HOST__  auto zipvec(const Ts&... ts)
  {
    static_assert(sizeof...(Ts) > 0, "zipvec must take at least one operator");

    return detail::ZipVecOp<Ts...>{ts...};
  }
} // end namespace matx
