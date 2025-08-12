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
   * ZipVecOp operator
   *
   * Class for concatening operators along a single dimension. Sizes of the operators not
   * being concatenated must be the same, and the new operator has dimensions equal to the original
   * operator on non-index dimension, and the sum of sizes along the index dimension.
   */
  namespace detail {
    template <typename... Ts>
      class ZipVecOp : public BaseOp<ZipVecOp<Ts...>>
    {
      using first_type = cuda::std::tuple_element_t<0, cuda::std::tuple<Ts...>>;
      using scalar_value_type = typename first_type::value_type;
      using self_type = ZipVecOp<Ts...>;

      static constexpr int RANK = first_type::Rank();

      public:
      using matxop = bool;
      using matxoplvalue = bool;

      // Scalar type of operation
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

      __MATX_INLINE__ ZipVecOp(const Ts&... ts) : ops_(ts...)
      {
        static_assert(sizeof...(Ts) > 0 && sizeof...(Ts) <= 4, "Must have between 1 and 4 tensors for zipvec");
        static_assert((... && (RANK == Ts::Rank())), "zipped ops must have the same rank");
        // All ops must have the same scalar value type; that is enforced by AggregateToVecType

        for (int32_t i = 0; i < RANK; i++) {
            MATX_ASSERT_STR(((ts.Size(i) == pp_get<0>(ts).Size(i)) && ...), 
                matxInvalidSize, "zipped operators must have the same size in all dimensions");
        }
      }

      template <ElementsPerThread EPT, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return cuda::std::apply([&](auto&&... ops) { 
          using scalar_type = typename AggregateToVec<typename Ts::value_type...>::common_type;
          return value_type{ static_cast<scalar_type>(ops(std::forward<Is>(is)...))... }; 
        }, ops_);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is) const
      {
        return cuda::std::apply([&](auto&&... ops) { 
          using scalar_type = typename AggregateToVec<typename Ts::value_type...>::common_type;
          return value_type{ static_cast<scalar_type>(ops(std::forward<Is>(is)...))... }; 
        }, ops_);
      }

      template <ElementsPerThread EPT, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return cuda::std::apply([&](auto&&... ops) { 
          using scalar_type = typename AggregateToVec<typename Ts::value_type...>::common_type;
          return value_type{ static_cast<scalar_type>(ops(std::forward<Is>(is)...))... }; 
        }, ops_);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... is)
      {
        return cuda::std::apply([&](auto&&... ops) { 
          using scalar_type = typename AggregateToVec<typename Ts::value_type...>::common_type;
          return value_type{ static_cast<scalar_type>(ops(std::forward<Is>(is)...))... }; 
        }, ops_);
      }

      template <OperatorCapability Cap>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
        if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          // For now, we do not support vectorization. We could support it, but it will require some
          // rework of the assumptions used in the matx::Vector class.
          return ElementsPerThread::ONE;
        } else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return self_has_cap;
        }
      }

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
    }; // end class ZipVecOp
  } // end namespace detail

  /**
   * @brief zipvec zips multiple operators together into a vectorized operator. This allows combining multiple operators
   * that represent scalar types into an operator with vectorized types. For example, two operators of type float can
   * be combined into an operator of type float2.
   *
   * The input operators must have the same rank and size in all dimensions and the types must be compatible. This
   * is only supported for the types for which CUDA has corresponding vector types, including [u]char, [u]short,
   * [u]int, [u]long, float, and double. For these sizes, the number of input operators and the corresponding zipped
   * vector length can be 1-4. __half types are also supported, but only for a vector length of 2.
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
