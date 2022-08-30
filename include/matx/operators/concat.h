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
   * Concatenate operators
   *
   * Class for concatening operators along a single dimension. Sizes of the operators not
   * being concatenated must be the same, and the new operator has dimensions equal to the original
   * operator on non-index dimension, and the sum of sizes along the index dimension.
   */
  namespace detail {  
    template <int Dim, typename... Ts>
      class Concatenate : public BaseOp<Concatenate<Dim, Ts...>>
    {
      using first_type = std::tuple_element_t<0, std::tuple<Ts...>>;
      using first_value_type = typename first_type::value_type;
      static constexpr int RANK = first_type::Rank();

      public:
      // Scalar type of operation
      using scalar_type = first_value_type;

      __MATX_INLINE__ Concatenate(Ts... ts) : ops_(ts...)
      {
        static_assert(RANK > 0, "Cannot concatenate rank-0 tensors");
        static_assert(sizeof...(Ts) > 0, "Must have more than one tensor to concatenate");
        static_assert((... && (RANK == ts.Rank())));

        auto tsum = [&](int d, auto ...args){ return (args.Size(d) + ...); };
        for (int32_t i = 0; i < RANK; i++) {
          size_[i] = (i == Dim) ? tsum(i, ts...) : pp_get<0>(ts...).Size(i);
        }
      }  

      // Base case. Cannot be reached
      template <size_t I = 0, typename... Is, std::enable_if_t<I == sizeof...(Ts), bool> = true>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto GetVal(cuda::std::tuple<Is...> tup) const {
          return static_cast<first_value_type>(0);
        }

      /* Check if the value of the index we're concatenating is smaller the size of the current
         operator size in that dimension. If so, we're on the correct operator and just return
         operator() from it. Otherwise we recursively call the same function moving to another 
         operator with a smaller index. */
      template <size_t I = 0, typename... Is, std::enable_if_t<I < sizeof...(Ts), bool> = true>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto GetVal(cuda::std::tuple<Is...> tup) const
        {
          if (cuda::std::get<Dim>(tup) < cuda::std::get<I>(ops_).Size(Dim)) {
            return mapply([&](auto &&...args) -> first_value_type {
                return cuda::std::get<I>(ops_).operator()(args...);
                }, tup);
          }

          cuda::std::get<Dim>(tup) -= cuda::std::get<I>(ops_).Size(Dim);
          return static_cast<first_value_type>(GetVal<I + 1, Is...>(tup));
        }    

      template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... is) const
        {
          return static_cast<first_value_type>(GetVal<0, Is...>(cuda::std::make_tuple(is...)));
        }


      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() noexcept
      {
        return RANK;
      }

      constexpr index_t __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const noexcept
      {
        return size_[dim];
      }

      private:
      cuda::std::tuple<Ts...> ops_;
      std::array<index_t, RANK> size_;    
    };
  }

  /**
   * @brief Concatenate multiple operators along a dimension
   * 
   * @tparam Dim dimension to concatenate
   * @tparam Ts operator types
   * @param ts operators
   * @return concatenated operator 
   */
  template <int Dim, typename... Ts>
    __MATX_INLINE__ __MATX_HOST__  auto concat(Ts... ts)
    {
      static_assert(((Dim < ts.Rank()) && ...), "Concatenation dimension larger than tensor rank");
      return detail::Concatenate<Dim, Ts...>{ts...};
    }  
} // end namespace matx
