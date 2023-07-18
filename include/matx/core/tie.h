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

#include <tuple>
#include "matx/core/type_utils.h"

namespace matx {

/* mtie exists because we can't overload std::tuple's operator= for operator assignments in
   a multiple return set statement. */
template <typename... Ts>
struct mtie : public BaseOp<mtie<Ts...>>{
  using mtie_type = bool;
  using scalar_type = void; // Doesn't matter since it's not used
  using value_type  = void; // Doesn't matter since it's not used
  using shape_type  = index_t;
  using matxoplvalue = bool;

  static_assert(sizeof...(Ts) > 1, "Only use mtie() when number of elements is greater than 1");

  mtie(Ts... ts) : ts_(ts...) {}

  // operator= to cover multiple return types using mtie in a run statement
  template <typename RHS, std::enable_if_t<is_matx_transform_op<RHS>(), bool> = true>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(RHS &&rhs)
  {
    return mapply([&](auto... args) {
      return mtie<Ts..., RHS>{args..., rhs};
    }, ts_);
  }

  template <int n>
  auto get() {
    return std::get<n>(ts_);
  }

  static __MATX_INLINE__ constexpr int32_t Rank()
  {
    return 1;
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size([[maybe_unused]] int dim) const noexcept
  {
    return 0;
  }  

  constexpr __MATX_INLINE__ auto Elements() const noexcept {
    return std::tuple_size(ts_);
  }

  constexpr __MATX_INLINE__ auto IsSingleAssign() const noexcept { 
    return sizeof...(Ts) == 2;
  }

  template <typename Executor>
  __MATX_INLINE__ void Exec(Executor &&ex) {
    std::get<sizeof...(Ts) - 1>(ts_).Exec(ts_, std::forward<Executor>(ex));
  }

  std::tuple<Ts...> ts_;
};

// /* mtie exists because we can't overload std::tuple's operator= for operator assignments in
//    a multiple return set statement. */
// template <typename TieType, typename XformOp>
// struct mtie_container {
//   using mtie_type = bool;
//   using scalar_type = void; // Doesn't matter since it's not used
//   using value_type  = void; // Doesn't matter since it's not used
//   using shape_type  = index_t;
//   using matxoplvalue = bool;

//   mtie_container(TieType &&t, XformOp &&xop) : tie_(t), xop_(xop) {}


//   template <int n>
//   auto get() {
//     return std::get<n>(ts_);
//   }

//   static __MATX_INLINE__ constexpr int32_t Rank()
//   {
//     return 1;
//   }

//   constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size([[maybe_unused]] int dim) const noexcept
//   {
//     return 0;
//   }  

//   constexpr __MATX_INLINE__ auto Elements() const noexcept {
//     return std::tuple_size(ts_);
//   }

//   constexpr __MATX_INLINE__ auto IsSingleAssign() const noexcept { 
//     return sizeof...(Ts) == 2;
//   }

//   template <typename Executor>
//   __MATX_INLINE__ void Exec(Executor &&ex) {
//     if constexpr (sizeof...(Ts) == 2) {
//       std::get<1>(ts_).Exec(std::get<0>(ts_), std::forward<Executor>(ex));
//     }
//     else {
//       // When we're not doing a direct assignment (ie (A = xform(B)) the last element of
//       // our tuple is the transform operator and the first N elements are the output parameters
//       // for the transform.
//       mapply([&](auto... args) {
//         std::get<sizeof...(Ts) - 1>(ts_).Exec(args..., std::forward<Executor>(ex));
//       }, ts_);
//     }
//   }

//   TieType tie_;
//   XformOp xop_;
// };

}; // namespace matx
