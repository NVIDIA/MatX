////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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

#include "matx/operators/constval.h"

namespace matx
{
  /**
   * Return value for all elements
   *
   * Creates an operator that returns the same value at every index. It can be
   * used in place of memset to fill all elements with a chosen value, or as
   * a zero-storage stand-in for a scalar in operator expressions and kernel
   * arguments that expect a MatX op.
   *
   * @tparam T
   *   Data type. Has no default; it is either deduced from `value` or
   *   specified explicitly. Omitting a default forces deduction so that
   *   `fill(shape, 3.14f)` produces a float-valued operator rather than
   *   collapsing the value to an unrelated default type.
   *
   * @param s
   *   Shape of tensor
   * @param value
   *   Value to fill
   */
  template <typename T, typename ShapeType>
    requires (!cuda::std::is_array_v<remove_cvref_t<ShapeType>>)
  inline auto fill(ShapeType &&s, T value)
  {
    return detail::ConstVal<T, ShapeType>(std::forward<ShapeType>(s), value);
  }

  /**
   * Return value for all elements
   *
   * Array-shape overload. See the ShapeType-taking overload for details.
   *
   * @tparam T
   *   Data type
   *
   * @param s
   *   Shape of tensor
   * @param value
   *   Value to fill
   */
  template <typename T, int RANK>
  inline auto fill(const index_t (&s)[RANK], T value)
  {
    return fill<T>(detail::to_array(s), value);
  }

  /**
   * Return value for all elements
   *
   * Rank-0 overload. `fill<T>({}, value)` produces a rank-0 ConstVal whose
   * single element is `value`. Mirrors the empty-brace `make_tensor<T>({})`
   * pattern.
   *
   * @tparam T
   *   Data type
   *
   * @param s
   *   Empty initializer list `{}`; unused, present to enable braced-init
   *   overload resolution for the rank-0 case.
   * @param value
   *   Value to fill
   */
  template <typename T>
  inline auto fill([[maybe_unused]] const std::initializer_list<detail::no_size_t> s, T value)
  {
    using shape_t = cuda::std::array<index_t, 0>;
    return fill<T, shape_t>(shape_t{}, value);
  }

  /**
   * Return value for all elements
   *
   * Shapeless form. The resulting operator has Rank() == matxNoRank and can
   * be used in contexts where the shape can be deduced (e.g. broadcast in an
   * operator expression).
   *
   * @tparam T
   *   Data type
   *
   * @param value
   *   Value to fill
   */
  template <typename T>
  inline auto fill(T value)
  {
    return fill<T, detail::NoShape>(detail::NoShape{}, value);
  }

} // end namespace matx
