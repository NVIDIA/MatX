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


#include "matx/core/nvtx.h"
#include "matx/core/type_utils.h"

namespace matx
{
  namespace detail {
    // inv()
    template <typename T, typename = void> struct is_inv_xform_op : std::false_type {};
    template <typename T> struct is_inv_xform_op<T, std::void_t<typename T::inv_xform_op>> : std::true_type {};
    template <typename T> inline constexpr bool is_inv_xform_op_v = is_inv_xform_op<typename remove_cvref<T>::type>::value;
  
  template <typename Op, typename Ex>
  __MATX_INLINE__ constexpr void CheckExecutor() {
    if constexpr (is_inv_xform_op_v<Op>) {
      static_assert(is_device_executor_v<Ex>, "Inverse only supports the CUDA executor currently");
    }
  }


  /**
   * @brief Dispatch a simple transform to the proper transform function
   * 
   * @tparam TensorType Type of output tensor
   * @tparam Op Input (transform) operator type
   * @tparam Ex Executor type
   * @param t Output tensor
   * @param op Input operator
   */
  template <typename TensorType, typename Op, typename Ex>
  __MATX_INLINE__ void transform_dispatch(TensorType &t, const Op &op, Ex &&ex) {
    CheckExecutor<Op, Ex>();

    if constexpr (is_inv_xform_op_v<Op>) {
      inv_impl(t, op.GetOp(), ex.getStream());
    }
    else {
      static_assert("Unknown transform operator when dispatching");
    }
  }
  }
} // end namespace matx
