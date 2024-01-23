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

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"

namespace matx {

enum class NormOrder {
  NONE,
  L1,
  L2,
  FROB
};

namespace detail {
  struct NormTypeVector{};
  struct NormTypeMatrix{};
};


template <typename NormType, typename OutputOp, typename InputOp, typename Executor>
__MATX_INLINE__ void norm_impl(OutputOp out, const InputOp &in,
          NormOrder order, Executor &&exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  if constexpr (std::is_same_v<NormType, detail::NormTypeVector>) {
    if (order == NormOrder::NONE || order == NormOrder::L2) {
      (out = sqrt(sum(abs2(in), {InputOp::Rank() - 1}))).run(exec);
    }
    else if (order == NormOrder::L1) {
      (out = sum(abs(in), {InputOp::Rank() - 1})).run(exec);
    }
    else {
      MATX_ASSERT_STR(false, matxInvalidParameter, "Invalid order type for vector norm");
    }
  }
  else {
    if (order == NormOrder::NONE || order == NormOrder::FROB) {
      (out = sqrt(sum(abs2(in), {InputOp::Rank() - 2, InputOp::Rank() - 1}))).run(exec);
    }
    else if (order == NormOrder::L1) {
      (out = max(sum(abs(in), {InputOp::Rank() - 2}), {InputOp::Rank() - 2})).run(exec);
    }
    else {
      MATX_ASSERT_STR(false, matxInvalidParameter, "Invalid order type for matrix norm");
    }
  }
}



} // end namespace matx
