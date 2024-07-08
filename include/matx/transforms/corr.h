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

#include "matx/transforms/conv.h"
#include "matx/operators/reverse.h"
#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"

namespace matx {

/**
 * @brief Correlate two input operators
 * 
 * @tparam OutputTensor Output tensor type
 * @tparam In1Type First input operator type
 * @tparam In2Type Second input operator type
 * @param o Output tensor
 * @param i1 First input operator
 * @param i2 Second input operator
 * @param mode Mode of correlation
 * @param method Method for correlation
 * @param exec Executor
 */
template <typename OutputTensor, typename In1Type, typename In2Type, typename Executor>
__MATX_INLINE__ void corr_impl(OutputTensor o, const In1Type i1, const In2Type i2,
          matxConvCorrMode_t mode, matxConvCorrMethod_t method,
          const Executor &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto i2r = reverse<In2Type::Rank()-1>(conj(i2));
  conv1d_impl(o, i1, i2r, mode, method, exec);
}



} // end namespace matx
