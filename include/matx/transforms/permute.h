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
  /**
   * Permute a tensor view out-of-place
   *
   * Rearranges the dimensions of a tensor view without touching the data. This is
   * accomplished by changing the strides between dimensions to reflect the new
   * transposed order. This function can result in very in efficient memory
   * accesses, so it's recommended only to use in places performance is not
   * critical.
   *
   * Both tensor views must be the same rank, and the dimensions that moved must
   * match their original size
   *
   * @param out
   *   Tensor to copy into
   * @param in
   *   Tensor to copy from
   * @param dims
   *   Order of transposed tensor dimensions
   * @param stream
   *   CUDA stream to operate in
   *
   */
  template <class T, int Rank>
    __MATX_INLINE__ void permute(detail::tensor_impl_t<T, Rank> &out, const detail::tensor_impl_t<T, Rank> &in,
        const std::initializer_list<uint32_t> &dims,
        const cudaStream_t stream)
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
      
      // This is very naive, we should make optimized versions for various swizzles
      auto in_t = in.Permute(dims.begin());

      copy(out, in_t, stream);
    };
} // end namespace matx
