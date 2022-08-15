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

namespace matx
{
  /**
   * Make a deep copy of a view into another view
   *
   * Copies the data from a view into another view. Views should normally be
   * backed by different data objects, but it's not necessary if there is no
   * overlap between the soure and destination. If the source in destination
   * overlap in any way, it is a race condition and the result of the operation
   * is undefined.
   *
   * Both tensor views must be the same rank and size in every dimension
   *
   * @param out
   *   Tensor to copy into
   * @param in
   *   Tensor to copy from
   * @param stream
   *   CUDA stream to operate in
   */
  template <typename OutputTensor, typename InputTensor>
    __MATX_INLINE__ void copy(OutputTensor &out, const InputTensor &in,
        const cudaStream_t stream)
    {
      for (int i = 0; i < OutputTensor::Rank(); i++)
      {
        MATX_ASSERT(out.Size(i) == in.Size(i), matxInvalidSize);
      }

      (out = in).run(stream);
    };
} // end namespace matx
