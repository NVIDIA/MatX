////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#ifdef __CUDACC__

#include <cuda.h>

namespace matx {

// Kernel that performs SpMV for an m x n DIA-I matrix.
template <typename VAL, typename CRD>
__global__ void diai_spmv_kernel(VAL *A, CRD *diags, uint64_t numDiags, VAL *B,
                                 VAL *C, uint64_t m, uint64_t n) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    VAL acc = 0.0;
    for (uint64_t d = 0; d < numDiags; d++) {
      int64_t j = i + diags[d]; // signed
      if (0 <= j && j < static_cast<int64_t>(n)) {
        acc += A[d * m + i] * B[j];
      }
    }
    C[i] = acc;
  }
}

// Kernel that performs SpMV for an m x n DIA-J matrix.
template <typename VAL, typename CRD>
__global__ void diaj_spmv_kernel(VAL *A, CRD *diags, uint64_t numDiags, VAL *B,
                                 VAL *C, uint64_t m, uint64_t n) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    VAL acc = 0.0;
    for (uint64_t d = 0; d < numDiags; d++) {
      int64_t j = i + diags[d]; // signed
      if (0 <= j && j < static_cast<int64_t>(n)) {
        acc += A[d * n + j] * B[j];
      }
    }
    C[i] = acc;
  }
}

} // namespace matx

#endif
