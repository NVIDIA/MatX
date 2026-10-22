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

#include <cublas_v2.h>
#include <cusparse.h>

namespace matx::detail {

template <typename T>
constexpr cublasComputeType_t MatXTypeToCudaComputeType()
{
  if constexpr (cuda::std::is_same_v<T, cuda::std::complex<float>> ||
                cuda::std::is_same_v<T, float> || is_matx_half_v<T> ||
                cuda::std::is_same_v<T, matxFp16Complex> ||
                cuda::std::is_same_v<T, matxBf16Complex> ||
                cuda::std::is_same_v<T, matxFp16ComplexPlanar> ||
                cuda::std::is_same_v<T, matxBf16ComplexPlanar>) {
    return CUBLAS_COMPUTE_32F;
  }
  if constexpr (cuda::std::is_same_v<T, cuda::std::complex<double>> ||
                cuda::std::is_same_v<T, double>) {
    return CUBLAS_COMPUTE_64F;
  }

MATX_IGNORE_WARNING_NEXT_LINE_MSVC(4702)
  return CUBLAS_COMPUTE_32F;
}

template <typename T>
constexpr cusparseIndexType_t MatXTypeToCuSparseIndexType()
{
  if constexpr (cuda::std::is_same_v<T, uint16_t>) {
    return CUSPARSE_INDEX_16U;
  }
  if constexpr (cuda::std::is_same_v<T, int32_t>) {
    return CUSPARSE_INDEX_32I;
  }
  if constexpr (cuda::std::is_same_v<T, int64_t>) {
    return CUSPARSE_INDEX_64I;
  }
  if constexpr (cuda::std::is_same_v<T, index_t>) {
    return CUSPARSE_INDEX_64I;
  }
  else {
    return CUSPARSE_INDEX_32I;
  }
}

}
