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
#ifdef __CUDACC__
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
#error "MatX requires CUDA compute capability 6.0 or newer."
#endif
#include <cuda_runtime_api.h>
#endif

// defines.h should always be included first. Its definitions may impact
// the behavior of other headers.
#include "matx/core/defines.h"
#include "matx/core/error.h"
#include "matx/core/log.h"
#include "matx/file_io/file_io.h"
#include "matx/core/half_complex.h"
#include "matx/core/half.h"
#include "matx/core/nvtx.h"
#include "matx/core/print.h"
#include "matx/core/pybind.h"
#include "matx/core/tensor.h"
#include "matx/core/sparse_tensor.h"  // sparse support is experimental
#include "matx/core/make_sparse_tensor.h"
#include "matx/core/tie.h"
#include "matx/core/utils.h"
#include "matx/core/viz.h"

#include "matx/executors/executors.h"
#include "matx/generators/generators.h"
#include "matx/operators/operators.h"
#include "matx/transforms/transforms.h"

#include <cuda/std/complex>
namespace matx {
  using fcomplex = cuda::std::complex<float>;
  using dcomplex = cuda::std::complex<double>;
}
