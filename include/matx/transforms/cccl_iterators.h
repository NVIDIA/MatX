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

//! CCCL is deprecating a lot of thrust iterators for the newer `cuda` iterators
//! Work around the deprecation warning by conditionally replacing the definition of the iterators

#include <cuda/std/version>
#if CCCL_VERSION >= 3002000
#include <cuda/iterator>
#else // CCCL_VERSION < 3002000
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#endif // CCCL_VERSION < 3002000

namespace matx::detail {

#if CCCL_VERSION >= 3002000
using ::cuda::counting_iterator;
using ::cuda::make_counting_iterator;
using ::cuda::zip_iterator;
using ::cuda::make_zip_iterator;
#else // CCCL_VERSION < 3002000
using thrust::counting_iterator;
using thrust::make_counting_iterator;
using thrust::zip_iterator;
using thrust::make_zip_iterator;
#endif // CCCL_VERSION < 3002000

} // namespace matx::detail
