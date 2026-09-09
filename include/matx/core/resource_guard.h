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

#include <memory>
#include <type_traits>

#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "matx/core/error.h"

namespace matx {
namespace detail {

/**
 * @brief RAII guards for opaque CUDA C-API handles, built on
 * std::unique_ptr with a stateless deleter. reset(handle) takes ownership;
 * the handle is destroyed on the next reset(), on guard destruction, or
 * if an exception unwinds through the guard. 
 *
 * Creation and validation stay at the call site, since cudaStreamCreate*,
 * cudaEventCreate, etc. each have their own success code/error type:
 *
 *   detail::CudaStreamGuard d2h_guard;
 *   cudaStream_t handle;
 *   MATX_CUDA_CHECK(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking));
 *   d2h_guard.reset(handle);
 *   ...
 *   MATX_CUDA_CHECK(cudaStreamSynchronize(d2h_guard.get()));
 */
inline constexpr auto CudaStreamDeleter = [](cudaStream_t stream) noexcept {
  MATX_CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(stream));
};

inline constexpr auto CudaEventDeleter = [](cudaEvent_t event) noexcept {
  MATX_CUDA_CHECK_NOEXCEPT(cudaEventDestroy(event));
};

using CudaStreamGuard = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(CudaStreamDeleter)>;
using CudaEventGuard = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(CudaEventDeleter)>;

} // namespace detail
} // namespace matx
