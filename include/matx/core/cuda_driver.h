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

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace matx::detail {

// Lookup is deferred until the first call, so merely loading a MatX consumer
// does not require a driver. The requested ABI matches the compiling headers.
template <typename Function>
inline Function GetCudaDriverFunction(const char *name)
{
  void *function = nullptr;
  cudaDriverEntryPointQueryResult status{};
#if CUDART_VERSION >= 12050
  const auto result = cudaGetDriverEntryPointByVersion(
      name, &function, CUDA_VERSION, cudaEnableDefault, &status);
#else
  const auto result = cudaGetDriverEntryPoint(
      name, &function, cudaEnableDefault, &status);
#endif
  if (result != cudaSuccess || status != cudaDriverEntryPointSuccess) {
    return nullptr;
  }
  return reinterpret_cast<Function>(function);
}

} // namespace matx::detail

// Each call site caches its typed entry point with thread-safe initialization.
// decltype only inspects the declaration; it creates no reference to libcuda.
#define MATX_CUDA_DRIVER_CALL(function, ...)                                  \
  ([&]() -> CUresult {                                                        \
    static const auto entry =                                                \
        matx::detail::GetCudaDriverFunction<decltype(&function)>(#function);   \
    return entry != nullptr ? entry(__VA_ARGS__) : CUDA_ERROR_NOT_INITIALIZED; \
  }())
