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

namespace matx {
  namespace detail {

#ifdef MATX_EN_MATHDX
static const char *matxKernelStr = "\n\
#include \"matx/core/defines.h\";\n\
#include \"matx/core/type_utils_both.h\";\n\
namespace matx {\n\
  namespace detail {\n\
    template <class Op>\n\
    __global__ void matxOpT0Kernel(Op op) {\n\
      if constexpr (std::is_pointer_v<Op>) {\n\
        (*op)();\n\
      } else {\n\
        op();\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT1Kernel(Op op, matx::index_t size0) {\n\
      matx::index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
      if (idx * static_cast<index_t>(CurrentCapabilities::ept) < size0) {\n\
        if constexpr (std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT2Kernel(Op op, matx::index_t size0, matx::index_t size1) {\n\
      matx::index_t idx = threadIdx.x;\n\
      matx::index_t idy = static_cast<matx::index_t>(blockIdx.x);\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size1 && idy < size0) {\n\
        if constexpr (std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idy, idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT2StrideKernel(Op op, matx::index_t size0, matx::index_t size1) {\n\
      matx::index_t idx = threadIdx.x;\n\
      for(matx::index_t idy = static_cast<matx::index_t>(blockIdx.x);\n\
        idy < size0;\n\
        idy += blockDim.x * gridDim.x) {\n\
        if constexpr (std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idy, idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT3Kernel(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2) {\n\
      matx::index_t idx = threadIdx.x;\n\
      matx::index_t idy = static_cast<matx::index_t>(blockIdx.x);\n\
      matx::index_t idz = static_cast<matx::index_t>(blockIdx.y);\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size2 && idy < size1 && idz < size0) {\n\
        if constexpr (std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idz, idy, idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idz, idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT3StrideKernel(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2) {\n\
      matx::index_t idx = threadIdx.x;\n\
      matx::index_t idy = static_cast<matx::index_t>(blockIdx.x);\n\
      matx::index_t idz = static_cast<matx::index_t>(blockIdx.y);\n\
      for(matx::index_t idz = static_cast<matx::index_t>(blockIdx.z);\n\
          idz < size0;\n\
          idz += gridDim.z) {\n\
        for (matx::index_t idy = static_cast<matx::index_t>(blockIdx.y);\n\
            idy < size1;\n\
            idy += gridDim.y) {\n\
          if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size2 && idy < size1 && idz < size0) {\n\
            if constexpr (std::is_pointer_v<Op>) {\n\
              (*op).template operator()<CurrentCapabilities>(idz, idy, idx);\n\
            } else {\n\
              op.template operator()<CurrentCapabilities>(idz, idy, idx);\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT4Kernel(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2, matx::index_t size3) {\n\
      matx::index_t idx = threadIdx.x;\n\
      matx::index_t idy = static_cast<matx::index_t>(blockIdx.x);\n\
      matx::index_t idz = static_cast<matx::index_t>(blockIdx.y);\n\
      matx::index_t idw = static_cast<matx::index_t>(blockIdx.z);\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size3 && idy < size2 && idz < size1 && idw < size0) {\n\
        if constexpr (std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT4StrideKernel(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2, matx::index_t size3) {\n\
      matx::index_t idx = threadIdx.x;\n\
      for(matx::index_t idw = static_cast<matx::index_t>(blockIdx.z);\n\
            idw < size0;\n\
            idw += gridDim.z) {\n\
        for(matx::index_t idz = static_cast<matx::index_t>(blockIdx.y);\n\
            idz < size1;\n\
            idz += gridDim.y) {\n\
          for (matx::index_t idy = static_cast<matx::index_t>(blockIdx.x);\n\
              idy < size2;\n\
              idy += gridDim.x) {\n\
            if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size3 && idy < size2 && idz < size1 && idw < size0) {\n\
              if constexpr (std::is_pointer_v<Op>) {\n\
                (*op).template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
              } else {\n\
                op.template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
              }\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
  }\n\
}";
#else
[[maybe_unused]] static const char *matxKernelStr = nullptr;
#endif 

}


} // end namespace matx

