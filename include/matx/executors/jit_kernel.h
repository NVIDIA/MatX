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

#ifdef MATX_EN_JIT
static const char *matxKernelStr = "\n\
namespace matx {\n\
  namespace detail {\n\
    template <class Op>\n\
    __global__ void matxOpT0KernelBlock(Op op) {\n\
      if constexpr (cuda::std::is_pointer_v<Op>) {\n\
        (*op).template operator()<CurrentCapabilities>();\n\
      } else {\n\
        op.template operator()<CurrentCapabilities>();\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT1KernelBlock(Op op, matx::index_t size0) {\n\
      matx::index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
      if (idx * static_cast<index_t>(CurrentCapabilities::ept) < size0) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT2KernelBlock(Op op, matx::index_t size0, matx::index_t size1) {\n\
      matx::index_t idx = threadIdx.x;\n\
      matx::index_t idy = static_cast<matx::index_t>(blockIdx.x)*blockDim.y + threadIdx.y;\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size1 && idy < size0) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idy, idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT2StrideKernelBlock(Op op, matx::index_t size0, matx::index_t size1) {\n\
      matx::index_t idx = threadIdx.x;\n\
      for(matx::index_t idy = static_cast<matx::index_t>(blockIdx.x);\n\
        idy < size0;\n\
        idy += blockDim.x * gridDim.x) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idy, idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT3KernelBlock(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2) {\n\
      matx::index_t idx = threadIdx.x;\n\
      matx::index_t idy = static_cast<matx::index_t>(blockIdx.x) * blockDim.y + threadIdx.y;\n\
      matx::index_t idz = static_cast<matx::index_t>(blockIdx.y) * blockDim.z + threadIdx.z;\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size2 && idy < size1 && idz < size0) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idz, idy, idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idz, idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT3StrideKernelBlock(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2) {\n\
      matx::index_t idx = threadIdx.x;\n\
      for(matx::index_t idz = static_cast<matx::index_t>(blockIdx.y) * blockDim.z + threadIdx.z;\n\
          idz < size0;\n\
          idz += blockDim.z * gridDim.y) {\n\
        for (matx::index_t idy = static_cast<matx::index_t>(blockIdx.x) * blockDim.y + threadIdx.y;\n\
            idy < size1;\n\
            idy += blockDim.y * gridDim.x) {\n\
          if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size2) {\n\
            if constexpr (cuda::std::is_pointer_v<Op>) {\n\
              (*op).template operator()<CurrentCapabilities>(idz, idy, idx);\n\
            }\n\
            else {\n\
              op.template operator()<CurrentCapabilities>(idz, idy, idx);\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
    template <class Op>\n\
    __global__ void matxOpT4KernelBlock(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2, matx::index_t size3) {\n\
      matx::index_t idx = threadIdx.x;\n\
      matx::index_t idy = blockIdx.x;\n\
      matx::index_t idz = blockIdx.y;\n\
      matx::index_t idw = blockIdx.z;\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size3 && idy < size2 && idz < size1 && idw < size0) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
        } else {\n\
          op.template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT4StrideKernelBlock(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2, matx::index_t size3) {\n\
      matx::index_t idx = threadIdx.x;\n\
      for(matx::index_t nmy = static_cast<matx::index_t>(blockIdx.x) * blockDim.y + threadIdx.y;\n\
          nmy < size1 * size2;\n\
          nmy += blockDim.y * gridDim.x) {\n\
        matx::index_t idy = nmy % size2;\n\
        matx::index_t idz = nmy / size2;\n\
        if(idy < size2 && idz < size1) {\n\
          for(matx::index_t idw = static_cast<matx::index_t>(blockIdx.y) * blockDim.z + threadIdx.z;\n\
              idw < size0;\n\
              idw += blockDim.z * gridDim.y) {\n\
            if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size3) {\n\
              if constexpr (cuda::std::is_pointer_v<Op>) {\n\
                (*op).template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
              } else {\n\
                op.template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
              }\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT0Kernel(Op op) {\n\
      if constexpr (cuda::std::is_pointer_v<Op>) {\n\
        (*op).template operator()<CurrentCapabilities>();\n\
      }\n\
      else {\n\
        op.template operator()<CurrentCapabilities>();\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT1Kernel(Op op, matx::index_t size0) {\n\
      matx::index_t idx = static_cast<matx::index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size0) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idx);\n\
        }\n\
        else {\n\
          op.template operator()<CurrentCapabilities>(idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT2Kernel(Op op, matx::index_t size0, matx::index_t size1) {\n\
      matx::index_t idx = static_cast<matx::index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
      matx::index_t idy = static_cast<matx::index_t>(blockIdx.y) * blockDim.y + threadIdx.y;\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size1 && idy < size0) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idy, idx);\n\
        }\n\
        else {\n\
          op.template operator()<CurrentCapabilities>(idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT2StrideKernel(Op op, matx::index_t size0, matx::index_t size1) {\n\
      for(matx::index_t idy = static_cast<matx::index_t>(blockIdx.y) * blockDim.y + threadIdx.y;\n\
          idy < size0;\n\
          idy += blockDim.y * gridDim.y) {\n\
        for(matx::index_t idx = static_cast<matx::index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
            idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size1;\n\
            idx += blockDim.x * gridDim.x) {\n\
          if constexpr (cuda::std::is_pointer_v<Op>) {\n\
            (*op).template operator()<CurrentCapabilities>(idy, idx);\n\
          }\n\
          else {\n\
            op.template operator()<CurrentCapabilities>(idy, idx);\n\
          }\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT3Kernel(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2) {\n\
      matx::index_t idx = static_cast<matx::index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
      matx::index_t idy = static_cast<matx::index_t>(blockIdx.y) * blockDim.y + threadIdx.y;\n\
      matx::index_t idz = static_cast<matx::index_t>(blockIdx.z) * blockDim.z + threadIdx.z;\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size2 && idy < size1 && idz < size0) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idz, idy, idx);\n\
        }\n\
        else {\n\
          op.template operator()<CurrentCapabilities>(idz, idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT3StrideKernel(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2) {\n\
      for(matx::index_t idz = static_cast<matx::index_t>(blockIdx.z) * blockDim.z + threadIdx.z;\n\
          idz < size0;\n\
          idz += blockDim.z * gridDim.z) {\n\
        for (matx::index_t idy = static_cast<matx::index_t>(blockIdx.y) * blockDim.y + threadIdx.y;\n\
            idy < size1;\n\
            idy += blockDim.y * gridDim.y) {\n\
          for(matx::index_t idx = static_cast<matx::index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
              idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size2;\n\
              idx += blockDim.x * gridDim.x) {\n\
            if constexpr (cuda::std::is_pointer_v<Op>) {\n\
              (*op).template operator()<CurrentCapabilities>(idz, idy, idx);\n\
            }\n\
            else {\n\
              op.template operator()<CurrentCapabilities>(idz, idy, idx);\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT4Kernel(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2, matx::index_t size3) {\n\
      matx::index_t idx = static_cast<matx::index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
      matx::index_t nmy = static_cast<matx::index_t>(blockIdx.y) * blockDim.y + threadIdx.y;\n\
      matx::index_t idy = nmy % size2;\n\
      matx::index_t idz = nmy / size2;\n\
      matx::index_t idw = static_cast<matx::index_t>(blockIdx.z) * blockDim.z + threadIdx.z;\n\
      if (idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size3 && idy < size2 && idz < size1 && idw < size0) {\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          (*op).template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
        }\n\
        else {\n\
          op.template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT4StrideKernel(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2, matx::index_t size3) {\n\
      for(matx::index_t nmy = static_cast<matx::index_t>(blockIdx.y) * blockDim.y + threadIdx.y;\n\
          nmy < size1 * size2;\n\
          nmy += blockDim.y * gridDim.y) {\n\
        matx::index_t idy = nmy % size2;\n\
        matx::index_t idz = nmy / size2;\n\
        if(idy < size2 && idz < size1) {\n\
          for(matx::index_t idw = static_cast<matx::index_t>(blockIdx.z) * blockDim.z + threadIdx.z;\n\
              idw < size0;\n\
              idw += blockDim.z * gridDim.z) {\n\
            for(matx::index_t idx = static_cast<matx::index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
                idx * static_cast<matx::index_t>(CurrentCapabilities::ept) < size3;\n\
                idx += blockDim.x * gridDim.x) {\n\
              if constexpr (cuda::std::is_pointer_v<Op>) {\n\
                (*op).template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
              }\n\
              else {\n\
                op.template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
              }\n\
            }\n\
          }\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpTDKernel(Op op, const cuda::std::array<matx::index_t, Op::Rank()> sizes, matx::index_t mult) {\n\
      cuda::std::array<matx::index_t, Op::Rank()> indices;\n\
      static_assert(Op::Rank() >= 1, \"rank must exceed zero\");\n\
      matx::index_t x_abs = static_cast<matx::index_t>(blockIdx.x) * blockDim.x + threadIdx.x;\n\
      const bool valid = x_abs < mult*sizes[0];\n\
      if (valid) {\n\
        MATX_LOOP_UNROLL\n\
        for (int r = 0; r < Op::Rank()-1; r++) {\n\
          indices[r] = x_abs / mult;\n\
          x_abs -= indices[r] * mult;\n\
          mult /= sizes[r+1];\n\
        }\n\
        indices[Op::Rank()-1] = x_abs / mult;\n\
        if constexpr (cuda::std::is_pointer_v<Op>) {\n\
          cuda::std::apply([&](auto... args){\n\
            (*op).template operator()<CurrentCapabilities>(args...);\n\
          }, indices);\n\
        }\n\
        else {\n\
          cuda::std::apply([&](auto... args){\n\
            op.template operator()<CurrentCapabilities>(args...);\n\
          }, indices);\n\
        }\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT2KernelBlock2D(Op op, matx::index_t size0, matx::index_t size1) {\n\
      int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;\n\
      matx::index_t idx = tid % size1;\n\
      matx::index_t idy = tid / size1;\n\
      if constexpr (cuda::std::is_pointer_v<Op>) {\n\
        (*op).template operator()<CurrentCapabilities>(idy, idx);\n\
      } else {\n\
        op.template operator()<CurrentCapabilities>(idy, idx);\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT3KernelBlock2D(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2) {\n\
      int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;\n\
      matx::index_t idx = tid % size2;\n\
      matx::index_t idy = tid / size2;\n\
      matx::index_t idz = blockIdx.x;\n\
      if constexpr (cuda::std::is_pointer_v<Op>) {\n\
        (*op).template operator()<CurrentCapabilities>(idz, idy, idx);\n\
      } else {\n\
        op.template operator()<CurrentCapabilities>(idz, idy, idx);\n\
      }\n\
    }\n\
    \n\
    template <class Op>\n\
    __global__ void matxOpT4KernelBlock2D(Op op, matx::index_t size0, matx::index_t size1, matx::index_t size2, matx::index_t size3) {\n\
      int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;\n\
      matx::index_t idx = tid % size3;\n\
      matx::index_t idy = tid / size3;\n\
      matx::index_t idz = blockIdx.x;\n\
      matx::index_t idw = blockIdx.y;\n\
      if constexpr (cuda::std::is_pointer_v<Op>) {\n\
        (*op).template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
      } else {\n\
        op.template operator()<CurrentCapabilities>(idw, idz, idy, idx);\n\
      }\n\
    }\n\
  }\n\
}";
#else
[[maybe_unused]] static const char *matxKernelStr = nullptr;
#endif 

}


} // end namespace matx

