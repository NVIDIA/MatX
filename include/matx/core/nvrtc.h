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

#include <cuda.h>

//#define JITIFY_VERBOSE_ERRORS 1
#define JITIFY_ENABLE_EMBEDDED_FILES 1
#define JITIFY_IGNORE_NOT_TRIVIALLY_COPYABLE_ARGS 1
#include "matx/core/jitify2.hpp"
#include "matx/executors/kernel.h"
#include "matx/core/type_utils.h"


namespace matx {

namespace detail {

template <typename Op>
std::string get_kernel_name([[maybe_unused]] const Op &op, bool stride) {
  if constexpr (Op::Rank() == 1) {
    return "matx::detail::matxOpT1Kernel";
  }
  else if constexpr (Op::Rank() == 2) {
    if (stride) {
      return "matx::detail::matxOpT2StrideKernel";
    } else {
      return "matx::detail::matxOpT2Kernel";
    }
  }
  else if constexpr (Op::Rank() == 3) {
    if (stride) {
      return "matx::detail::matxOpT3StrideKernel";
    } else {
      return "matx::detail::matxOpT3Kernel";
    }
  }
  else if constexpr (Op::Rank() == 4) {
    if (stride) {
      return "matx::detail::matxOpT4StrideKernel";
    } else {
      return "matx::detail::matxOpT4Kernel";
    }
  }

  return "MatXInvalidKernel";
}

template <typename Op>
std::string generate_capability_params_string(const Op &op, ElementsPerThread EPT, bool JIT) {
  std::string ept_str;
  switch (EPT) {
    case ElementsPerThread::ONE:
      ept_str = "matx::detail::ElementsPerThread::ONE";
      break;
    case ElementsPerThread::TWO:
      ept_str = "matx::detail::ElementsPerThread::TWO";
      break;
    case ElementsPerThread::FOUR:
      ept_str = "matx::detail::ElementsPerThread::FOUR";
      break;
    case ElementsPerThread::EIGHT:
      ept_str = "matx::detail::ElementsPerThread::EIGHT";
      break;
    case ElementsPerThread::SIXTEEN:
      ept_str = "matx::detail::ElementsPerThread::SIXTEEN";
      break;
    case ElementsPerThread::THIRTY_TWO:
      ept_str = "matx::detail::ElementsPerThread::THIRTY_TWO";
      break;
    default:
      ept_str = "matx::detail::ElementsPerThread::ONE";
      break;
  }
  
  std::string jit_str = JIT ? "true" : "false";

  std::string jit_caps_str = "";
  if (detail::get_operator_capability<OperatorCapability::SUPPORTS_JIT>(op)) {
    jit_caps_str = detail::get_operator_capability<OperatorCapability::JIT_CAP_QUERY>(op);
  }
  
  return "namespace matx { namespace detail {\n"
         "template <ElementsPerThread EPT, bool JIT>\n"
         "struct CapabilityParams {\n"
         "  static constexpr ElementsPerThread ept = EPT;\n"
         "  static constexpr bool jit = JIT;\n"
         + jit_caps_str + "\n"
         "};\n"
         "using CurrentCapabilities = CapabilityParams<" + ept_str + ", " + jit_str + ">;\n"
         "} }\n";
}

template <typename Op>
std::string build_rtc_string(const Op &op, ElementsPerThread EPT, bool JIT) {
  std::string rtc_str = "#include \"matx.h\"\n\n";
  rtc_str += generate_capability_params_string(op, EPT, JIT);
  rtc_str += matxKernelStr;
  printf("%s\n", rtc_str.c_str());
  return rtc_str;
}


template <typename Op, typename SizeArray>
auto nvrtc_compile_and_run(const std::string &name, Op op, const SizeArray &sa, dim3 &blocks, dim3 &threads, ElementsPerThread ept, bool stride) {

  const auto kernel_name_str = get_kernel_name(op, stride);
  const auto capability_params_str = build_rtc_string(op, ept, false);

  jitify2::PreprocessedProgram preprog = jitify2::Program(name, capability_params_str)
      // Preprocess source code and load all included headers.
      ->preprocess({"-DMATX_EN_MATHDX",
      "-I/repro/MatX/include", "-I/repro/MatX/include/matx/kernels", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/thrust/../../../thrust", 
      "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/cub/../../../cub", 
      "-I/repro/MatX/build/_deps/pybind11-src/include", "-I/usr/include/python3.10", "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.01/include", 
      "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.01/external/cutlass/include", "-I/usr/local/cuda/include",

                  "-no-system-headers-workaround",
                  "-arch=sm_80","-std=c++17"});

  using jitify2::reflection::Type;
  using jitify2::reflection::NonType;
  //auto kernel_name = jitify2::reflection::Template("matx::detail::matxOpT1Kernel").instantiate<Type<detail::ElementsPerThread>(), Op>();
  auto kernel_name = jitify2::reflection::Template(kernel_name_str).instantiate<Op>();
  std::cout << "kernel name: " << kernel_name << std::endl;
  if (!preprog) {
    std::cerr << preprog.error() << std::endl;
    *preprog;
  } else {
    jitify2::PreprocessedProgramData preprog_data = *preprog;
    jitify2::CompiledProgram compiled = preprog->compile(kernel_name);
    printf("Compiled program\n");
    compiled->link()
    ->load()
    ->get_kernel(kernel_name)
    ->configure(blocks, threads)
    ->launch(op, sa[0]);
  }
}

}
}
