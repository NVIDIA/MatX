////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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
//#define JITIFY_ENABLE_NVTX 1
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

  std::string final_str =  "#include <matx.h>\n"
         "namespace matx { namespace detail {\n"
         "template <ElementsPerThread EPT, bool JIT>\n"
         "struct CapabilityParams {\n"
         "  static constexpr ElementsPerThread ept = EPT;\n"
         "  static constexpr bool jit = JIT;\n"
         "};\n"
         "\n" + jit_caps_str + "\n"
         "using CurrentCapabilities = CapabilityParams<" + ept_str + ", " + jit_str + ">;\n"
         "} }\n";

  printf("%s\n", final_str.c_str());
  return final_str;
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
auto nvrtc_compile_and_run(const std::string &name, Op op, const SizeArray &sa, dim3 &blocks, dim3 &threads, ElementsPerThread ept, bool stride, int dynamic_shmem_size) {
  //static bool initialized = false;
  //static jitify2::PreprocessedProgram preprog;
  // if (!initialized) {
  //   initialized = true;
  using jitify2::get_cuda_include_dir, jitify2::Program, jitify2::ProgramCache;
  using jitify2::reflection::Template, jitify2::reflection::Type;  

  auto start_time = std::chrono::high_resolution_clock::now();
  
  static ProgramCache<> cache(
      /*max_size=*/100,
      *Program(name, std::string(matxKernelStr))
           // Preprocess source code and load all included headers.
           ->preprocess(
               {"-DMATX_EN_MATHDX",
      "-I/repro/MatX/include", "-I/repro/MatX/include/matx/kernels", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/thrust/../../../thrust", 
      "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/cub/../../../cub", 
      "-I/repro/MatX/build/_deps/pybind11-src/include", "-I/usr/include/python3.10", "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.06/include", 
      "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.06/external/cutlass/include", "-I/usr/local/cuda/include",
      "-no-system-headers-workaround",
      "-arch=sm_80","-std=c++20"}));

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  printf("Preprocess step took %ld microseconds\n", duration.count());

  // static ProgramCache<> cache(
  //     /*max_size=*/100,
  //     *Program(name, std::string(matxKernelStr))
  //          // Preprocess source code and load all included headers.
  //          ->preprocess(
  //              {"-DMATX_EN_MATHDX",
  //     "-I/repro/MatX/include", "-I/repro/MatX/include/matx/kernels", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/thrust/../../../thrust", 
  //     "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/cub/../../../cub", 
  //     "-I/repro/MatX/build/_deps/pybind11-src/include", "-I/usr/include/python3.10", "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.06/include", 
  //     "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.06/external/cutlass/include", "-I/usr/local/cuda/include",
  //     "-no-system-headers-workaround",
  //     "-arch=sm_80","-std=c++17"}));

    // static ProgramCache<> cache(100, *Program("my_program", std::string(matxKernelStr)))
    //   ->preprocess({"-DMATX_EN_MATHDX",
    //   "-I/repro/MatX/include", "-I/repro/MatX/include/matx/kernels", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/thrust/../../../thrust", 
    //   "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/cub/../../../cub", 
    //   "-I/repro/MatX/build/_deps/pybind11-src/include", "-I/usr/include/python3.10", "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.06/include", 
    //   "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.06/external/cutlass/include", "-I/usr/local/cuda/include",
    //   "-no-system-headers-workaround",
    //   "-arch=sm_80","-std=c++17"});
    auto capstr = generate_capability_params_string(op, ept, false);
  auto start_time_kernel = std::chrono::high_resolution_clock::now();
  auto kernel = cache
      // Compile, link, and load the program, and obtain the loaded kernel.
      .get_kernel(Template(get_kernel_name(op, stride)).instantiate<Op>(), {}, {{"matx_generated_code_hdr", capstr}}, {"-include=matx_generated_code_hdr"});
  auto end_time_kernel = std::chrono::high_resolution_clock::now();
  auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_time_kernel - start_time_kernel);
  printf("Kernel step took %ld microseconds\n", duration_kernel.count());
    // Get the current static shared memory size for the device

    auto start_time_device = std::chrono::high_resolution_clock::now();
    int device;
    cudaGetDevice(&device);
    int static_shared_size;
    cudaDeviceGetAttribute(&static_shared_size, cudaDevAttrMaxSharedMemoryPerBlock, device);
    auto end_time_device = std::chrono::high_resolution_clock::now();
    auto duration_device = std::chrono::duration_cast<std::chrono::microseconds>(end_time_device - start_time_device);
    printf("Device attribute calls took %ld microseconds\n", duration_device.count());
    
    
    // Need to set dynamic shared memory size if it is greater than the static shared memory size
    if (dynamic_shmem_size > static_shared_size) {
      kernel->set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_shmem_size);
    }      
      printf("dynamic_shmem_size %d\n", dynamic_shmem_size);    
    printf(" threads %d %d\n", threads.x, threads.y);
    // Configure the kernel launch.
    auto start_time_configure = std::chrono::high_resolution_clock::now();
    kernel->configure(blocks, threads, dynamic_shmem_size)
          // Launch the kernel.
          ->launch(op, sa[0]);
  auto end_time_configure = std::chrono::high_resolution_clock::now();
  auto duration_configure = std::chrono::duration_cast<std::chrono::microseconds>(end_time_configure - start_time_configure);
  printf("Configure step took %ld microseconds\n", duration_configure.count());


  //}
  // const auto kernel_name_str = get_kernel_name(op, stride);
  // const auto capability_params_str = build_rtc_string(op, ept, false);

  // jitify2::PreprocessedProgram preprog = jitify2::Program(name, capability_params_str)
  //     // Preprocess source code and load all included headers.
  //     ->preprocess({"-DMATX_EN_MATHDX",
  //     "-I/repro/MatX/include", "-I/repro/MatX/include/matx/kernels", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/thrust/../../../thrust", 
  //     "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include", "-I/repro/MatX/build/_deps/cccl-src/lib/cmake/cub/../../../cub", 
  //     "-I/repro/MatX/build/_deps/pybind11-src/include", "-I/usr/include/python3.10", "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.06/include", 
  //     "-I/repro/MatX/build/_deps/mathdx-src/nvidia/mathdx/25.06/external/cutlass/include", "-I/usr/local/cuda/include",
  //     "-no-system-headers-workaround",
  //     "-arch=sm_80","-std=c++17"});

  // using jitify2::reflection::Type;
  // using jitify2::reflection::NonType;


  // auto kernel_name = jitify2::reflection::Template(kernel_name_str).instantiate<Op>();
  // std::cout << "kernel name: " << kernel_name << std::endl;
  // if (!preprog) {
  //   std::cerr << preprog.error() << std::endl;
  //   *preprog;
  // } else {
  //   jitify2::PreprocessedProgramData preprog_data = *preprog;
  //   jitify2::CompiledProgram compiled = preprog->compile(kernel_name);

  //   auto program = compiled->link()->load();
  //   printf("program done\n");
  //   auto kernel = program->get_kernel(kernel_name);
    
  //   // Get the current static shared memory size for the device
  //   int device;
  //   cudaGetDevice(&device);
  //   int static_shared_size;
  //   cudaDeviceGetAttribute(&static_shared_size, cudaDevAttrMaxSharedMemoryPerBlock, device);
  //   printf("Device %d static shared memory size: %d bytes\n", device, static_shared_size);
    
  //   // Need to set dynamic shared memory size if it is greater than the static shared memory size
  //   if (dynamic_shmem_size > static_shared_size) {
  //     printf("dynamic_shmem_size %d\n", dynamic_shmem_size);
  //     kernel->set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_shmem_size);
  //   }
    
    
  //   kernel->configure(blocks, threads, dynamic_shmem_size)
  //         ->launch(op, sa[0]);
  // }
}

}
}
