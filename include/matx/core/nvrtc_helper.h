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

#ifdef MATX_EN_JIT

#include <cuda.h>
#define JITIFY_ENABLE_NVTX 1
#define JITIFY_VERBOSE_ERRORS 1
#define JITIFY_ENABLE_EMBEDDED_FILES 1
#define JITIFY_IGNORE_NOT_TRIVIALLY_COPYABLE_ARGS 1
#ifdef MATX_EN_JIT_PREPROCESSING
  #include "matx.h.jit.hpp"
#endif
#include "matx/core/jitify2.hpp"
#include "matx/executors/jit_kernel.h"
//#include "matx/core/type_utils.h"
#include <filesystem>
#include <source_location>
#include <vector>
#include <string>


namespace matx {

namespace detail {  

std::vector<std::string> __MATX_HOST__ __MATX_INLINE__ get_preprocessor_options() {
    // Get the project root from the current file's location
    const auto source_path = std::filesystem::path(std::source_location::current().file_name());
    // This assumes nvrtc.h is in <root>/include/matx/core/
    const auto matx_root = source_path.parent_path().parent_path().parent_path().parent_path();
    const auto build_dir = std::filesystem::current_path();

    std::vector<std::string> options;
    options.push_back("-DMATX_EN_MATHDX");
    options.push_back("-DMATX_EN_JIT");
    options.push_back("-I" + matx_root.string() + "/include");
    //options.push_back("-I" + matx_root.string() + "/include/matx/core/");
    //options.push_back("-I" + matx_root.string() + "/include/matx/kernels");
    
    // Dependencies in the build directory
    options.push_back("-I" + (build_dir / "_deps/cccl-src/thrust").string());
    options.push_back("-I" + (build_dir / "_deps/cccl-src/libcudacxx/include").string());
    options.push_back("-I" + (build_dir / "_deps/cccl-src/cub").string());
    //options.push_back("-I" + (build_dir / "_deps/pybind11-src/include").string());
    options.push_back("-I" + (build_dir / "_deps/mathdx-src/nvidia/mathdx/25.06/include").string());
    options.push_back("-I" + (build_dir / "_deps/mathdx-src/nvidia/mathdx/25.06/external/cutlass/include").string());

    // System paths
    //options.push_back("-I/usr/include/python3.10"); // This might need to be configured differently
    options.push_back("-I" + jitify2::get_cuda_include_dir());

    options.push_back("-no-system-headers-workaround");
    
    // Use CMake-configured CUDA architecture and C++ standard
    #ifdef NVRTC_CUDA_ARCH
        options.push_back("-arch=sm_" NVRTC_CUDA_ARCH);
    #else
        options.push_back("-arch=sm_80");  // fallback
    #endif
    
    #ifdef NVRTC_CXX_STANDARD
        options.push_back("-std=c++" NVRTC_CXX_STANDARD);
    #else
        options.push_back("-std=c++20");   // fallback
    #endif

    return options;
}

template <typename Op>
std::string get_kernel_name([[maybe_unused]] const Op &op, bool stride) {
  if constexpr (Op::Rank() == 0) {
    return "matx::detail::matxOpT0Kernel";
  }  
  else if constexpr (Op::Rank() == 1) {
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
std::string generate_capability_params_string([[maybe_unused]] const Op &op, ElementsPerThread EPT, bool JIT, int osize, int block_size) {
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

  // std::string jit_caps_str = "";
  // if (detail::get_operator_capability<OperatorCapability::SUPPORTS_JIT>(op)) {
  //   const auto jit_query_input = detail::JITQueryInput{EPT};
  //   jit_caps_str = detail::get_operator_capability<OperatorCapability::JIT_CLASS_QUERY>(op, jit_query_input);
  // }

  std::string final_str =  
         "namespace matx { namespace detail {\n"     
         "template <ElementsPerThread EPT, bool JIT>\n"
         "struct CapabilityParams {\n"
         "  static constexpr ElementsPerThread ept = EPT;\n"
         "  static constexpr bool jit = JIT;\n"
         "  static constexpr int osize = " + std::to_string(osize) + ";\n"
         "  static constexpr int block_size = " + std::to_string(block_size) + ";\n"
         "};\n"
         //"\n" + jit_caps_str + "\n"
         "using CurrentCapabilities = CapabilityParams<" + ept_str + ", " + jit_str + ">;\n"
         "} }\n";

  //final_str += std::string(matxKernelStr);       
  return final_str;
}


template <typename Op, typename SizeArray>
auto nvrtc_compile_and_run([[maybe_unused]] const std::string &name, Op op, const SizeArray &sa, dim3 &blocks, dim3 &threads, ElementsPerThread ept, bool stride, int dynamic_shmem_size, int osize) {
  //static bool initialized = false;
  //static jitify2::PreprocessedProgram preprog;
  // if (!initialized) {
  //   initialized = true;
  using jitify2::get_cuda_include_dir, jitify2::Program, jitify2::ProgramCache;
  using jitify2::reflection::Template, jitify2::reflection::Type;  

  //auto start_time = std::chrono::high_resolution_clock::now();
  
#ifndef MATX_EN_JIT_PREPROCESSING  
  static ProgramCache<> cache(
      100,
      *Program(name, std::string(matxKernelStr))
           // Preprocess source code and load all included headers.
           ->preprocess(get_preprocessor_options()));
#else          
  static ProgramCache<> cache(
      100,
      *matx_h_jit);
#endif

  // Get all the JIT strings from each operator
  std::unordered_map<std::string, std::string> jit_strings;
  auto string_res = detail::get_operator_capability<OperatorCapability::JIT_CLASS_QUERY>(op, jit_strings);
  for (const auto& kv : jit_strings) {
    std::cout << "JIT string key: " << kv.first << "\n";
    std::cout << "JIT string value:\n" << kv.second << "\n";
  }

  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  //printf("Preprocess step took %ld microseconds\n", duration.count());

  printf("DEBUG: nvrtc_compile_and_run called with operator type: %s\n", typeid(op).name());
  const auto ltoir_query_input = detail::LTOIRQueryInput{ept};
  auto ltoir = detail::get_operator_capability<OperatorCapability::GENERATE_LTOIR>(op, ltoir_query_input);
  printf("DEBUG: LTOIR capability result: %s\n", ltoir ? "true" : "false");
    auto capstr = generate_capability_params_string(op, ept, false, osize, threads.x);
    std::cout << "DEBUG: Capability string: " << capstr << std::endl;
    auto tstr = jitify2::reflection::reflect_template<Op>();
    printf("DEBUG: Template string: %s\n", tstr.c_str());
    //auto start_time_kernel = std::chrono::high_resolution_clock::now();
    auto kernel = cache
    .get_kernel(Template(get_kernel_name(op, stride)).instantiate<Op>(), 
    {}, 
    {{"matx_generated_code_hdr", capstr}}, {"-include=matx_generated_code_hdr"});    
        // Compile, link, and load the program, and obtain the loaded kernel.
        // .get_kernel(Template(get_kernel_name(op, stride)).instantiate<Op>(), 
        //   {"defines.h", "half.h", "complex_half.h", "type_utils_both.h"}, 
        //   {{"matx_generated_code_hdr", capstr}}, {"-include=matx_generated_code_hdr"});
    // auto end_time_kernel = std::chrono::high_resolution_clock::now();
    // auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_time_kernel - start_time_kernel);
    //printf("Kernel step took %ld microseconds\n", duration_kernel.count());
      // Get the current static shared memory size for the device

    //auto start_time_device = std::chrono::high_resolution_clock::now();
    int device;
    cudaGetDevice(&device);
    int static_shared_size;
    cudaDeviceGetAttribute(&static_shared_size, cudaDevAttrMaxSharedMemoryPerBlock, device);
    // auto end_time_device = std::chrono::high_resolution_clock::now();
    // auto duration_device = std::chrono::duration_cast<std::chrono::microseconds>(end_time_device - start_time_device);
   // printf("Device attribute calls took %ld microseconds\n", duration_device.count());
    
    
    // Need to set dynamic shared memory size if it is greater than the static shared memory size
    if (dynamic_shmem_size > static_shared_size) {
      kernel->set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_shmem_size);
    }      


    // Configure the kernel launch.
    //auto start_time_configure = std::chrono::high_resolution_clock::now();
    if constexpr (Op::Rank() == 0) {
      kernel->configure(blocks, threads, dynamic_shmem_size)
            // Launch the kernel.
            ->launch(op);
    }
    else if constexpr (Op::Rank() == 1) {
      kernel->configure(blocks, threads, dynamic_shmem_size)
            // Launch the kernel.
            ->launch(op, sa[0]);
    }
    else if constexpr (Op::Rank() == 2) {
    kernel->configure(blocks, threads, dynamic_shmem_size)
            // Launch the kernel.
            ->launch(op, sa[0], sa[1]);
    }
    else if constexpr (Op::Rank() == 3) {
      kernel->configure(blocks, threads, dynamic_shmem_size)
            // Launch the kernel.
            ->launch(op, sa[0], sa[1], sa[2]);
    }
    else if constexpr (Op::Rank() == 4) {
      kernel->configure(blocks, threads, dynamic_shmem_size)
            // Launch the kernel.
            ->launch(op, sa[0], sa[1], sa[2], sa[3]);
    }    
    else {
      MATX_THROW(matxInvalidParameter, "Rank not supported");
    }
  // auto end_time_configure = std::chrono::high_resolution_clock::now();
  // auto duration_configure = std::chrono::duration_cast<std::chrono::microseconds>(end_time_configure - start_time_configure);
  // printf("Configure step took %ld microseconds\n", duration_configure.count());

}

}
}
#endif