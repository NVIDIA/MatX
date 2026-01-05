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

#include <nvrtc.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <matx/core/cache.h>  
#include <matx/core/log.h>

#include "matx/executors/jit_kernel.h"
#include <filesystem>
#include <nvJitLink.h>
#include <source_location>
#include <vector>
#include <string>
#include <mutex>


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
    // Dependencies in the build directory
    options.push_back("-I" + (build_dir / "_deps/cccl-src/thrust").string());
    options.push_back("-I" + (build_dir / "_deps/cccl-src/libcudacxx/include").string());
    options.push_back("-I" + (build_dir / "_deps/cccl-src/cub").string());
    //options.push_back("-I" + (build_dir / "_deps/pybind11-src/include").string());
    options.push_back("-I" + (build_dir / "_deps/mathdx-src/nvidia/mathdx/25.06/include").string());
    options.push_back("-I" + (build_dir / "_deps/mathdx-src/nvidia/mathdx/25.06/external/cutlass/include").string());

    // System paths
    // Get CUDA include directory manually for pure NVRTC
    const char* cuda_path = std::getenv("CUDA_PATH");
    std::string cuda_inc_dir = cuda_path ? std::string(cuda_path) + "/include" : "/usr/local/cuda/include";
    options.push_back("-I" + cuda_inc_dir);
    
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

// Helper function to check NVRTC errors
#define NVRTC_CHECK(call)                                                      \
  do {                                                                         \
    nvrtcResult result = call;                                                 \
    if (result != NVRTC_SUCCESS) {                                             \
      MATX_LOG_ERROR("NVRTC error: {}", nvrtcGetErrorString(result));          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Helper function to check CUDA Driver API errors
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult result = call;                                                    \
    if (result != CUDA_SUCCESS) {                                              \
      const char* errStr;                                                      \
      cuGetErrorString(result, &errStr);                                       \
      MATX_LOG_ERROR("CUDA error: {}", errStr);                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Helper function to check CUDA Runtime API errors
#define CUDA_RT_CHECK(call)                                                    \
  do {                                                                         \
    cudaError_t result = call;                                                 \
    if (result != cudaSuccess) {                                               \
      MATX_LOG_ERROR("CUDA Runtime error: {}", cudaGetErrorString(result));    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

  #ifndef NVJITLINK_CHECK
  #define NVJITLINK_CHECK(handle, ans)                                                                                   \
      do {                                                                                                               \
          nvJitLinkResult result = (ans);                                                                                \
          if (result != NVJITLINK_SUCCESS) {                                                                             \
              MATX_LOG_ERROR("nvJitLink error: {}", static_cast<int>(result));                                           \
              size_t lsize;                                                                                              \
              result = nvJitLinkGetErrorLogSize(handle, &lsize);                                                         \
              if (result == NVJITLINK_SUCCESS && lsize > 0) {                                                            \
                  std::vector<char> log(lsize);                                                                          \
                  result = nvJitLinkGetErrorLog(handle, log.data());                                                     \
                  if (result == NVJITLINK_SUCCESS) {                                                                     \
                      MATX_LOG_ERROR("nvJitLink log: {}", log.data());                                                   \
                  }                                                                                                      \
              }                                                                                                          \
              abort();                                                                                                   \
          }                                                                                                              \
      } while (0)
  #endif // NVJITLINK_CHECK
  

// Read file contents into a string
inline std::string read_file_contents(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        MATX_LOG_ERROR("Failed to open file: {}", filepath);
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Get the full path to jit_includes.h
inline std::string get_jit_includes_path() {
    const auto source_path = std::filesystem::path(std::source_location::current().file_name());
    const auto matx_root = source_path.parent_path().parent_path().parent_path().parent_path();
    return (matx_root / "include" / "matx" / "core" / "jit_includes.h").string();
}

template <typename Op>
std::string get_kernel_name([[maybe_unused]] const Op &op, bool stride, bool global_kernel, bool pass_through_threads = false) {
  if constexpr (Op::Rank() == 0) {
    return "matx::detail::matxOpT0Kernel";
  }  
  else if constexpr (Op::Rank() == 1) {
    return global_kernel ? "matx::detail::matxOpT1Kernel" : "matx::detail::matxOpT1KernelBlock";
  }
  else if constexpr (Op::Rank() == 2) {
    if (pass_through_threads) {
      return "matx::detail::matxOpT2KernelBlock2D";
    } else if (stride) {
      return global_kernel ? "matx::detail::matxOpT2StrideKernel" : "matx::detail::matxOpT2StrideKernelBlock";
    } else {
      return global_kernel ? "matx::detail::matxOpT2Kernel" : "matx::detail::matxOpT2KernelBlock";
    }
  }
  else if constexpr (Op::Rank() == 3) {
    if (pass_through_threads) {
      return "matx::detail::matxOpT3KernelBlock2D";
    } else if (stride) {
      return global_kernel ? "matx::detail::matxOpT3StrideKernel" : "matx::detail::matxOpT3StrideKernelBlock";
    } else {
      return global_kernel ? "matx::detail::matxOpT3Kernel" : "matx::detail::matxOpT3KernelBlock";
    }
  }
  else if constexpr (Op::Rank() == 4) {
    if (pass_through_threads) {
      return "matx::detail::matxOpT4KernelBlock2D";
    } else if (stride) {
      return global_kernel ? "matx::detail::matxOpT4StrideKernel" : "matx::detail::matxOpT4StrideKernelBlock";
    } else {
      return global_kernel ? "matx::detail::matxOpT4Kernel" : "matx::detail::matxOpT4KernelBlock";
    }
  }
  else {
    // For ranks > 4, use the TD (Tensor Dynamic) kernel
    return "matx::detail::matxOpTDKernel";
  }

  return "MatXInvalidKernel";
}

template <typename Op>
std::string generate_capability_params_string([[maybe_unused]] const Op &op, ElementsPerThread EPT, bool JIT, int osize, int block_size, bool pass_through_threads = false) {
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

  std::string pass_through_str = pass_through_threads ? "true" : "false";
  
  std::string final_str =  
         "namespace matx { namespace detail {\n"
         "template <ElementsPerThread EPT, bool JIT>\n"
         "struct CapabilityParams {\n"
         "  static constexpr ElementsPerThread ept = EPT;\n"
         "  static constexpr bool jit = JIT;\n"
         "  static constexpr int osize = " + std::to_string(osize) + ";\n"
         "  static constexpr int block_size = " + std::to_string(block_size) + ";\n"
         "  static constexpr bool pass_through_threads = " + pass_through_str + ";\n"
         "};\n"
         "using CurrentCapabilities = CapabilityParams<" + ept_str + ", " + jit_str + ">;\n"
         "} }\n";
   
  return final_str;
}

template <typename Op>
std::string get_all_jit_classes_string(const Op& op) {
  std::unordered_map<std::string, std::string> jit_strings;
  // Query all JIT class strings for the operator and its children
  detail::get_operator_capability<OperatorCapability::JIT_CLASS_QUERY>(op, jit_strings);

  std::string result;
  result += " #include <matx/core/jit_includes.h>\n";
  if (!jit_strings.empty()) {
    result += "namespace matx { namespace detail {\n";
    for (const auto& kv : jit_strings) {
      result += kv.second;
      if (!result.empty() && result.back() != '\n') {
        result += "\n\n\n";
      }
    }
    result += "} // namespace matx\n";
    result += "} // namespace detail\n";
  }

  return result;
}

// Helper function to qualify JIT class names with matx::detail:: namespace
// This is needed for NVRTC's #pragma nv_mangled_name to resolve the types
inline std::string qualify_jit_type_names(const std::string& type_str) {
  std::string result = type_str;
  std::string search_prefix = "JIT";
  std::string replacement = "matx::detail::JIT";
  
  size_t pos = 0;
  while ((pos = result.find(search_prefix, pos)) != std::string::npos) {
    // Check if this JIT is already qualified with matx::detail::
    if (pos >= 14 && result.substr(pos - 14, 14) == "matx::detail::") {
      pos += search_prefix.length();
      continue;
    }
    // Check if it's at the start or preceded by a delimiter (not alphanumeric or ::)
    if (pos == 0 || result[pos - 1] == '<' || result[pos - 1] == ',' || result[pos - 1] == ' ') {
      result.insert(pos, "matx::detail::");
      pos += replacement.length();
    } else {
      pos += search_prefix.length();
    }
  }
  
  return result;
}

template <typename Op, typename SizeArray>
auto nvrtc_compile_and_run([[maybe_unused]] const std::string &name, Op op, const SizeArray &sa, dim3 &blocks, dim3 &threads, ElementsPerThread ept, bool stride, int dynamic_shmem_size, int osize, bool global_kernel, bool pass_through_threads = false) {
  // Pure NVRTC implementation
  // Cache both module and function to prevent resource leaks
  // CUmodule must remain loaded for CUfunction to be valid
  struct CachedKernel {
    CUmodule module;
    CUfunction function;
  };
  static std::unordered_map<std::string, CachedKernel> kernel_cache;
  static std::mutex kernel_cache_mutex;
  
  const auto all_jit_classes_string = get_all_jit_classes_string(op);
  auto capstr = generate_capability_params_string(op, ept, false, osize, threads.x, pass_through_threads);
  const auto kernel_op_type = detail::get_operator_capability<OperatorCapability::JIT_TYPE_QUERY>(op);
  
  std::string kernel_name = get_kernel_name(op, stride, global_kernel, pass_through_threads);
  std::string cache_key = kernel_name + "_" + kernel_op_type;

  MATX_LOG_DEBUG("nvrtc_compile_and_run called with operator type: {}", typeid(op).name());
  
  CUfunction kernel_func;
  std::string lowered_name;
  const auto cubin_filename = detail::GetCache().TypeStringToFilename(kernel_op_type);
  
  // Check if kernel is already compiled and cached in memory
  {
    std::lock_guard<std::mutex> lock(kernel_cache_mutex);
    auto it = kernel_cache.find(cache_key);
    if (it != kernel_cache.end()) {
      // Found in memory cache - use cached function (module is already loaded)
      kernel_func = it->second.function;
      goto launch_kernel;
    }
  }
  
  // Not in memory cache, check disk cache (outside lock to minimize critical section)
  {
    auto cached_cubin_ptr = detail::GetCache().GetLTOIRCachedBytes(cubin_filename);
    
    if (cached_cubin_ptr != nullptr) {
      // Found cached cubin on disk, try to load metadata
      lowered_name = detail::GetCache().GetLTOIRMetadata(cubin_filename);
      
      if (!lowered_name.empty()) {
        MATX_LOG_DEBUG("Loading cached kernel for type: {}", kernel_op_type);
        MATX_LOG_DEBUG("Cached lowered name: {}", lowered_name);
        
        // Load the cached cubin into a CUDA module
        CUmodule module;
        CUDA_CHECK(cuModuleLoadDataEx(&module, cached_cubin_ptr->data, 0, nullptr, nullptr));
        
        // Get kernel function using the cached lowered name
        CUDA_CHECK(cuModuleGetFunction(&kernel_func, module, lowered_name.c_str()));
        
        // Cache both module and function to prevent resource leak
        // Module must stay loaded for function to remain valid
        {
          std::lock_guard<std::mutex> lock(kernel_cache_mutex);
          kernel_cache[cache_key] = CachedKernel{module, kernel_func};
        }
        
        // Skip compilation since we loaded from cache
        goto launch_kernel;
      } else {
        MATX_LOG_DEBUG("Found cached cubin but no metadata, recompiling");
      }
    }
    
    MATX_LOG_DEBUG("Compiling kernel with NVRTC for type: {}", kernel_op_type);
    
    // Read jit_includes.h content
    std::string jit_includes_path = get_jit_includes_path();
    std::string jit_includes_content = read_file_contents(jit_includes_path);
    //MATX_ASSERT_STR(jit_includes_content == "", matxInvalidParameter, "Failed to read jit_includes.h");
    
    // Construct the main kernel source that includes headers by name
    // This matches the Jitify approach where headers are included via -include directives
    std::string main_source = std::string(matxKernelStr);
    
    // Prepare headers array: jit_includes.h, matx_generated_code_hdr (capstr), matx_class_strings
    const int numHeaders = 3;
    const char* headers[numHeaders] = {
      jit_includes_content.c_str(),
      capstr.c_str(),
      all_jit_classes_string.c_str()
    };
    const char* includeNames[numHeaders] = {
      "matx/core/jit_includes.h",
      "matx_generated_code_hdr",
      "matx_class_strings"
    };

    // Print the contents of each include for debugging
    MATX_LOG_TRACE("jit_includes.h content:\n{}", jit_includes_content);
    MATX_LOG_TRACE("matx_generated_code_hdr content:\n{}", capstr);
    MATX_LOG_TRACE("matx_class_strings content:\n{}", all_jit_classes_string);
    
    // Create NVRTC program with headers
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, main_source.c_str(), "matx_kernel.cu", 
                                   numHeaders, headers, includeNames));
    
    // Add name expression to get the proper mangled name after compilation
    // Qualify JIT class names with matx::detail:: namespace so NVRTC can resolve them
    // Note: The matxKernelStr kernels only take ONE template parameter (Op), not two
    std::string qualified_kernel_op_type = qualify_jit_type_names(kernel_op_type);
    std::string kernel_name_expr = kernel_name + "<" + qualified_kernel_op_type + ">";
    MATX_LOG_DEBUG("Kernel name expression: {}", kernel_name_expr);
    NVRTC_CHECK(nvrtcAddNameExpression(prog, kernel_name_expr.c_str()));
    
    // Get compilation options
    auto options = get_preprocessor_options();
    
    // Add -include directives for the generated headers (matching Jitify behavior)
    // IMPORTANT: Include jit_includes.h FIRST to ensure all base types are defined
    options.push_back("-include=matx/core/jit_includes.h");
    options.push_back("-include=matx_generated_code_hdr");
    options.push_back("-include=matx_class_strings");
    options.push_back("-default-device");
    options.push_back("--relocatable-device-code=true");
    options.push_back("-dlto");


    std::vector<const char*> opts;
    for (const auto& opt : options) {
      opts.push_back(opt.c_str());
    }
    // Compile the program
    nvrtcResult compile_result = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    
    // Get compilation log
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
      std::vector<char> log(log_size);
      NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
      MATX_LOG_DEBUG("NVRTC Compilation log:\n{}", log.data());
    }
    
    if (compile_result != NVRTC_SUCCESS) {
      MATX_LOG_ERROR("NVRTC compilation failed!");
      nvrtcDestroyProgram(&prog);
      MATX_THROW(matxInvalidParameter, "NVRTC compilation failed");
    }
    
    // Get the lowered (mangled) kernel name
    const char* lowered_name_ptr;
    NVRTC_CHECK(nvrtcGetLoweredName(prog, kernel_name_expr.c_str(), &lowered_name_ptr));
    // Copy the lowered name before destroying the program
    lowered_name = std::string(lowered_name_ptr);
    MATX_LOG_DEBUG("Lowered kernel name: {}", lowered_name);

    // Compile any LTO-IR required for expression:
    auto ltoir_query_input = detail::LTOIRQueryInput{};
    ltoir_query_input.ept = ept;
    auto ltoir_result = detail::get_operator_capability<OperatorCapability::GENERATE_LTOIR>(op, ltoir_query_input);
    MATX_LOG_DEBUG("LTOIR capability result: {}", ltoir_result);          
    
    size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &lto_size));
    std::vector<char> compiled_code(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog, compiled_code.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));


    // Link and LTO-IR files if needed
    nvJitLinkHandle handle {};
    std::vector<std::string> link_options = { "-lto", std::string("-arch=sm_") + std::string(NVRTC_CUDA_ARCH) };

    std::vector<const char*> lto_opts;
    for (const auto& o : link_options) {
        lto_opts.emplace_back(o.c_str());
    }
    NVJITLINK_CHECK(handle, nvJitLinkCreate(&handle, static_cast<int>(lto_opts.size()), lto_opts.data()));

    // First add all our LTO-IR from the operator
    for (const auto& lto : ltoir_query_input.ltoir_symbols) {
      const auto ltoir_ptr = detail::GetCache().GetLTOIRCachedBytes(lto);
      if (ltoir_ptr == nullptr) {
        std::string error_msg = "LTOIR not found in cache: " + lto;
        MATX_THROW(matxInvalidParameter, error_msg);
      }

      MATX_LOG_TRACE("Adding LTOIR for symbol {}, size={} bytes, first 4 bytes: {:02x} {:02x} {:02x} {:02x}", 
             lto, ltoir_ptr->length, 
             static_cast<unsigned char>(ltoir_ptr->data[0]), 
             static_cast<unsigned char>(ltoir_ptr->data[1]), 
             static_cast<unsigned char>(ltoir_ptr->data[2]), 
             static_cast<unsigned char>(ltoir_ptr->data[3]));
      
      // Validate that the data looks like LTOIR (LLVM bitcode typically starts with 'BC')
      if (ltoir_ptr->length < 4) {
        MATX_LOG_WARN("LTOIR data for {} is too small ({} bytes)", lto, ltoir_ptr->length);
      }
      
      NVJITLINK_CHECK(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoir_ptr->data, ltoir_ptr->length, lto.c_str()));
    }    

    // Add main program LTO-IR
    NVJITLINK_CHECK(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, compiled_code.data(), lto_size, "main"));
    NVJITLINK_CHECK(handle, nvJitLinkComplete(handle));

    size_t cubin_size = 0;
    NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
    std::vector<char> cubin(cubin_size);
    NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubin(handle, cubin.data()));
    NVJITLINK_CHECK(handle, nvJitLinkDestroy(&handle));

    // Store the entire linked kernel to the cache along with the lowered name
    detail::GetCache().StoreLTOIRCachedBytes(cubin_filename, static_cast<const char*>(cubin.data()), cubin_size);
    detail::GetCache().StoreLTOIRMetadata(cubin_filename, lowered_name);

    
    // Load LTO-IR into CUDA module
    CUmodule module;
    CUDA_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr));
    
    // Get kernel function using the lowered name
    CUDA_CHECK(cuModuleGetFunction(&kernel_func, module, lowered_name.c_str()));
    
    // Cache both module and function to prevent resource leak
    // Module must stay loaded for function to remain valid
    {
      std::lock_guard<std::mutex> lock(kernel_cache_mutex);
      kernel_cache[cache_key] = CachedKernel{module, kernel_func};
    }
  }
  
launch_kernel:
  // Get device attributes
  int device;
  CUDA_RT_CHECK(cudaGetDevice(&device));
  int static_shared_size;
  CUDA_RT_CHECK(cudaDeviceGetAttribute(&static_shared_size, cudaDevAttrMaxSharedMemoryPerBlock, device));

  // Set dynamic shared memory if needed
  if (dynamic_shmem_size > static_shared_size) {
    MATX_LOG_DEBUG("Requested dynamic shared memory ({} bytes) exceeds static shared memory ({}) for kernel, using dynamic shared memory", 
                   dynamic_shmem_size, static_shared_size);
    CUDA_CHECK(cuFuncSetAttribute(kernel_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_shmem_size));
  }

  auto storage = op.ToJITStorage();
  
  // Prepare kernel arguments
  if constexpr (Op::Rank() > 4) {
    // ND kernel: matxOpTDKernel(Op op, const cuda::std::array<matx::index_t, Op::Rank()> sizes, matx::index_t mult)
    // mult is the product of all sizes except the first
    index_t mult = cuda::std::accumulate(cuda::std::begin(sa) + 1, cuda::std::end(sa), 1, cuda::std::multiplies<index_t>());
    
    void* args[3];
    args[0] = &storage;
    args[1] = const_cast<void*>(reinterpret_cast<const void*>(&sa));
    args[2] = &mult;
    
    MATX_LOG_DEBUG("Launching kernel with grid=({}, {}, {}), block=({}, {}, {}), dynamic_shmem_size={} bytes",
                   blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, dynamic_shmem_size);
    // Launch kernel
    CUDA_CHECK(cuLaunchKernel(kernel_func,
                              blocks.x, blocks.y, blocks.z,
                              threads.x, threads.y, threads.z,
                              dynamic_shmem_size,
                              nullptr,  // stream
                              args,
                              nullptr));
  }
  else {
    // Rank 0-4 kernels: Pass individual size parameters
    void* args[Op::Rank() + 1];
    args[0] = &storage;
    if constexpr (Op::Rank() >= 1) {
      args[1] = const_cast<void*>(reinterpret_cast<const void*>(&sa[0]));
    }
    if constexpr (Op::Rank() >= 2) {
      args[2] = const_cast<void*>(reinterpret_cast<const void*>(&sa[1]));
    }
    if constexpr (Op::Rank() >= 3) {
      args[3] = const_cast<void*>(reinterpret_cast<const void*>(&sa[2]));
    }
    if constexpr (Op::Rank() == 4) {
      args[4] = const_cast<void*>(reinterpret_cast<const void*>(&sa[3]));
    }
    
    MATX_LOG_DEBUG("Launching kernel with grid=({}, {}, {}), block=({}, {}, {}), dynamic_shmem_size={} bytes",
                   blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, dynamic_shmem_size);
    // Launch kernel
    CUDA_CHECK(cuLaunchKernel(kernel_func,
                              blocks.x, blocks.y, blocks.z,
                              threads.x, threads.y, threads.z,
                              dynamic_shmem_size,
                              nullptr,  // stream
                              args,
                              nullptr));
  }
}

}
}
#endif