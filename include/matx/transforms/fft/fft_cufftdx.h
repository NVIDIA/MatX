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

#define FFT_DX_FUNC_PREFIX "fft_cufftdx_func"

#ifdef __CUDACC__

#include "matx/core/operator_options.h"
#include "matx/core/capabilities.h"
#include "matx/core/jit_param_structs.h"
#ifndef __CUDACC_RTC__
#include <libcufftdx.h>
#endif

#define LIBMATHDX_CHECK(ans)                                                                                           \
  do {                                                                                                               \
    commondxStatusType result = (ans);                                                                             \
    if (result != commondxStatusType::COMMONDX_SUCCESS) {                                                          \
      MATX_THROW(matxCufftError, "cuFFTDx failed");                                                                                       \
    }                                                                                                              \
  } while (0)


namespace matx {
  namespace detail {

  template <typename T> 
  struct fft_precision {
    using type = T;
  };

  template <> 
  struct fft_precision<cuda::std::complex<float>> {
    using type = float;
  };  

  template <> 
  struct fft_precision<cuda::std::complex<double>> {
    using type = double;
  };  
  
  template <> 
  struct fft_precision<matxFp16> {
    using type = __half;
  };
  
  template <> 
  struct fft_precision<matxBf16> {
    using type = __nv_bfloat16;
  };
  
  template <> 
  struct fft_precision<matxFp16Complex> {
    using type = __half;
  };
  
  template <> 
  struct fft_precision<matxBf16Complex> {
    using type = __nv_bfloat16;
  };    



#if defined(MATX_EN_MATHDX) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)
  template <typename InputType>
  class cuFFTDxHelper {
    public:
      static cufftdxDescriptor Init(index_t fft_size, FFTType fft_type, FFTDirection direction) {
        cufftdxDescriptor h_;
        LIBMATHDX_CHECK(cufftdxCreateDescriptor(&h_));
        // CUFFTDX_API_LMEM means the function will be of signature:
        //     void(value_type*, value_type*)
        //   with the first argument being local memory ("registers"), with each thread holding "EPT" elements
        //   and the second being a pointer to a shared memory scratch buffer
        // CUFFTDX_API_SMEM would mean that the function will be of signature:
        //     void(value_type*)
        //   and takes a shared memory pointer with all the elements laid out in natural order
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_API, cufftdxApi::CUFFTDX_API_SMEM));

        LIBMATHDX_CHECK(
            cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));

        // FFT size
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_SIZE, fft_size));

        cufftdxType cufftdx_type;
        if (fft_type == FFTType::C2C) {
          cufftdx_type = cufftdxType::CUFFTDX_TYPE_C2C;
        } else if (fft_type == FFTType::C2R) {
          cufftdx_type = cufftdxType::CUFFTDX_TYPE_C2R;
        } else if (fft_type == FFTType::R2C) {
          cufftdx_type = cufftdxType::CUFFTDX_TYPE_R2C;
        } else {
          MATX_THROW(matxInvalidParameter, "Unsupported FFT type for cuFFTDx");
        }

        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_TYPE, cufftdx_type));

        cufftdxDirection cufftdx_direction;
        if (direction == FFTDirection::FORWARD) {
          cufftdx_direction = cufftdxDirection::CUFFTDX_DIRECTION_FORWARD;
        } else {
          cufftdx_direction = cufftdxDirection::CUFFTDX_DIRECTION_INVERSE;
        }
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_DIRECTION, cufftdx_direction));

        if constexpr (cuda::std::is_same_v<InputType, matxBf16Complex> || cuda::std::is_same_v<InputType, matxFp16Complex> || 
                      cuda::std::is_same_v<InputType, matxBf16> || cuda::std::is_same_v<InputType, matxFp16>) {
          LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F16));
        } else if constexpr (cuda::std::is_same_v<InputType, cuda::std::complex<float>> || cuda::std::is_same_v<InputType, float>) {
          LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F32));
        } else if constexpr (cuda::std::is_same_v<InputType, cuda::std::complex<double>> || cuda::std::is_same_v<InputType, double>) {
          LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F64));
        } else {
          MATX_THROW(matxInvalidParameter, "Unsupported input type for cuFFTDx");
        }

        int major = 0;
        int minor = 0;
        int device;
        cudaGetDevice(&device);            
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        int cc = major * 100 + minor;
    
        // Compute capability to target
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_SM, cc));
        return h_;
      }

      static std::string GetSymbolName(index_t fft_size, FFTType fft_type, FFTDirection direction, int cc) {
        std::string symbol_name;
        symbol_name += std::to_string(fft_size);
        symbol_name += "_T";
        symbol_name += std::to_string(static_cast<int>(fft_type));
        symbol_name += "_D";
        symbol_name += std::to_string(static_cast<int>(direction));
        symbol_name += "_CC";
        symbol_name += std::to_string(cc);

        // Add CUDA version to the symbol name
#if defined(CUDA_VERSION)
        symbol_name += "_CUDA";
        symbol_name += std::to_string(CUDART_VERSION);
#else
        symbol_name += "_CUDAUNKNOWN";
#endif

        //symbol_name += ".ltoir";
        
        return symbol_name;
      }

      static bool IsSupported(index_t fft_size, FFTType fft_type, FFTDirection direction) {
        auto handle = Init(fft_size, fft_type, direction);
        int valid = -1;
        LIBMATHDX_CHECK(cufftdxIsSupported(handle, &valid));
        return static_cast<bool>(valid);
      }

      static int GetShmRequired(index_t fft_size, FFTType fft_type, FFTDirection direction, ElementsPerThread ept) {
        auto handle = Init(fft_size, fft_type, direction);
        // SHM size is based on EPT, so set the one we're using here. Eventually make these uncoupled
        long long int ept_int = static_cast<long long int>(ept);
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(handle, CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD, ept_int));

        long long int shared_memory_size = 0;
        LIBMATHDX_CHECK(cufftdxGetTraitInt64(handle, CUFFTDX_TRAIT_SHARED_MEMORY_SIZE, &shared_memory_size));
        return static_cast<int>(shared_memory_size);
      }


      static auto GetEPTs(index_t fft_size, FFTType fft_type, FFTDirection direction) {
        auto handle = Init(fft_size, fft_type, direction);
        // ElementsPerThread type is needed for EPT capability, but cuFFTDx uses long long int for EPTs
        cufftdxKnobType_t knobs = CUFFTDX_KNOB_ELEMENTS_PER_THREAD;
        size_t num_epts = 0;
        LIBMATHDX_CHECK(cufftdxGetKnobInt64Size(handle, 1, &knobs, &num_epts));
        std::vector<long long int> epts(num_epts, 0);
        LIBMATHDX_CHECK(cufftdxGetKnobInt64s(handle, 1, &knobs, epts.size(), epts.data()));      

        if (epts.size() == 0) {
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return my_cap;
        }

        return cuda::std::array<ElementsPerThread, 2>{static_cast<ElementsPerThread>(*std::min_element(epts.begin(), epts.end())),
                                                      static_cast<ElementsPerThread>(*std::max_element(epts.begin(), epts.end()))};
      }

      static int GetBlockDim(index_t fft_size, FFTType fft_type, FFTDirection direction, ElementsPerThread ept) {
        auto handle = Init(fft_size, fft_type, direction);
        cuda::std::array<long long int, 3> block_dim = { 0, 0, 0 };
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(handle, CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD, static_cast<long long int>(ept)));

        LIBMATHDX_CHECK(
            cufftdxGetTraitInt64s(handle, cufftdxTraitType::CUFFTDX_TRAIT_BLOCK_DIM, block_dim.size(), block_dim.data()));
        return static_cast<int>(block_dim[0]);
      }

      static bool GenerateLTOIR(index_t fft_size, FFTType fft_type, FFTDirection direction, ElementsPerThread ept, std::vector<std::string> &ltoir_symbols) {
                // Specify arch to compile to. This should eventually be pulled from the handle, but there's no way to do that currently
        int major = 0;
        int minor = 0;
        int device;
        cudaGetDevice(&device);            
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        long long int cc = major * 100 + minor;

        const auto symbol_name = std::string(FFT_DX_FUNC_PREFIX) + "_" + GetSymbolName(fft_size, fft_type, direction, static_cast<int>(cc));
        ltoir_symbols.push_back(symbol_name);
        LTOIRData ltoir;

        if (detail::GetCache().GetLTOIRCachedBytes(symbol_name) != nullptr) {
          printf("LTOIR found in cache with size %zd\n", detail::GetCache().GetLTOIRCachedBytes(symbol_name)->length);
          return true;
        }

        printf("fft_size %lld fft_type %d direction %d cc %d symbol_name %s\n", fft_size, static_cast<int>(fft_type), static_cast<int>(direction), static_cast<int>(cc), symbol_name.c_str());
        auto handle = Init(fft_size, fft_type, direction);
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(handle, CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD, static_cast<long long int>(ept)));

        LIBMATHDX_CHECK(cufftdxSetOptionStr(handle, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, symbol_name.c_str())); 

        commondxCode code;
        LIBMATHDX_CHECK(commondxCreateCode(&code));

        LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, cc));
        LIBMATHDX_CHECK(cufftdxFinalizeCode(code, handle));

        LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &ltoir.length));
        ltoir.data = static_cast<char*>(malloc(ltoir.length));
        MATX_ASSERT_STR(ltoir.data != nullptr, matxInvalidParameter, "Failed to allocate LTO IR data");

        LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, ltoir.length, ltoir.data));

        printf("Function %s\n", symbol_name.c_str());        
        printf("LTOIR size %zd\n", ltoir.length);
        printf("LTOIR first 8 bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n",
               static_cast<unsigned char>(ltoir.data[0]),
               static_cast<unsigned char>(ltoir.data[1]),
               static_cast<unsigned char>(ltoir.data[2]),
               static_cast<unsigned char>(ltoir.data[3]),
               static_cast<unsigned char>(ltoir.data[4]),
               static_cast<unsigned char>(ltoir.data[5]),
               static_cast<unsigned char>(ltoir.data[6]),
               static_cast<unsigned char>(ltoir.data[7]));
        
        // Check LTOIR format - note that cuFFTDx may generate various formats
        // (LLVM bitcode 'BC', NVVM IR, or other LTOIR formats)
        if (ltoir.length >= 4) {
          bool is_llvm_bc = (static_cast<unsigned char>(ltoir.data[0]) == 0x42 && 
                            static_cast<unsigned char>(ltoir.data[1]) == 0x43);
          if (is_llvm_bc) {
            printf("LTOIR format: LLVM bitcode (BC)\n");
          } else {
            printf("LTOIR format: Other (first bytes: %02x %02x %02x %02x)\n",
                   static_cast<unsigned char>(ltoir.data[0]),
                   static_cast<unsigned char>(ltoir.data[1]),
                   static_cast<unsigned char>(ltoir.data[2]),
                   static_cast<unsigned char>(ltoir.data[3]));
          }
        }

        detail::GetCache().StoreLTOIRCachedBytes(symbol_name, ltoir.data, ltoir.length);
        
        // CRITICAL: Set to nullptr after transferring ownership to cache to prevent double-free
        // The cache now owns this memory, so we must not let LTOIRData destructor free it
        ltoir.data = nullptr;
        ltoir.length = 0;
        
        LIBMATHDX_CHECK(commondxDestroyCode(code));
    
        return true;
      }

      static const char* GetFuncStr() {
          return  R"(
            using input_type = %s;
            [[maybe_unused]] static constexpr int fft_size = %d;
            [[maybe_unused]] static constexpr int fft_forward = %d;
            [[maybe_unused]] static constexpr int fft_norm = %d;
            [[maybe_unused]] static constexpr int fft_type = %d;
      
            // If it's a half precision type we don't use value_type
            using input_type_converted = typename detail::convert_matx_type_t<input_type>;
      
            extern __shared__  Vector<input_type_converted, static_cast<int>(CapType::ept)> thread_data[];
            thread_data[threadIdx.x] = a_.template operator()<CapType>(indices...);
            __syncthreads();
      
            %s(reinterpret_cast<input_type_converted*>(&thread_data[0]));
      
            if constexpr (fft_norm == 2) { // ORTHO
              #pragma unroll
              for (int i = 0; i < static_cast<int>(CapType::ept); i++) {        
                thread_data[threadIdx.x].data[i] = thread_data[threadIdx.x].data[i] * static_cast<precision>(1.f / cuda::std::sqrt(fft_size));
              }
            }
            else if constexpr ((fft_norm == 1 && fft_forward) || (fft_norm == 0 && !fft_forward)) {
              #pragma unroll
              for (int i = 0; i < static_cast<int>(CapType::ept); i++) {        
                thread_data[threadIdx.x].data[i] = thread_data[threadIdx.x].data[i] * static_cast<precision>(1.f / fft_size);
              }
            }
      
            return thread_data[threadIdx.x];  
        )";        
      }
  };


#endif

  } // namespace detail
} // namespace matx

#endif