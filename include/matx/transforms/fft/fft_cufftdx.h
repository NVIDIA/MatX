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


#include "matx/core/operator_options.h"
#include "matx/core/capabilities.h"
#include "matx/core/log.h"
#include <libcufftdx.h>

#define LIBMATHDX_CHECK(ans)                                                                                           \
  do {                                                                                                               \
    commondxStatusType result = (ans);                                                                             \
    MATX_ASSERT_STR_EXP(result, commondxStatusType::COMMONDX_SUCCESS, matxCufftError, "cuFFTDx failed");\
  } while (0)


namespace matx {
  namespace detail {


  template <typename InputType>
  class cuFFTDxHelper {
    private:
      index_t fft_size_;
      FFTType fft_type_;
      FFTDirection direction_;
      ElementsPerThread current_elements_per_thread_ = ElementsPerThread::INVALID;
      int ffts_per_block_ = 1;
      int cc_;
      bool contiguous_input_;
    public:
      // Constructor
      cuFFTDxHelper() = default;

      // Getters
      index_t get_fft_size() const { return fft_size_; }
      FFTType get_fft_type() const { return fft_type_; }
      FFTDirection get_direction() const { return direction_; }
      ElementsPerThread get_current_elements_per_thread() const { return current_elements_per_thread_; }
      int get_ffts_per_block() const { return ffts_per_block_; }
      int get_cc() const { return cc_; }
      bool get_contiguous_input() const { return contiguous_input_; }


      // Setters
      void set_fft_size(index_t size) { fft_size_ = size; }
      void set_fft_type(FFTType type) { fft_type_ = type; }
      void set_direction(FFTDirection dir) { direction_ = dir; }
      void set_current_elements_per_thread(ElementsPerThread ept) { current_elements_per_thread_ = ept; }
      void set_ffts_per_block(int ffts_per_block) { ffts_per_block_ = ffts_per_block; }
      void set_cc(int cc) { cc_ = cc; }
      void set_contiguous_input(bool contiguous_input) { contiguous_input_ = contiguous_input; }
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      cufftdxDescriptor GeneratePlan() const {
        cufftdxDescriptor h_;
        LIBMATHDX_CHECK(cufftdxCreateDescriptor(&h_));
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_API, cufftdxApi::CUFFTDX_API_SMEM));
        LIBMATHDX_CHECK(
            cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));

        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_SIZE, fft_size_));

        cufftdxType cufftdx_type;
        if (fft_type_ == FFTType::C2C) {
          cufftdx_type = cufftdxType::CUFFTDX_TYPE_C2C;
        } else if (fft_type_ == FFTType::C2R) {
          cufftdx_type = cufftdxType::CUFFTDX_TYPE_C2R;
        } else if (fft_type_ == FFTType::R2C) {
          cufftdx_type = cufftdxType::CUFFTDX_TYPE_R2C;
        } else {
          MATX_THROW(matxInvalidParameter, "Unsupported FFT type for cuFFTDx");
        }

        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_TYPE, cufftdx_type));
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_FFTS_PER_BLOCK, ffts_per_block_));   

        cufftdxDirection cufftdx_direction;
        if (direction_ == FFTDirection::FORWARD) {
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

        if (current_elements_per_thread_ != ElementsPerThread::INVALID) {
          LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD, static_cast<long long int>(current_elements_per_thread_)));
        }

        // Compute capability to target
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_SM, cc_));
        return h_;
      }

      std::string GetSymbolName() {
        std::string symbol_name;
        symbol_name += std::to_string(fft_size_);
        symbol_name += "_T";
        symbol_name += std::to_string(static_cast<int>(fft_type_));
        symbol_name += "_D";
        symbol_name += std::to_string(static_cast<int>(direction_));
        symbol_name += "_CC";
        symbol_name += std::to_string(cc_);

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

      void PrintMembers() const {
        std::cout << "fft_size_ = " << fft_size_ << std::endl;
        std::cout << "fft_type_ = " << static_cast<int>(fft_type_) << std::endl;
        std::cout << "direction_ = " << static_cast<int>(direction_) << std::endl;
        std::cout << "cc_ = " << cc_ << std::endl;
        std::cout << "current_elements_per_thread_ = " << static_cast<int>(current_elements_per_thread_) << std::endl;
        std::cout << "ffts_per_block_ = " << ffts_per_block_ << std::endl;
      }

      bool IsSupported() const {
        auto handle = GeneratePlan();
        int valid = -1;
        LIBMATHDX_CHECK(cufftdxIsSupported(handle, &valid));
        return static_cast<bool>(valid);
      }

      int GetShmRequired() const {
        auto handle = GeneratePlan();

        long long int shared_memory_size = 0;
        LIBMATHDX_CHECK(cufftdxGetTraitInt64(handle, CUFFTDX_TRAIT_SHARED_MEMORY_SIZE, &shared_memory_size));
        shared_memory_size = static_cast<long long int>(current_elements_per_thread_) * sizeof(InputType) * static_cast<long long int>(ffts_per_block_);
        MATX_LOG_DEBUG("Shared memory size {}", shared_memory_size);
        return static_cast<int>(shared_memory_size);
      }


      auto GetEPTs() const {
        auto handle = GeneratePlan();

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

      int GetBlockDim() const {
        auto handle = GeneratePlan();
        cuda::std::array<long long int, 3> block_dim = { 0, 0, 0 };

        LIBMATHDX_CHECK(
            cufftdxGetTraitInt64s(handle, cufftdxTraitType::CUFFTDX_TRAIT_BLOCK_DIM, block_dim.size(), block_dim.data()));
        MATX_LOG_DEBUG("Block dim {} {} {}", block_dim[0], block_dim[1], block_dim[2]);
        return static_cast<int>(block_dim[0]);
      }

      int GetFFTsPerBlock() const {
        auto handle = GeneratePlan();

        // How many FFTs per Block is *suggested*?
        long long int sfpb = 0;
        
        LIBMATHDX_CHECK(cufftdxGetTraitInt64(handle, CUFFTDX_TRAIT_SUGGESTED_FFTS_PER_BLOCK, &sfpb));

        MATX_LOG_DEBUG("Getting FFTs per block {} elements per thread {}", sfpb, static_cast<int>(current_elements_per_thread_));
        return static_cast<int>(sfpb);
      }

      bool GenerateLTOIR(std::set<std::string> &ltoir_symbols) {
        LTOIRData ltoir;
        const auto symbol_name = std::string(FFT_DX_FUNC_PREFIX) + "_" + GetSymbolName();
        ltoir_symbols.insert(symbol_name);        

        if (detail::GetCache().GetLTOIRCachedBytes(symbol_name) != nullptr) {
          MATX_LOG_DEBUG("LTOIR found in cache with size {}", detail::GetCache().GetLTOIRCachedBytes(symbol_name)->length);
          return true;
        }

        auto handle = GeneratePlan();

        LIBMATHDX_CHECK(cufftdxSetOptionStr(handle, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, symbol_name.c_str())); 

        commondxCode code;
        LIBMATHDX_CHECK(commondxCreateCode(&code));

        LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, cc_));
        LIBMATHDX_CHECK(cufftdxFinalizeCode(code, handle));

        LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &ltoir.length));
        ltoir.data = static_cast<char*>(malloc(ltoir.length));
        MATX_ASSERT_STR(ltoir.data != nullptr, matxInvalidParameter, "Failed to allocate LTO IR data");

        LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, ltoir.length, ltoir.data));

        MATX_LOG_DEBUG("Function {}", symbol_name);        
        MATX_LOG_DEBUG("LTOIR size {}", ltoir.length);
        MATX_LOG_TRACE("LTOIR first 8 bytes: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
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
            MATX_LOG_TRACE("LTOIR format: LLVM bitcode (BC)");
          } else {
            MATX_LOG_TRACE("LTOIR format: Other (first bytes: {:02x} {:02x} {:02x} {:02x})",
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

      std::string GetFuncStr(const std::string &fft_func_name, int fft_norm) {
          int fft_forward = (direction_ == FFTDirection::FORWARD) ? 1 : 0;
          
          std::string result = R"(
            using input_type = )";
          result += detail::type_to_string<InputType>();
          result += R"(;
            [[maybe_unused]] static constexpr int fft_size = )";
          result += std::to_string(static_cast<int>(fft_size_));
          result += R"(;
            [[maybe_unused]] static constexpr int ffts_per_block = )";
          result += std::to_string(ffts_per_block_);
          result += R"(;            
            [[maybe_unused]] static constexpr int fft_forward = )";
          result += std::to_string(fft_forward);
          result += R"(;
            [[maybe_unused]] static constexpr int fft_norm = )";
          result += std::to_string(fft_norm);
          result += R"(;
            [[maybe_unused]] static constexpr int fft_type = )";
          result += std::to_string(static_cast<int>(fft_type_));
          result += R"(;
            [[maybe_unused]] static constexpr int contiguous_input = )";
          result += std::to_string(contiguous_input_);
          result += R"(;

            const int local_fft_id = threadIdx.y;
      
            // If it's a half precision type we don't use value_type
            using input_type_converted = typename detail::convert_matx_type_t<input_type>;
            using precision = typename detail::inner_precision<input_type>::type;
            constexpr int total_threads_per_block = CapType::block_size * ffts_per_block;

            // using BlockLoadToShm = cub::detail::BlockLoadToShared<total_threads_per_block>;
            // using TempStorage       = BlockLoadToShm::TempStorage;
             using VecType = Vector<input_type_converted, static_cast<int>(CapType::ept)>;
            constexpr size_t to_copy   = sizeof(input_type) * static_cast<int>(CapType::ept) * static_cast<int>(total_threads_per_block);

            // constexpr int buff_align = BlockLoadToShm::template SharedBufferAlignBytes<VecType>();
            // constexpr int buff_size  = BlockLoadToShm::template SharedBufferSizeBytes<VecType>(total_threads_per_block);            
      
            extern  __shared__  VecType thread_data[];

            if constexpr (contiguous_input) {
            //   __shared__ TempStorage temp_storage;
            //   BlockLoadToShm load_to_shared(temp_storage);
            //   cuda::std::span<const VecType> gmem_src(reinterpret_cast<const VecType*>(a_.template data_ptr<CapType>(blockIdx.x, blockDim.x * blockDim.y)), total_threads_per_block);
            //   cuda::std::span<VecType> smem_dst_buff(thread_data, total_threads_per_block);
            //   auto smem_dst = load_to_shared.CopyAsync(smem_dst_buff, gmem_src);
            //   load_to_shared.Commit();
            //   load_to_shared.Wait();
              cuda::barrier<cuda::thread_scope_block> bar;
              init(&bar, 1);            
              cuda::memcpy_async(thread_data,     reinterpret_cast<const VecType*>(a_.template data_ptr<CapType>(blockIdx.x, blockDim.x * blockDim.y)),      to_copy, bar);              
              bar.arrive_and_wait();
            }
            else {
              thread_data[local_fft_id * blockDim.x + threadIdx.x] = a_.template operator()<CapType>(indices...);
              __syncthreads();            
            }

      
            )";
          result += fft_func_name;
          result += R"((reinterpret_cast<input_type_converted*>(&thread_data[local_fft_id * blockDim.x]));
      
            if constexpr (fft_norm == 2) { // ORTHO
              #pragma unroll
              for (int i = 0; i < static_cast<int>(CapType::ept); i++) {        
                thread_data[local_fft_id * blockDim.x + threadIdx.x].data[i] = thread_data[local_fft_id * blockDim.x + threadIdx.x].data[i] * static_cast<precision>(1.f / cuda::std::sqrt(fft_size));
              }
            }
            else if constexpr ((fft_norm == 1 && fft_forward) || (fft_norm == 0 && !fft_forward)) {
              #pragma unroll
              for (int i = 0; i < static_cast<int>(CapType::ept); i++) {        
                thread_data[local_fft_id * blockDim.x + threadIdx.x].data[i] = thread_data[local_fft_id * blockDim.x + threadIdx.x].data[i] * static_cast<precision>(1.f / fft_size);
              }
            }

            return thread_data[local_fft_id * blockDim.x + threadIdx.x];  
        )";

          return result;
      }
#endif
  };

  } // namespace detail
} // namespace matx

