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
#include <algorithm>
#include <libcufftdx.h>

#define LIBMATHDX_CHECK(ans)                                                                                           \
  do {                                                                                                               \
    commondxStatusType result = (ans);                                                                             \
    MATX_ASSERT_STR_EXP(result, commondxStatusType::COMMONDX_SUCCESS, matxCufftError, "cuFFTDx failed");\
  } while (0)


namespace matx {
  namespace detail {

  enum class cuFFTDxMethod {
    REGISTER,
    SHARED
  };


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
      cuFFTDxMethod method_;
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
      cuFFTDxMethod get_method() const { return method_; }

      // Setters
      void set_fft_size(index_t size) { fft_size_ = size; }
      void set_fft_type(FFTType type) { fft_type_ = type; }
      void set_direction(FFTDirection dir) { direction_ = dir; }
      void set_current_elements_per_thread(ElementsPerThread ept) { current_elements_per_thread_ = ept; }
      void set_ffts_per_block(int ffts_per_block) { ffts_per_block_ = ffts_per_block; }
      void set_cc(int cc) { cc_ = cc; }
      void set_contiguous_input(bool contiguous_input) { contiguous_input_ = contiguous_input; }
      void set_method(cuFFTDxMethod method) { method_ = method; }
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      cufftdxDescriptor GeneratePlan() const {
        cufftdxDescriptor h_;
        LIBMATHDX_CHECK(cufftdxCreateDescriptor(&h_));

        // if (fft_size_ <= 32) {
        //   LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_API, cufftdxApi::CUFFTDX_API_LMEM));   
        //   method_ = cuFFTDxMethod::REGISTER;
        // } else {
          LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_API, method_ == cuFFTDxMethod::REGISTER ? cufftdxApi::CUFFTDX_API_LMEM : cufftdxApi::CUFFTDX_API_SMEM));
        //}
        
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
        symbol_name += "_E";
        symbol_name += std::to_string(static_cast<int>(current_elements_per_thread_));
        symbol_name += "_FPB";
        symbol_name += std::to_string(ffts_per_block_);

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

      template <typename OpType>
      bool CheckJITSizeAndTypeRequirements() const {
        using OpInputType = typename OpType::value_type;
        
        // Only support power-of-2 FFT sizes for JIT support
        if ((fft_size_ & (fft_size_ - 1)) != 0 || fft_size_ == 0) {
          return false;
        }
        
        // No half support in MatX for fusion yet
        if constexpr (is_complex_half_v<OpInputType>) {
          return false;
        }
        
        // Only support C2C for JIT support
        if constexpr (!is_complex_v<OpInputType>) {
          return false;
        }
        
        return true;
      }

      int GetShmRequired() const {
        auto handle = GeneratePlan();

        long long int shared_memory_size = 0;
        LIBMATHDX_CHECK(cufftdxGetTraitInt64(handle, CUFFTDX_TRAIT_SHARED_MEMORY_SIZE, &shared_memory_size));
        MATX_LOG_DEBUG("Shared memory size from cuFFTDx: {}", shared_memory_size);
        if (method_ == cuFFTDxMethod::SHARED) {
          // Add in the input/output shm
          shared_memory_size = static_cast<long long int>(fft_size_) * sizeof(InputType) * static_cast<long long int>(ffts_per_block_) + shared_memory_size;
        }

        MATX_LOG_DEBUG("Shared memory size computed: {}", shared_memory_size);
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

        for (size_t i = 0; i < epts.size(); ++i) {
          MATX_LOG_DEBUG("cuFFTDx EPT[{}]: {}", i, epts[i]);
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

        MATX_LOG_DEBUG("Getting FFTs per block {} elements per thread {} and fft_size {}", sfpb, static_cast<int>(current_elements_per_thread_), fft_size_);
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

        if (!detail::GetCache().StoreLTOIRCachedBytes(symbol_name, ltoir.data, ltoir.length)) {
          free(ltoir.data);
          MATX_LOG_ERROR("Failed to store LTOIR cached bytes for: {}", symbol_name);
          return false;
        }
        
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
          // result += R"(;
          // [[maybe_unused]] static constexpr bool register_api = )";
          // result += std::to_string(method_ == cuFFTDxMethod::REGISTER) ? "true" : "false";          
          result += R"(;

            const int local_fft_id = threadIdx.y;
      
            // If it's a half precision type we don't use value_type
            using input_type_converted = typename detail::convert_matx_type_t<input_type>;
            using precision = typename detail::inner_precision<input_type>::type;
            constexpr int total_threads_per_block = CapType::block_size * ffts_per_block;

            // using BlockLoadToShm = cub::detail::BlockLoadToShared<total_threads_per_block>;
            // using TempStorage       = BlockLoadToShm::TempStorage;
             using VecType = Vector<input_type_converted, static_cast<int>(CapType::ept)>;
            //constexpr size_t to_copy   = sizeof(input_type) * static_cast<int>(CapType::ept) * static_cast<int>(total_threads_per_block);

            // constexpr int buff_align = BlockLoadToShm::template SharedBufferAlignBytes<VecType>();
            // constexpr int buff_size  = BlockLoadToShm::template SharedBufferSizeBytes<VecType>(total_threads_per_block);            
      
            extern  __shared__  VecType thread_data[];

            // if constexpr (contiguous_input) {
            // //   __shared__ TempStorage temp_storage;
            // //   BlockLoadToShm load_to_shared(temp_storage);
            // //   cuda::std::span<const VecType> gmem_src(reinterpret_cast<const VecType*>(a_.template data_ptr<CapType>(blockIdx.x, blockDim.x * blockDim.y)), total_threads_per_block);
            // //   cuda::std::span<VecType> smem_dst_buff(thread_data, total_threads_per_block);
            // //   auto smem_dst = load_to_shared.CopyAsync(smem_dst_buff, gmem_src);
            // //   load_to_shared.Commit();
            // //   load_to_shared.Wait();
            //   cuda::barrier<cuda::thread_scope_block> bar;
            //   init(&bar, 1);            
            //   cuda::memcpy_async(thread_data,     reinterpret_cast<const VecType*>(a_.template data_ptr<CapType>(blockIdx.x, blockDim.x * blockDim.y)),      to_copy, bar);              
            //   bar.arrive_and_wait();
            // }
            // else {
              thread_data[local_fft_id * blockDim.x + threadIdx.x] = a_.template operator()<CapType>(indices...);
              __syncthreads();         
            //}

      
            )";
          result += fft_func_name;
          result += R"((reinterpret_cast<input_type_converted*>(&thread_data[0]));
          __syncthreads();      

          if constexpr (fft_norm == 2) { // ORTHO
            #pragma unroll
            for (int i = 0; i < static_cast<int>(CapType::ept); i++) {        
              thread_data[local_fft_id * blockDim.x + threadIdx.x].data[i] = thread_data[local_fft_id * blockDim.x + threadIdx.x].data[i] * static_cast<precision>(1.f) / static_cast<precision>(cuda::std::sqrt(fft_size));
            }
          }
          else if constexpr ((fft_norm == 1 && fft_forward) || (fft_norm == 0 && !fft_forward)) {
            #pragma unroll
            for (int i = 0; i < static_cast<int>(CapType::ept); i++) {        
              thread_data[local_fft_id * blockDim.x + threadIdx.x].data[i] = thread_data[local_fft_id * blockDim.x + threadIdx.x].data[i] * static_cast<precision>(1.f) / static_cast<precision>(fft_size);
            }
          }

          return thread_data[local_fft_id * blockDim.x + threadIdx.x];  
        )";

          return result;
      }
#endif
  };

  template <typename InputType>
  class cuFFTDx2DHelper {
    private:
      index_t fft_size_x_ = 0;
      index_t fft_size_y_ = 0;
      FFTType fft_type_ = FFTType::C2C;
      FFTDirection direction_ = FFTDirection::FORWARD;
      int cc_ = 0;
      cuFFTDxHelper<InputType> fft_x_helper_;
      cuFFTDxHelper<InputType> fft_y_helper_;

      static ElementsPerThread IntToElementsPerThread(int ept) {
        switch (ept) {
          case 1: return ElementsPerThread::ONE;
          case 2: return ElementsPerThread::TWO;
          case 4: return ElementsPerThread::FOUR;
          case 8: return ElementsPerThread::EIGHT;
          case 16: return ElementsPerThread::SIXTEEN;
          case 32: return ElementsPerThread::THIRTY_TWO;
          default: return ElementsPerThread::INVALID;
        }
      }

      void Configure1DHelpers() {
        if (fft_size_x_ <= 0 || fft_size_y_ <= 0 || cc_ <= 0) {
          return;
        }

        fft_x_helper_.set_fft_size(fft_size_x_);
        fft_x_helper_.set_fft_type(fft_type_);
        fft_x_helper_.set_direction(direction_);
        fft_x_helper_.set_ffts_per_block(static_cast<int>(fft_size_y_));
        fft_x_helper_.set_cc(cc_);
        fft_x_helper_.set_contiguous_input(false);
        fft_x_helper_.set_method(cuFFTDxMethod::SHARED);

        fft_y_helper_.set_fft_size(fft_size_y_);
        fft_y_helper_.set_fft_type(fft_type_);
        fft_y_helper_.set_direction(direction_);
        fft_y_helper_.set_ffts_per_block(static_cast<int>(fft_size_x_));
        fft_y_helper_.set_cc(cc_);
        fft_y_helper_.set_contiguous_input(false);
        fft_y_helper_.set_method(cuFFTDxMethod::SHARED);
      }

    public:
      cuFFTDx2DHelper() = default;

      index_t get_fft_size_x() const { return fft_size_x_; }
      index_t get_fft_size_y() const { return fft_size_y_; }
      FFTType get_fft_type() const { return fft_type_; }
      FFTDirection get_direction() const { return direction_; }
      int get_cc() const { return cc_; }

      void set_fft_size_x(index_t size) { fft_size_x_ = size; Configure1DHelpers(); }
      void set_fft_size_y(index_t size) { fft_size_y_ = size; Configure1DHelpers(); }
      void set_fft_type(FFTType type) { fft_type_ = type; Configure1DHelpers(); }
      void set_direction(FFTDirection dir) { direction_ = dir; Configure1DHelpers(); }
      void set_cc(int cc) { cc_ = cc; Configure1DHelpers(); }

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      std::string GetSymbolName() const {
        std::string symbol_name;
        symbol_name += std::to_string(fft_size_x_);
        symbol_name += "_";
        symbol_name += std::to_string(fft_size_y_);
        symbol_name += "_T";
        symbol_name += std::to_string(static_cast<int>(fft_type_));
        symbol_name += "_D";
        symbol_name += std::to_string(static_cast<int>(direction_));
        symbol_name += "_CC";
        symbol_name += std::to_string(cc_);

#if defined(CUDA_VERSION)
        symbol_name += "_CUDA";
        symbol_name += std::to_string(CUDART_VERSION);
#else
        symbol_name += "_CUDAUNKNOWN";
#endif

        return symbol_name;
      }

      template <typename OpType>
      bool CheckJITSizeAndTypeRequirements() const {
        using OpInputType = typename OpType::value_type;

        if (fft_type_ != FFTType::C2C) {
          return false;
        }

        if ((fft_size_x_ & (fft_size_x_ - 1)) != 0 || fft_size_x_ == 0 ||
            (fft_size_y_ & (fft_size_y_ - 1)) != 0 || fft_size_y_ == 0) {
          return false;
        }

        // The single-kernel JIT path uses one CUDA block for a complete 2D tile.
        if (fft_size_x_ != fft_size_y_ || (fft_size_x_ * fft_size_y_) > 1024) {
          return false;
        }

        if constexpr (is_complex_half_v<OpInputType> || !is_complex_v<OpInputType>) {
          return false;
        }

        return true;
      }

      bool IsSupported() const {
        return fft_x_helper_.IsSupported() && fft_y_helper_.IsSupported();
      }

      int GetShmRequired() const {
        const auto data_size = static_cast<int64_t>(fft_size_x_) *
                               static_cast<int64_t>(fft_size_y_) *
                               static_cast<int64_t>(sizeof(InputType));
        const auto x_shm = static_cast<int64_t>(fft_x_helper_.GetShmRequired());
        const auto y_shm = static_cast<int64_t>(fft_y_helper_.GetShmRequired());
        const auto extra_shm = std::max<int64_t>(0, std::max(x_shm, y_shm) - data_size);
        const auto total = data_size * 2 + extra_shm;
        MATX_LOG_DEBUG("cuFFTDx 2D shared memory: data={}, scratch={}, extra={}, total={}",
                       data_size, data_size, extra_shm, total);
        return static_cast<int>(total);
      }

      int GetBlockDim() const {
        const auto block_x = fft_x_helper_.GetBlockDim();
        const auto block_y = fft_y_helper_.GetBlockDim();
        if (block_x != block_y) {
          MATX_LOG_DEBUG("cuFFTDx 2D block dims differ: x={}, y={}", block_x, block_y);
          return -1;
        }

        return block_x;
      }

      int GetFFTsPerBlock() const {
        return static_cast<int>(fft_size_y_);
      }

      ElementsPerThread GetElementsPerThread() const {
        const auto block_dim = GetBlockDim();
        if (block_dim <= 0 || fft_size_x_ % block_dim != 0) {
          return ElementsPerThread::INVALID;
        }

        return IntToElementsPerThread(static_cast<int>(fft_size_x_ / block_dim));
      }

      bool GenerateLTOIR(std::set<std::string> &ltoir_symbols) {
        return fft_x_helper_.GenerateLTOIR(ltoir_symbols) &&
               fft_y_helper_.GenerateLTOIR(ltoir_symbols);
      }

      std::string GetXFuncName() {
        return std::string(FFT_DX_FUNC_PREFIX) + "_" + fft_x_helper_.GetSymbolName();
      }

      std::string GetYFuncName() {
        return std::string(FFT_DX_FUNC_PREFIX) + "_" + fft_y_helper_.GetSymbolName();
      }

      std::string GetFuncStr(const std::string &fft_x_func_name,
                             const std::string &fft_y_func_name,
                             int fft_norm,
                             int actual_rank) {
        const int fft_forward = (direction_ == FFTDirection::FORWARD) ? 1 : 0;

        std::string result = R"(
          using input_type = )";
        result += detail::type_to_string<InputType>();
        result += R"(;
          using input_type_converted = typename detail::convert_matx_type_t<input_type>;
          using precision = typename detail::inner_precision<input_type>::type;
          using ScalarCap = typename CapType::scalar_cap;
          [[maybe_unused]] static constexpr int fft_size_x = )";
        result += std::to_string(static_cast<int>(fft_size_x_));
        result += R"(;
          [[maybe_unused]] static constexpr int fft_size_y = )";
        result += std::to_string(static_cast<int>(fft_size_y_));
        result += R"(;
          [[maybe_unused]] static constexpr int fft_forward = )";
        result += std::to_string(fft_forward);
        result += R"(;
          [[maybe_unused]] static constexpr int fft_norm = )";
        result += std::to_string(fft_norm);
        result += R"(;
          [[maybe_unused]] static constexpr int fft_rank = )";
        result += std::to_string(actual_rank);
        result += R"(;
          [[maybe_unused]] static constexpr int fft_elements = fft_size_x * fft_size_y;

          extern __shared__ __align__(16) unsigned char fft2_smem_raw[];
          auto *fft_data = reinterpret_cast<input_type_converted *>(fft2_smem_raw);
          auto *fft_scratch = fft_data + fft_elements;

          const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
          const int total_threads = blockDim.x * blockDim.y * blockDim.z;
          cuda::std::array<index_t, fft_rank> fft_indices{static_cast<index_t>(indices)...};

          for (int elem = tid; elem < fft_elements; elem += total_threads) {
            const int row = elem / fft_size_x;
            const int col = elem - row * fft_size_x;
            fft_indices[fft_rank - 2] = row;
            fft_indices[fft_rank - 1] = col;
            fft_data[elem] = static_cast<input_type_converted>(
              cuda::std::apply([&](auto... args) {
                return a_.template operator()<ScalarCap>(args...);
              }, fft_indices));
          }

          __syncthreads();
          )";
        result += fft_x_func_name;
        result += R"((reinterpret_cast<input_type_converted *>(fft_data));
          __syncthreads();

          for (int elem = tid; elem < fft_elements; elem += total_threads) {
            const int row = elem / fft_size_x;
            const int col = elem - row * fft_size_x;
            fft_scratch[col * fft_size_y + row] = fft_data[row * fft_size_x + col];
          }

          __syncthreads();
          )";
        result += fft_y_func_name;
        result += R"((reinterpret_cast<input_type_converted *>(fft_scratch));
          __syncthreads();

            static constexpr int fft_output_ept = static_cast<int>(CapType::ept);
            const int out_row = static_cast<int>(cuda::std::get<fft_rank - 2>(cuda::std::make_tuple(indices...)));
            const int out_col_base = static_cast<int>(cuda::std::get<fft_rank - 1>(cuda::std::make_tuple(indices...))) * fft_output_ept;

            if constexpr (CapType::ept == ElementsPerThread::ONE) {
              input_type_converted result = fft_scratch[out_col_base * fft_size_y + out_row];
              if constexpr (fft_norm == 2) {
                result = result * static_cast<precision>(1.f) / static_cast<precision>(cuda::std::sqrt(static_cast<precision>(fft_elements)));
              }
              else if constexpr ((fft_norm == 1 && fft_forward) || (fft_norm == 0 && !fft_forward)) {
                result = result * static_cast<precision>(1.f) / static_cast<precision>(fft_elements);
              }

              return static_cast<input_type>(result);
            }
            else {
              Vector<input_type, fft_output_ept> result_vec;
              #pragma unroll
              for (int i = 0; i < fft_output_ept; i++) {
                const int out_col = out_col_base + i;
                input_type_converted result = input_type_converted{};
                if (out_col < fft_size_x) {
                  result = fft_scratch[out_col * fft_size_y + out_row];
                  if constexpr (fft_norm == 2) {
                    result = result * static_cast<precision>(1.f) / static_cast<precision>(cuda::std::sqrt(static_cast<precision>(fft_elements)));
                  }
                  else if constexpr ((fft_norm == 1 && fft_forward) || (fft_norm == 0 && !fft_forward)) {
                    result = result * static_cast<precision>(1.f) / static_cast<precision>(fft_elements);
                  }
                }
                result_vec.data[i] = static_cast<input_type>(result);
              }

              return result_vec;
            }
          )";

        return result;
      }
#endif
  };

  } // namespace detail
} // namespace matx
