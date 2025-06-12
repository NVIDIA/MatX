#pragma once

#include "matx/core/operator_options.h"
#include "matx/core/capabilities.h"
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

  /**
   * @brief Checks if a given Block FFT configuration is supported by cuFFTDx.
   *
   * Based on the table and notes at: https://docs.nvidia.com/cuda/cufftdx/requirements_func.html#supported-functionality
   *
   * @param arch_cc The CUDA compute capability (e.g., 70, 75, 80, 86, 90).
   * @param precision The data precision (HALF, FLOAT, DOUBLE).
   * @param fft_size The size of the FFT.
   * @return True if the configuration is supported, false otherwise.
   */
  template <typename T>
  __MATX_INLINE__ __MATX_HOST__  bool IsDxBlockFFTSupported(
      int arch_cc,
      index_t fft_size,
      FFTType fft_type)
  {
      if (fft_size < 0) { // FFT sizes are non-negative.
          return false;
      }

      // Don't fuse non-C2C for now
      if (fft_type != FFTType::C2C) {
        return false;
      }

      // If not covered by the general rule, check "selected FFT sizes" from the normal table.
      // Min size in table is 2.
      if (fft_size < 2) {
            // Only covered if fft_size was 0 or 1 AND met the max_size_fp64_for_arch/2 rule above.
            // Otherwise, sizes < 2 are not in the explicit table ranges.
          return false;
      }

      if constexpr (cuda::std::is_same_v<T, matxFp16Complex> || 
                    cuda::std::is_same_v<T, matxBf16Complex> || 
                    cuda::std::is_same_v<T, cuda::std::complex<float>>) {
            if (arch_cc == 75) return (fft_size <= 16384);
            if (arch_cc == 70 || arch_cc == 72 || arch_cc == 86 || arch_cc == 89) return (fft_size <= 24389);
            if (arch_cc == 80 || arch_cc == 87 || arch_cc == 90) return (fft_size <= 32768);
      }
      else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
            if (arch_cc == 75) return (fft_size <= 8192);
            if (arch_cc == 70 || arch_cc == 72 || arch_cc == 86 || arch_cc == 89) return (fft_size <= 12167);
            if (arch_cc == 80 || arch_cc == 87 || arch_cc == 90) return (fft_size <= 16384);
      }


      return false; // Default if architecture or specific configuration not explicitly supported by table logic
  }

#ifdef __CUDA_ARCH__
  template <typename input_type, typename CapType, typename Op, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ auto RunDxFFT(const Op &op, Is... indices) {
    __syncthreads();

    extern __shared__  Vector<input_type, static_cast<int>(CapType::ept)> thread_data[];

    thread_data[threadIdx.x] = op.template operator()<CapType>(indices...);
    __syncthreads();

    if constexpr (CapType::fft_forward) {
      using FFT = decltype(cufftdx::Block() + cufftdx::Size<CapType::fft_size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                  cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<typename input_type::value_type>() +
                  cufftdx::FFTsPerBlock<1>() + cufftdx::SM<__CUDA_ARCH__>() + cufftdx::ElementsPerThread<static_cast<int>(CapType::ept)>());
      
      FFT().execute(&thread_data[0]);
    }
    else { // IFFT
      using FFT = decltype(cufftdx::Block() + cufftdx::Size<CapType::fft_size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                  cufftdx::Direction<cufftdx::fft_direction::inverse>() + cufftdx::Precision<typename input_type::value_type>() +
                  cufftdx::FFTsPerBlock<1>() + cufftdx::SM<__CUDA_ARCH__>() + cufftdx::ElementsPerThread<static_cast<int>(CapType::ept)>());
      
      FFT().execute(&thread_data[0]);
      // IFFTs get normalized to match Python
      #pragma unroll
      for (int i = 0; i < static_cast<int>(CapType::ept); i++) {
        thread_data[threadIdx.x].data[i] = thread_data[threadIdx.x].data[i] / static_cast<typename input_type::value_type>(CapType::fft_size);
      }
    }
printf("%d %f %f\n", threadIdx.x, thread_data[threadIdx.x].data[0].real(), thread_data[threadIdx.x].data[0].imag());    
    return thread_data[threadIdx.x];  
  }
#endif

#ifndef __CUDACC_RTC__
  template <typename InputType>
  class cuFFTDxHelper {
    public:
      cuFFTDxHelper() = default;
      cuFFTDxHelper(index_t fft_size, FFTType fft_type, FFTDirection direction) {
        LIBMATHDX_CHECK(cufftdxCreateDescriptor(&h_));
        // CUFFTDX_API_LMEM means the function will be of signature:
        //     void(value_type*, value_type*)
        //   with the first argument being local memory ("registers"), with each thread holding "EPT" elements
        //   and the second being a pointer to a shared memory scratch buffer
        // CUFFTDX_API_SMEM would mean that the function will be of signature:
        //     void(value_type*)
        //   and takes a shared memory pointer with all the elements laid out in natural order
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_API, cufftdxApi::CUFFTDX_API_SMEM));
        // COMMONDX_EXECUTION_BLOCK means multiple threads in a block participate in the FFT
        // COMMONDX_EXECUTION_THREAD would mean that each thread computes a single FFT
        LIBMATHDX_CHECK(
            cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));

        // FFT size
        LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h_, CUFFTDX_OPERATOR_SIZE, fft_size));
        // CUFFTDX_TYPE_C2C means complex-to-complex FFT type
        // CUFFTDX_TYPE_R2C means real-to-complex FFT type
        // CUFFTDX_TYPE_C2R means complex-to-real FFT type
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
        // COMMONDX_PRECISION_F16 for half precision
        // COMMONDX_PRECISION_F32 for single precision
        // COMMONDX_PRECISION_F64 for double precision
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

        // COMMONDX_OPTION_SYMBOL_NAME indicates the required name for the device function.
        LIBMATHDX_CHECK(cufftdxSetOptionStr(h_, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, "my_fft"));    
      }

      bool IsSupported() const {
        int valid = -1;
        LIBMATHDX_CHECK(cufftdxHasImplementation(h_, &valid));
        return static_cast<bool>(valid);
      }

      int GetShmRequired() const {
        long long int shared_memory_size = 0;
        LIBMATHDX_CHECK(cufftdxGetTraitInt64(h_, CUFFTDX_TRAIT_SHARED_MEMORY_SIZE, &shared_memory_size));      
        return static_cast<int>(shared_memory_size);
      }


      ElementsPerThread GetMaxEPT() const {
        // ElementsPerThread type is needed for EPT capability, but cuFFTDx uses long long int for EPTs
        cufftdxKnobType_t knobs = CUFFTDX_KNOB_ELEMENTS_PER_THREAD;
        size_t num_epts = 0;
        LIBMATHDX_CHECK(cufftdxGetKnobInt64Size(h_, 1, &knobs, &num_epts));
        std::vector<long long int> epts(num_epts, 0);
        LIBMATHDX_CHECK(cufftdxGetKnobInt64s(h_, 1, &knobs, epts.size(), epts.data()));      

        return static_cast<ElementsPerThread>(*std::max_element(epts.begin(), epts.end()));
      }

    private:
      std::vector<long long int> valid_epts_;
      cufftdxDescriptor h_;
  };


  template <typename InputType>
  bool IsDxBlockFFTSupported(int arch_cc, index_t fft_size, FFTType fft_type, FFTDirection direction) {
    cufftdxDescriptor h{0};

    LIBMATHDX_CHECK(cufftdxCreateDescriptor(&h));
    // CUFFTDX_API_LMEM means the function will be of signature:
    //     void(value_type*, value_type*)
    //   with the first argument being local memory ("registers"), with each thread holding "EPT" elements
    //   and the second being a pointer to a shared memory scratch buffer
    // CUFFTDX_API_SMEM would mean that the function will be of signature:
    //     void(value_type*)
    //   and takes a shared memory pointer with all the elements laid out in natural order
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_API, cufftdxApi::CUFFTDX_API_SMEM));
    // COMMONDX_EXECUTION_BLOCK means multiple threads in a block participate in the FFT
    // COMMONDX_EXECUTION_THREAD would mean that each thread computes a single FFT
    LIBMATHDX_CHECK(
        cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));

    // FFT size
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_SIZE, fft_size));
    // CUFFTDX_TYPE_C2C means complex-to-complex FFT type
    // CUFFTDX_TYPE_R2C means real-to-complex FFT type
    // CUFFTDX_TYPE_C2R means complex-to-real FFT type
    cufftdxType cufftdx_type;
    if (fft_type == FFTType::C2C) {
      cufftdx_type = cufftdxType::CUFFTDX_TYPE_C2C;
    } else if (fft_type == FFTType::C2R) {
      cufftdx_type = cufftdxType::CUFFTDX_TYPE_C2R;
    } else if (fft_type == FFTType::R2C) {
      cufftdx_type = cufftdxType::CUFFTDX_TYPE_R2C;
    } else {
      return false;
    }

    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_TYPE, cufftdx_type));

    cufftdxDirection cufftdx_direction;
    if (direction == FFTDirection::FORWARD) {
      cufftdx_direction = cufftdxDirection::CUFFTDX_DIRECTION_FORWARD;
    } else {
      cufftdx_direction = cufftdxDirection::CUFFTDX_DIRECTION_INVERSE;
    }
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_DIRECTION, cufftdx_direction));
    // COMMONDX_PRECISION_F16 for half precision
    // COMMONDX_PRECISION_F32 for single precision
    // COMMONDX_PRECISION_F64 for double precision
    if constexpr (cuda::std::is_same_v<InputType, matxBf16Complex> || cuda::std::is_same_v<InputType, matxFp16Complex> || 
                  cuda::std::is_same_v<InputType, matxBf16> || cuda::std::is_same_v<InputType, matxFp16>) {
      LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F16));
    } else if constexpr (cuda::std::is_same_v<InputType, cuda::std::complex<float>> || cuda::std::is_same_v<InputType, float>) {
      LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F32));
    } else if constexpr (cuda::std::is_same_v<InputType, cuda::std::complex<double>> || cuda::std::is_same_v<InputType, double>) {
      LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F64));
    } else {
      return false;
    }
    
    // Compute capability to target
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_SM, arch_cc));

    // COMMONDX_OPTION_SYMBOL_NAME indicates the required name for the device function.
    LIBMATHDX_CHECK(cufftdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, "my_fft"));    

    int valid = -1;
    LIBMATHDX_CHECK(cufftdxHasImplementation(h, &valid));

    return static_cast<bool>(valid);
  }
#endif

  } // namespace detail
} // namespace matx