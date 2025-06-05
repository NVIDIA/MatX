#pragma once

#include "matx/core/operator_options.h"

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

      // } else { // is_r2c_c2r_folded_mode
      //     if (!is_power_of_two(fft_size)) return false;

      //     switch (precision) {
      //         case Precision::HALF:
      //             if (arch_cc == 75) return (fft_size >= 4 && fft_size <= 8192);
      //             if (arch_cc == 70 || arch_cc == 72 || arch_cc == 86 || arch_cc == 89) return (fft_size >= 4 && fft_size <= 32768);
      //             if (arch_cc == 80 || arch_cc == 87 || arch_cc == 90) return (fft_size >= 2 && fft_size <= 65536);
      //             break;
      //         case Precision::FLOAT:
      //             if (arch_cc == 75) return (fft_size >= 2 && fft_size <= 8192);
      //             if (arch_cc == 70 || arch_cc == 72 || arch_cc == 86 || arch_cc == 89) return (fft_size >= 2 && fft_size <= 32768);
      //             if (arch_cc == 80 || arch_cc == 87 || arch_cc == 90) return (fft_size >= 2 && fft_size <= 65536);
      //             break;
      //         case Precision::DOUBLE:
      //             if (arch_cc == 75) return (fft_size >= 2 && fft_size <= 4096);
      //             if (arch_cc == 70 || arch_cc == 72 || arch_cc == 86 || arch_cc == 89) return (fft_size >= 2 && fft_size <= 16384);
      //             if (arch_cc == 80 || arch_cc == 87 || arch_cc == 90) return (fft_size >= 2 && fft_size <= 32768);
      //             break;
      //     }
      // }

      return false; // Default if architecture or specific configuration not explicitly supported by table logic
  }


  template <typename input_type, ElementsPerThread EPT, int size, typename Op, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ auto RunDxFFT(const Op &op, Is... indices) {
    __syncthreads();

    __shared__  Vector<input_type, static_cast<int>(EPT)> thread_data[size];
    thread_data[threadIdx.x] = op.template operator()<EPT>(indices...);
    __syncthreads();

    using FFT = decltype(cufftdx::Block() + cufftdx::Size<size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<float>() +
                cufftdx::FFTsPerBlock<1>() + cufftdx::SM<800>() + cufftdx::ElementsPerThread<static_cast<int>(EPT)>());

    //auto thread_data = a_.template operator()<EPT>(indices...);
    //extern __shared__ __align__(alignof(float2)) input_type shared_mem[];
    printf("before %d %f%+f %f%+f \n", threadIdx.x, thread_data[threadIdx.x].data[0].real(), thread_data[threadIdx.x].data[0].imag(), thread_data[threadIdx.x].data[1].real(), thread_data[threadIdx.x].data[1].imag());
    FFT().execute(&thread_data[0]);
    printf("after %d %f%+f %f%+f \n", threadIdx.x, thread_data[threadIdx.x].data[0].real(), thread_data[threadIdx.x].data[0].imag(), thread_data[threadIdx.x].data[1].real(), thread_data[threadIdx.x].data[1].imag());
    return thread_data[threadIdx.x];  
  }

  } // namespace detail
} // namespace matx