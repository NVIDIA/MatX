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

#include "matx.h"
#include <cassert>
#include <cstdio>
#include <cuda/std/ccomplex>

using namespace matx;

/**
 * FFT Convolution
 *
 * This example shows how to perform an FFT convolution using the MatX library.
 * The example shows the convolution theorem of:
 *
 * \f(h*x \leftrightarrow H \cdot X$  \f)
 *
 * Namely, a convolution in the time domain is a point-wise multiplication in
 * the frequency domain. In this example we start with two signals in the time
 * domain, convert them to frequency domain, perform the multiply, then convert
 * them back to the time domain. This should give very close results to
 * performing a direct convolution in the time domain, so the results are
 * compared to a direct convolution. They will not match identically since the
 * types and order of operations are different, but they will match within a
 * close margin.
 *
 * FFT convolution is frequently used in signal processing when a signal or
 * filter is larger than a threshold, since it will outperform direct
 * convolution past this threshold. Another benefit of FFT convolution is the
 * number of operations is the same, regardless of the filter size. This allows
 * a user to FFT a very long filter one time, and that buffer can be used many
 * times for any incoming samples.
 *
 * For smaller signal sizes, the FFT convolution typically performs worse since
 * there is some buffer and 3 FFT operations (2 for FFT of signal and filter,
 * and 1 IFFT after the multiply) that causes the setup time to dominate.
 * 
 * Note that the conv1d() operator has a mode to perform FFT-based convolution
 * automatically.
 *
 */
int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  using complex = cuda::std::complex<float>;
//   cudaExecutor exec{};

//   index_t signal_size = 1ULL << 16;
//   index_t filter_size = 16;
//   index_t batches = 8;
//   index_t filtered_size = signal_size + filter_size - 1;
//   float separate_ms;
//   float fused_ms;
//   constexpr int iterations = 2;
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);  
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);  

//   // Create time domain buffers
//   auto sig_time  = make_tensor<complex>({batches, signal_size});
//   auto filt_time = make_tensor<complex>({batches, filter_size});
//   auto time_out  = make_tensor<complex>({batches, filtered_size});

//   // Frequency domain buffers
//   auto sig_freq  = make_tensor<complex>({batches, filtered_size});
//   auto filt_freq = make_tensor<complex>({batches, filtered_size});

//   for (index_t b = 0; b < batches; b++) {
//     // Fill the time domain signals with data
//     for (index_t i = 0; i < signal_size; i++) {
//       sig_time(b,i) = {-1.0f * (2.0f * static_cast<float>(i % 2) + 1.0f) *
//                             (static_cast<float>(i % 10) / 10.0f) +
//                         0.1f,
//                     -1.0f * (static_cast<float>(i % 2) == 0.0f) *
//                             (static_cast<float>(i % 10) / 5.0f) -
//                         0.1f};
//     }
//     for (index_t i = 0; i < filter_size; i++) {
//       filt_time(b,i) = {static_cast<float>(i) / static_cast<float>(filter_size),
//                       static_cast<float>(-i) / static_cast<float>(filter_size) +
//                           0.5f};
//     }
//   }

//   // Perform the FFT in-place on both signal and filter
//   for (int i = 0; i < iterations; i++) {
//     if (i == 1) {
//       cudaEventRecord(start, stream);
//     }    
//     printf("first fft\n");
//     (sig_freq = fft(sig_time, filtered_size)).run(exec);
// printf("second fft\n");    
//     (filt_freq = fft(filt_time, filtered_size)).run(exec);

//     printf("multiply\n");
//     (sig_freq = sig_freq * filt_freq).run(exec);

//     // IFFT in-place
//     printf("ifft\n");
//     (sig_freq = ifft(sig_freq)).run(exec);
    
//   }

//   cudaEventRecord(stop, stream);
//   exec.sync();
//   cudaEventElapsedTime(&separate_ms, start, stop);   

//   // for (int i = 0; i < iterations; i++) {
//   //   if (i == 1) {
//   //     cudaEventRecord(start, stream);
//   //   }
//   //   (sig_freq = ifft(fft(sig_time, filtered_size) * fft(filt_time, filtered_size))).run(exec);
//   // }
  
//   cudaEventRecord(stop, stream);
//   exec.sync();
//   cudaEventElapsedTime(&fused_ms, start, stop);  

//   printf("FFT runtimes for separate = %.2f ms, fused = %.2f ms\n", separate_ms/(iterations-1), fused_ms/(iterations-1));

//   // Now the sig_freq view contains the full convolution result. Verify against
//   // a direct convolution. The conv1d function only accepts a 1D filter, so we
//   // create a sliced view here.
//   // auto filt1 = slice<1>(filt_time, {0,0}, {matxDropDim, matxEnd});
//   // (time_out = conv1d(sig_time, filt1, matxConvCorrMode_t::MATX_C_MODE_FULL)).run(exec);

//   // exec.sync();
 
//   // Compare signals
//   for (index_t b = 0; b < batches; b++) {
//     for (index_t i = 0; i < filtered_size; i++) {
//       if (fabs(time_out(b,i).real() - sig_freq(b,i).real()) > 0.001 ||
//           fabs(time_out(b,i).imag() - sig_freq(b,i).imag()) > 0.001) {
//         std::cout <<
//             "Verification failed at item " << i << ". Direct=" << time_out(b,i).real() << " " << time_out(b,i).imag() << ", FFT=" <<
//             sig_freq(b,i).real() << " " <<
//             sig_freq(b,i).imag() << "\n";
//         return -1;
//       }
//     }
//   }

//   std::cout << "Verification successful" << std::endl;

  {
    auto a = make_tensor<cuda::std::complex<float>>({1000000});
    auto b = make_tensor<cuda::std::complex<float>>({1000000});
    auto c = make_tensor<cuda::std::complex<float>>({1000000});
    (c = (complex)0).run();
    (c = fft(a)).run();
    cudaDeviceSynchronize();
  }

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}