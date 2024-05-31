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

#include <matx.h>

using namespace matx;

int main() {

  using complex = cuda::std::complex<float>;
  cudaExecutor exec{};
  
  index_t signal_size = 16;
  index_t filter_size = 3;
  index_t filtered_size = signal_size + filter_size - 1;

  // Create time domain buffers
  auto sig_time  = make_tensor<complex>({signal_size});
  auto filt_time = make_tensor<complex>({filter_size});
  auto time_out  = make_tensor<complex>({filtered_size});

  // Frequency domain buffers
  auto sig_freq  = make_tensor<complex>({filtered_size});
  auto filt_freq = make_tensor<complex>({filtered_size});

  // Fill the time domain signals with data
  for (index_t i = 0; i < signal_size; i++) {
    sig_time(i) = {-1.0f * (2.0f * static_cast<float>(i % 2) + 1.0f) *
                          (static_cast<float>(i % 10) / 10.0f) +
                      0.1f,
                  -1.0f * (static_cast<float>(i % 2) == 0.0f) *
                          (static_cast<float>(i % 10) / 5.0f) -
                      0.1f};
  }
  for (index_t i = 0; i < filter_size; i++) {
    filt_time(i) = {static_cast<float>(i) / static_cast<float>(filter_size),
                    static_cast<float>(-i) / static_cast<float>(filter_size) +
                        0.5f};
  }

  // TODO: Perform FFT convolution
  // Perform the FFT in-place on both signal and filter, do an element-wise multiply of the two, then IFFT that output


  // TODO: Perform a time-domain convolution
  

  exec.sync();

  // Compare signals
  for (index_t i = 0; i < filtered_size; i++) {
      if (  fabs(time_out(i).real() - sig_freq(i).real()) > 0.001 || 
            fabs(time_out(i).imag() - sig_freq(i).imag()) > 0.001) {
          printf("Verification failed at item %lld. Direct=%f%+.2fj, FFT=%f%+.2fj\n", i,
            time_out(i).real(), time_out(i).imag(), sig_freq(i).real(), sig_freq(i).imag());
          return -1;
      }
  }

  std::cout << "Verification successful" << std::endl;

  return 0;
}
