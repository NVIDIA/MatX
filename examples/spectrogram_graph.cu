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
#include "matx_viz.h"
#include <cassert>
#include <cstdio>
#include <math.h>
#include <memory>

using namespace matx;
#define FFT_TYPE CUFFT_C2C

/** Create a spectrogram of a signal
 *
 * This example creates a set of data representing signal power versus frequency
 * and time. Traditionally the signal power is plotted as the Z dimension using
 * color, and time/frequency are the X/Y axes. The time taken to run the
 * spectrogram is computed, and a simple scatter plot is output. This version
 * does uses CUDA graphs, and records the workload on the second iteration of
 * the intialization loop. The first iteration is used only for plan caching and
 * should not include any graph recording.
 */

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  using complex = cuda::std::complex<float>;
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float fs = 10000;
  index_t N = 100000;
  float amp = static_cast<float>(2 * sqrt(2));
  index_t nperseg = 256;
  index_t nfft = 256;
  index_t noverlap = nperseg / 8;
  index_t nstep = nperseg - noverlap;
  constexpr uint32_t num_iterations = 100;
  float time_ms;

  std::array<index_t, 1> num_samps{N};
  std::array<index_t, 1> half_win{nfft / 2 + 1};
  std::array<index_t, 1> s_time_shape{(N - noverlap) / nstep};

  tensor_t<float, 1> time({N});
  tensor_t<float, 1> modulation({N});
  tensor_t<float, 1> carrier({N});
  tensor_t<float, 1> noise({N});
  tensor_t<float, 1> x({N});
  auto freqs = make_tensor<float, 1>(half_win);
  tensor_t<complex, 2> fftStackedMatrix(
      {(N - noverlap) / nstep, nfft / 2 + 1});
  tensor_t<float, 1> s_time({(N - noverlap) / nstep});

  randomGenerator_t<float> randData({N}, 0);
  auto randDataView = randData.GetTensorView<1>(num_samps, NORMAL);

  // Set up all static buffers
  // time = np.arange(N) / float(fs)
  (time = linspace<0>(num_samps, 0.0f, static_cast<float>(N) - 1.0f) / fs)
      .run(stream);
  // mod = 500 * np.cos(2*np.pi*0.25*time)
  (modulation = 500 * cos(2 * M_PI * 0.25 * time)).run(stream);
  // carrier = amp * np.sin(2*np.pi*3e3*time + modulation)
  (carrier = amp * sin(2 * M_PI * 3000 * time + modulation)).run(stream);
  // noise = 0.01 * fs / 2 * np.random.randn(time.shape)
  (noise = sqrt(0.01 * fs / 2) * randDataView).run(stream);
  // noise *= np.exp(-time/5)
  (noise = noise * exp(-1.0f * time / 5.0f)).run(stream);
  // x = carrier + noise
  (x = carrier + noise).run(stream);

  for (uint32_t i = 0; i < 2; i++) {
    // Record graph on second loop to get rid of plan caching in the graph
    if (i == 1) {
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    }

    // DFT Sample Frequencies (rfftfreq)
    (freqs = (1.0 / (static_cast<float>(nfft) * 1 / fs)) *
               linspace<0>(half_win, 0.0f, static_cast<float>(nfft) / 2.0f))
        .run(stream);

    // Create overlapping matrix of segments.
    auto stackedMatrix = x.OverlapView({nperseg}, {nstep});
    // FFT along rows
    fft(fftStackedMatrix, stackedMatrix, 0, stream);
    // Absolute value
    (fftStackedMatrix = conj(fftStackedMatrix) * fftStackedMatrix)
        .run(stream);
    // Get real part and transpose
    auto Sxx = fftStackedMatrix.RealView().Permute({1, 0});

    // Spectral time axis
    (s_time = linspace<0>(s_time_shape, static_cast<float>(nperseg) / 2.0f,
                           static_cast<float>(N - nperseg) / 2.0f + 1) /
                fs)
        .run(stream);

    if (i == 1) {
      cudaStreamEndCapture(stream, &graph);
      cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

#if MATX_ENABLE_VIZ
      // Generate a spectrogram visualization using a contour plot
      viz::contour(time, freqs, Sxx);
#else
      printf("Not outputting plot since visualizations disabled\n");
#endif            
    }
  }

  // Time graph execution of same kernels
  cudaEventRecord(start, stream);
  for (uint32_t i = 0; i < num_iterations; i++) {
    cudaGraphLaunch(instance, stream);
  }
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  cudaEventElapsedTime(&time_ms, start, stop);

  printf("Spectrogram Time With Graphs = %.2fus per iteration\n",
         time_ms * 1e3 / num_iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}