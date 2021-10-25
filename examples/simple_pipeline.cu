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

#include "simple_pipeline.h"

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  index_t numChannels = 16;
  index_t numPulses = 128;
  index_t numSamples = 9000;
  index_t waveformLength = 1000;
  uint32_t iterations = 100;

  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "numChannels: " << numChannels << std::endl;
  std::cout << "numPulses: " << numPulses << std::endl;
  std::cout << "numNumSamples: " << numSamples << std::endl;
  std::cout << "waveformLength: " << waveformLength << std::endl;

  // cuda stream to place work in
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // create some events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("Initializing data structures...\n");
  auto radar =
      RadarPipeline(numPulses, numSamples, waveformLength, numChannels, stream);
  radar.GetInputView()->PrefetchDevice(stream);

  printf("Running test...\n");
  cudaStreamSynchronize(stream);
  cudaEventRecord(start, stream);
  for (uint32_t i = 0; i < iterations; i++) {
    radar.PulseCompression();
    radar.ThreePulseCanceller();
    radar.DopplerProcessing();
    radar.CFARDetections();
  }

  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  float time_ms;
  cudaEventElapsedTime(&time_ms, start, stop);
  float time_s = time_ms * .001f;

  printf("Pipeline rate: %f pulses/channel/sec\n",
         static_cast<float>(iterations * numPulses) / time_s);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  cudaDeviceSynchronize();
  CUDA_CHECK_LAST_ERROR();

  matxPrintMemoryStatistics();

  printf("Done\n");

  MATX_EXIT_HANDLER();
  return 0;
}
