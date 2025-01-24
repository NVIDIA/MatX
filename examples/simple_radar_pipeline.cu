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

#include "simple_radar_pipeline.h"

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  index_t numChannels = 16;
  index_t numPulses = 128;
  index_t numSamples = 9000;
  index_t waveformLength = 1000;
  uint32_t iterations = 100;

#ifdef USE_STF
  std::cout << "Using STF executor\n";
#else
  std::cout << "Using CUDA executor\n";
#endif

#if 0
  constexpr int numStreams = 8;
#else
  int numStreams = 1;
#endif

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--numChannels" && i + 1 < argc) {
            numChannels = std::stoi(argv[++i]);
        } else if (arg == "--numPulses" && i + 1 < argc) {
            numPulses = std::stoi(argv[++i]);
        } else if (arg == "--numSamples" && i + 1 < argc) {
            numSamples = std::stoi(argv[++i]);
        } else if (arg == "--waveformLength" && i + 1 < argc) {
            waveformLength = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--numStreams" && i + 1 < argc) {
            numStreams = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown option or missing value: " << arg << std::endl;
            return 1; // Exit with error
        }
    }

  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "numChannels: " << numChannels << std::endl;
  std::cout << "numPulses: " << numPulses << std::endl;
  std::cout << "numSamples: " << numSamples << std::endl;
  std::cout << "waveformLength: " << waveformLength << std::endl;
  std::cout << "numStreams: " << numStreams << std::endl;

  constexpr bool ENABLE_GRAPHS = false;
  cudaGraph_t graphs[numStreams];
  cudaGraphExec_t instances[numStreams];  
  using complex = cuda::std::complex<float>;
  RadarPipeline<complex> *pipelines[numStreams];

  // cuda stream to place work in
  cudaStream_t streams[numStreams];
  
  // manually set to log all NVTX levels
  MATX_NVTX_SET_LOG_LEVEL( matx_nvxtLogLevels::MATX_NVTX_LOG_ALL );
  
  // create some events for timing
  cudaEvent_t starts[numStreams];
  cudaEvent_t stops[numStreams];

  for (int s = 0; s < numStreams; s++) {
    cudaEventCreate(&starts[s]);
    cudaEventCreate(&stops[s]);
    cudaStreamCreate(&streams[s]);
    
    MATX_NVTX_START_RANGE("Pipeline Initialize", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 1)
#if 0
    printf("Initializing data structures for stream %d...\n", s);
#endif
    pipelines[s] = new RadarPipeline(numPulses, numSamples, waveformLength, numChannels, streams[s]);
    MATX_NVTX_END_RANGE(1)

    pipelines[s]->sync();  
  }

  MATX_NVTX_START_RANGE("Pipeline Test", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 2)
#if 0
  printf("Running test...\n");
#endif

  auto run_pipeline = [&](int s) {
    MATX_NVTX_START_RANGE("PulseCompression", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 21)
    pipelines[s]->PulseCompression();
    MATX_NVTX_END_RANGE(21)  
    
    MATX_NVTX_START_RANGE("ThreePulseCanceller", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 22)
    pipelines[s]->ThreePulseCanceller();
    MATX_NVTX_END_RANGE(22)
    
    MATX_NVTX_START_RANGE("DopplerProcessing", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 23)
    pipelines[s]->DopplerProcessing();
    MATX_NVTX_END_RANGE(23)
    
    MATX_NVTX_START_RANGE("CFARDetections", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 24)
    pipelines[s]->CFARDetections();
    MATX_NVTX_END_RANGE(24)
  }; 

  // Warmup
  for (int s = 0; s < numStreams; s++) {
    run_pipeline(s);
  }

  if (ENABLE_GRAPHS) {
    for (int s = 0; s < numStreams; s++) {
      cudaStreamBeginCapture(streams[s], cudaStreamCaptureModeGlobal);
      run_pipeline(s);
      cudaStreamEndCapture(streams[s], &graphs[s]);
      cudaGraphInstantiate(&instances[s], graphs[s], NULL, NULL, 0);     
    }
  }
  
  for (uint32_t i = 0; i < iterations; i++) {
    for (int s = 0; s < numStreams; s++) {
      if (i == 1) {
#ifdef USE_STF
        auto ctx = pipelines[s]->exec.getCtx();
        cudaEventRecord(starts[s], ctx.task_fence());
#else
        cudaEventRecord(starts[s], streams[s]);
#endif
      }

      if (ENABLE_GRAPHS) {
        cudaGraphLaunch(instances[s], streams[s]);
      }
      else {
        run_pipeline(s);
      }
    }
  }

  for (int s = 0; s < numStreams; s++) {
#ifdef USE_STF
    auto ctx = pipelines[s]->exec.getCtx();
    cudaEventRecord(stops[s], ctx.task_fence());
    std::cout << "using stf 2\n";
#else
    cudaEventRecord(stops[s], streams[s]);
    std::cout << "using cudaexec 2\n";
#endif
    pipelines[s]->sync();
  }

#ifdef USE_STF
      for (int s = 0; s < numStreams; s++) {
          auto ctx = pipelines[s]->exec.getCtx();
          ctx.finalize();
      }
#endif

  MATX_NVTX_END_RANGE(2)
  
  MATX_NVTX_START_RANGE("Pipeline Results", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 3)
  float time_ms;
  cudaEventElapsedTime(&time_ms, starts[numStreams-1], stops[numStreams-1]);
  float time_s = time_ms * .001f;

  auto mult = iterations * numChannels * numPulses * numStreams;
  printf("Pipeline finished in %.2fms, rate: %.2f pulses/channel/sec (%.2f Gbps)\n",
        time_ms,
         static_cast<float>(mult) / time_s,
         static_cast<float>(mult*sizeof(complex)*numSamples*8)/time_s/1e9);

for (int s = 0; s < numStreams; s++) {
    cudaEventDestroy(starts[s]);
    cudaEventDestroy(stops[s]);
    cudaStreamDestroy(streams[s]);
}

  cudaDeviceSynchronize();
  CUDA_CHECK_LAST_ERROR();

  matxPrintMemoryStatistics();

  printf("Done\n");
  MATX_NVTX_END_RANGE(3)
  MATX_EXIT_HANDLER();
  return 0;
}
