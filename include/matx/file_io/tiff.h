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

#pragma once

#ifdef MATX_ENABLE_NVTIFF

#include <vector>
#include <nvtiff.h>
#define MATX_CHECK_NVTIFF(call) do {                                                             \
  nvtiffStatus_t _e = (call);                                                                    \
  if (_e != NVTIFF_STATUS_SUCCESS) {                                                             \
    fprintf(stderr, "nvtiff error code %d in file '%s' in line %i\n", _e, __FILE__, __LINE__);   \
    MATX_THROW(matx::matxCudaError, "nvTIFF error");                                             \
  }                                                                                              \
} while (0)

namespace matx {
  namespace io {

    class Tiff
    {
      public:
        Tiff(cudaStream_t stream=0, int NUM_DECODERS=1)
          : NUM_DECODERS_(NUM_DECODERS), decoder_idx(0), stream_(stream)
        {
          events.resize(NUM_DECODERS_);
          tiff_streams.resize(NUM_DECODERS_);
          decoders.resize(NUM_DECODERS_);
          for (int k=0; k<NUM_DECODERS_; k++)
          {
            MATX_CUDA_CHECK(cudaEventCreate(&events[k]));
            MATX_CUDA_CHECK(cudaEventRecord(events[k],stream_));
            MATX_CHECK_NVTIFF(nvtiffStreamCreate(&tiff_streams[k]));
            MATX_CHECK_NVTIFF(nvtiffDecoderCreate(&decoders[k], nullptr, nullptr, stream_));
          }
        }

        ~Tiff()
        {
          for (int k=0; k<NUM_DECODERS_; k++)
          {
            nvtiffDecoderDestroy(decoders[k],stream_);
            nvtiffStreamDestroy(tiff_streams[k]);
            cudaEventDestroy(events[k]);
          }
        }

        template<typename T>
        void load(const char* filename, uint32_t image_id, T& t)
        {
          nvtiffFileInfo_t file_info;
          nvtiffImageInfo_t image_info;

          MATX_CUDA_CHECK(cudaEventSynchronize(events[decoder_idx]));
          MATX_CHECK_NVTIFF(nvtiffStreamParseFromFile(filename, tiff_streams[decoder_idx]));
          MATX_CHECK_NVTIFF(nvtiffStreamGetFileInfo(tiff_streams[decoder_idx], &file_info));
          MATX_CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_streams[decoder_idx], image_id, &image_info));
          make_tensor(t, {image_info.image_height, image_info.image_width});
          uint8_t* data[1] {reinterpret_cast<uint8_t*>(t.Data())};
          MATX_CHECK_NVTIFF(nvtiffDecodeRange(tiff_streams[decoder_idx], decoders[decoder_idx], image_id, 1, data, stream_));
          MATX_CUDA_CHECK(cudaEventRecord(events[decoder_idx],stream_));
          decoder_idx++;
          if (decoder_idx >= NUM_DECODERS_) {
            decoder_idx = 0;
          }
        }

        void load_toptr(const char* filename, uint32_t image_id, uint8_t* t_data)
        {
          MATX_CUDA_CHECK(cudaEventSynchronize(events[decoder_idx]));
          MATX_CHECK_NVTIFF(nvtiffStreamParseFromFile(filename, tiff_streams[decoder_idx]));
          uint8_t* data[1] {reinterpret_cast<uint8_t*>(t_data)};
          MATX_CHECK_NVTIFF(nvtiffDecodeRange(tiff_streams[decoder_idx], decoders[decoder_idx], image_id, 1, data, stream_));
          MATX_CUDA_CHECK(cudaEventRecord(events[decoder_idx],stream_));
          decoder_idx++;
          if (decoder_idx >= NUM_DECODERS_) {
            decoder_idx = 0;
          }
        }

        void info(const char* filename, uint32_t image_id)
        {
          nvtiffFileInfo_t file_info;
          nvtiffImageInfo_t image_info;

          MATX_CUDA_CHECK(cudaEventSynchronize(events[decoder_idx]));
          MATX_CHECK_NVTIFF(nvtiffStreamParseFromFile(filename, tiff_streams[decoder_idx]));
          MATX_CHECK_NVTIFF(nvtiffStreamGetFileInfo(tiff_streams[decoder_idx], &file_info));
          printf("%s:\n",filename);
          for (uint32_t k=0; k<file_info.num_images; k++)
          {
            MATX_CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_streams[decoder_idx], image_id, &image_info));
            printf("  image %u\n",k);
            printf("    image_width: %u\n",image_info.image_width);
            printf("    image_height: %u\n",image_info.image_height);
            printf("    samples_per_pixel: %u\n",image_info.samples_per_pixel);
            printf("    bits_per_pixel:  %u\n",image_info.bits_per_pixel);
            printf("    sample_format[0]: %u\n",image_info.sample_format[0]);
          }
        }

      private:
        int NUM_DECODERS_;
        int decoder_idx;
        cudaStream_t stream_;
        std::vector<cudaEvent_t> events;
        std::vector<nvtiffStream_t> tiff_streams;
        std::vector<nvtiffDecoder_t> decoders;
    };

  }
}

#endif // MATX_ENABLE_NVTIFF defined