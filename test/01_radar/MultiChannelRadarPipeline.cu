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

#include "assert.h"
#include "matx.h"
#include "simple_radar_pipeline.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"

template <typename T> class MultiChannelRadarPipeline : public ::testing::Test {
protected:
  void SetUp() override
  {
    assert(numSamples > waveformLength);

    pb = std::make_unique<detail::MatXPybind>();
    pb->InitTVGenerator<T>("01_radar", "simple_radar_pipeline",
                           {numPulses, numSamples, waveformLength});

    // Set the number of channels before running
    auto f = pb->GetMethod("set_channels");
    f(numChannels);

    pb->RunTVGenerator("run");
  }

  void TearDown() { pb.reset(); }

  uint32_t iterations = 100;
  index_t numChannels = 16;
  index_t numPulses = 128;
  index_t numSamples = 9000;
  index_t waveformLength = 1000;
  index_t numSamplesRnd = 16384;
  index_t numCompressedSamples = numSamples - waveformLength + 1;
  std::unique_ptr<detail::MatXPybind> pb;
};

template <typename TensorType>
class MultiChannelRadarPipelineTypes
    : public MultiChannelRadarPipeline<TensorType> {
};

TYPED_TEST_SUITE(MultiChannelRadarPipelineTypes, MatXComplexNonHalfTypes);

TYPED_TEST(MultiChannelRadarPipelineTypes, PulseCompression)
{
  MATX_ENTER_HANDLER();

  auto p = RadarPipeline<TypeParam>(this->numPulses, this->numSamples,
                                    this->waveformLength, this->numChannels, 0);
  auto d = p.GetInputView();

  auto data =
      tensor_t<TypeParam, 2>({this->numPulses, this->numSamplesRnd});
  auto x_in = data.Slice({0, 0}, {this->numPulses, this->numSamples});
  auto x_clone =
      data.template Clone<3>({this->numChannels, matxKeepDim, matxKeepDim});
  this->pb->NumpyToTensorView(x_in, "x_init");

  // Copy the replicated data into the actual data pointer
  matx::copy(d, x_clone, 0);

  d.PrefetchDevice(0);

  auto wfd = p.GetwaveformView();
  auto wf = wfd.Slice({0}, {this->waveformLength});

  this->pb->NumpyToTensorView(wf, "waveform");

  wfd.PrefetchDevice(0);

  p.PulseCompression();

  auto xc = d.Slice({0, 0, 0}, {this->numChannels, this->numPulses,
                                 this->numCompressedSamples});

  MATX_TEST_ASSERT_COMPARE(this->pb, xc, "x_compressed", 0.01);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MultiChannelRadarPipelineTypes, ThreePulseCanceller)
{
  MATX_ENTER_HANDLER();

  auto p = RadarPipeline<TypeParam>(this->numPulses, this->numSamples,
                                    this->waveformLength, this->numChannels, 0);

  auto d = p.GetInputView();

  // auto data = tensor_t<complex,2>({this->numPulses,
  // this->numSamplesRnd}); auto x_in = data.View().Slice({0, 0},
  // {this->numPulses, this->numCompressedSamples}); auto x_clone =
  // data.View().Clone<3>({this->numChannels, matxKeepDim, matxKeepDim});

  auto v = d.Slice({0, 0, 0}, {this->numChannels, this->numPulses,
                                this->numCompressedSamples});
  this->pb->NumpyToTensorView(v, "x_compressed");

  // copy(d.View(), x_clone, 0);

  p.ThreePulseCanceller();

  auto out = p.GetTPCView();
  auto tpcView = out.Slice({0, 1, 0}, {this->numChannels, this->numPulses - 1,
                                        this->numCompressedSamples});

  MATX_TEST_ASSERT_COMPARE(this->pb, tpcView, "x_conv2", 0.01);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MultiChannelRadarPipelineTypes, Doppler)
{
  MATX_ENTER_HANDLER();

  auto p = RadarPipeline<TypeParam>(this->numPulses, this->numSamples,
                                    this->waveformLength, this->numChannels, 0);
  auto in = p.GetTPCView();

  auto v = in.Slice({0, 0, 0}, {this->numChannels, this->numPulses - 2,
                                 this->numCompressedSamples});
  this->pb->NumpyToTensorView(v, "x_conv2");

  in.PrefetchHost(0);

  p.DopplerProcessing();

  auto out = p.GetTPCView();

  MATX_TEST_ASSERT_COMPARE(this->pb, out, "X_window", 0.01);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MultiChannelRadarPipelineTypes, CFARDetection)
{
  MATX_ENTER_HANDLER();
  const int fft_size = 256;
  const int cfarMaskX = 13;
  const int cfarMaskY = 5;

  auto p = RadarPipeline<TypeParam>(this->numPulses, this->numSamples,
                                    this->waveformLength, this->numChannels, 0);
  auto inView = p.GetTPCView();

  this->pb->NumpyToTensorView(inView, "X_window");

  p.CFARDetections();

  auto detsView = p.GetDetections();
  auto ba = p.GetBackgroundAverages();

  auto baTrim = ba.Slice({0, cfarMaskY / 2, cfarMaskX / 2},
                         {this->numChannels, fft_size + cfarMaskY / 2,
                          this->numCompressedSamples + cfarMaskX / 2});

  MATX_TEST_ASSERT_COMPARE(this->pb, baTrim, "background_averages", 0.01);
  MATX_TEST_ASSERT_COMPARE(this->pb, detsView, "dets", 0.01);

  uint32_t detects = 0;
  for (index_t s0 = 0; s0 < detsView.Size(0); s0++) {
    for (index_t s1 = 0; s1 < detsView.Size(1); s1++) {
      for (index_t s2 = 0; s2 < detsView.Size(2); s2++) {
        detects += detsView(s0, s1, s2);
      }
    }
  }

  // 24 detections per channel
  ASSERT_EQ(detects, 17 * this->numChannels);
  MATX_EXIT_HANDLER();
}
