#include <nvbench/nvbench.cuh>
#include "simple_radar_pipeline.h"
#include "matx.h"

using namespace matx;

//using radar_types = nvbench::type_list<cuda::std::complex<float>, cuda::std::complex<double>>;
using radar_types = nvbench::type_list<cuda::std::complex<float>>;


template <typename ValueType>
void simple_radar_pipeline_pulse_compression(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const index_t numPulses = static_cast<int>(state.get_int64("Pulses"));
  const index_t numChannels = static_cast<int>(state.get_int64("Channels"));
  const index_t numSamples = static_cast<int>(state.get_int64("Samples"));
  const index_t waveformLength = static_cast<int>(state.get_int64("Waveform Length"));

  state.exec( nvbench::exec_tag::timer, 
    [&numPulses, &numChannels, &numSamples, &waveformLength](nvbench::launch &launch, auto &timer) {
      auto radar = RadarPipeline(numPulses, numSamples, waveformLength, numChannels, launch.get_stream());
      radar.GetInputView().PrefetchDevice(launch.get_stream());      

      timer.start();
      radar.PulseCompression();
      timer.stop();
    });

}
NVBENCH_BENCH_TYPES(simple_radar_pipeline_pulse_compression, NVBENCH_TYPE_AXES(radar_types))
  .add_int64_axis("Pulses", {128})
  .add_int64_axis("Channels", {16})
  .add_int64_axis("Samples", {9000})
  .add_int64_axis("Waveform Length", {1000});

template <typename ValueType>
void simple_radar_pipeline_three_pulse_canceller(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const index_t numPulses = static_cast<int>(state.get_int64("Pulses"));
  const index_t numChannels = static_cast<int>(state.get_int64("Channels"));
  const index_t numSamples = static_cast<int>(state.get_int64("Samples"));
  const index_t waveformLength = static_cast<int>(state.get_int64("Waveform Length"));

  state.exec( nvbench::exec_tag::timer, 
    [&numPulses, &numChannels, &numSamples, &waveformLength](nvbench::launch &launch, auto &timer) {
      auto radar = RadarPipeline(numPulses, numSamples, waveformLength, numChannels, launch.get_stream());
      radar.GetInputView().PrefetchDevice(launch.get_stream());      

      timer.start();
      radar.ThreePulseCanceller();
      timer.stop();
    });

}
NVBENCH_BENCH_TYPES(simple_radar_pipeline_three_pulse_canceller, NVBENCH_TYPE_AXES(radar_types))
  .add_int64_axis("Pulses", {128})
  .add_int64_axis("Channels", {16})
  .add_int64_axis("Samples", {9000})
  .add_int64_axis("Waveform Length", {1000});

template <typename ValueType>
void simple_radar_pipeline_doppler(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const index_t numPulses = static_cast<int>(state.get_int64("Pulses"));
  const index_t numChannels = static_cast<int>(state.get_int64("Channels"));
  const index_t numSamples = static_cast<int>(state.get_int64("Samples"));
  const index_t waveformLength = static_cast<int>(state.get_int64("Waveform Length"));

  state.exec( nvbench::exec_tag::timer, 
    [&numPulses, &numChannels, &numSamples, &waveformLength](nvbench::launch &launch, auto &timer) {
      auto radar = RadarPipeline(numPulses, numSamples, waveformLength, numChannels, launch.get_stream());
      radar.GetInputView().PrefetchDevice(launch.get_stream());      

      timer.start();
      radar.DopplerProcessing();
      timer.stop();
    });

}
NVBENCH_BENCH_TYPES(simple_radar_pipeline_doppler, NVBENCH_TYPE_AXES(radar_types))
  .add_int64_axis("Pulses", {128})
  .add_int64_axis("Channels", {16})
  .add_int64_axis("Samples", {9000})
  .add_int64_axis("Waveform Length", {1000});


template <typename ValueType>
void simple_radar_pipeline_cfar(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const index_t numPulses = static_cast<int>(state.get_int64("Pulses"));
  const index_t numChannels = static_cast<int>(state.get_int64("Channels"));
  const index_t numSamples = static_cast<int>(state.get_int64("Samples"));
  const index_t waveformLength = static_cast<int>(state.get_int64("Waveform Length"));

  state.exec( nvbench::exec_tag::timer, 
    [&numPulses, &numChannels, &numSamples, &waveformLength](nvbench::launch &launch, auto &timer) {
      auto radar = RadarPipeline(numPulses, numSamples, waveformLength, numChannels, launch.get_stream());
      radar.GetInputView().PrefetchDevice(launch.get_stream());      

      timer.start();
      radar.CFARDetections();
      timer.stop();
    });

}
NVBENCH_BENCH_TYPES(simple_radar_pipeline_cfar, NVBENCH_TYPE_AXES(radar_types))
  .add_int64_axis("Pulses", {128})
  .add_int64_axis("Channels", {16})
  .add_int64_axis("Samples", {9000})
  .add_int64_axis("Waveform Length", {1000});  

template <typename ValueType>
void simple_radar_pipeline_end_to_end(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  const index_t numPulses = static_cast<int>(state.get_int64("Pulses"));
  const index_t numChannels = static_cast<int>(state.get_int64("Channels"));
  const index_t numSamples = static_cast<int>(state.get_int64("Samples"));
  const index_t waveformLength = static_cast<int>(state.get_int64("Waveform Length"));

  state.exec( nvbench::exec_tag::timer, 
    [&numPulses, &numChannels, &numSamples, &waveformLength](nvbench::launch &launch, auto &timer) {
      auto radar = RadarPipeline(numPulses, numSamples, waveformLength, numChannels, launch.get_stream());
      radar.GetInputView().PrefetchDevice(launch.get_stream());      

      timer.start();
      radar.PulseCompression();
      radar.ThreePulseCanceller();
      radar.DopplerProcessing();      
      radar.CFARDetections();
      timer.stop();
    });

}
NVBENCH_BENCH_TYPES(simple_radar_pipeline_end_to_end, NVBENCH_TYPE_AXES(radar_types))
  .add_int64_axis("Pulses", {128})
  .add_int64_axis("Channels", {16})
  .add_int64_axis("Samples", {9000})
  .add_int64_axis("Waveform Length", {1000});    


