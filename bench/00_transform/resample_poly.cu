////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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
#include <nvbench/nvbench.cuh>
#include "matx/core/half_complex.h"
#include "matx/core/nvtx.h"

using namespace matx;

using resample_poly_types =
    nvbench::type_list<cuda::std::complex<float>, cuda::std::complex<double>, float, double>;

// 1D polyphase resampler.
//
// Axes (defaults can be overriden on the command line, e.g.
//   --axis "Up=[2,3,4,5]" --axis "Down=[1,2,3]" --axis "Signal Size=[131072,524288]"
// ):
//   - Up           : upsample factor (default 4)
//   - Down         : downsample factor (default 5)
//   - Signal Size  : input length in samples (default 512000)
//   - Filter Size  : filter length, or 0 to auto-compute as
//                    2 * 10 * max(up, down) + 1 (default: 0)
template <typename ValueType>
void resample_poly_1d(nvbench::state &state,
                     nvbench::type_list<ValueType>)
{
  cudaExecutor exec{0};

  const index_t up         = static_cast<index_t>(state.get_int64("Up"));
  const index_t down       = static_cast<index_t>(state.get_int64("Down"));
  const index_t signal_len = static_cast<index_t>(state.get_int64("Signal Size"));
  index_t       filter_len = static_cast<index_t>(state.get_int64("Filter Size"));
  if (filter_len <= 0) {
    const index_t half_len = 10 * std::max(up, down);
    filter_len = 2 * half_len + 1;
  }

  // Output length matches the one-shot resample_poly_impl convention:
  //   out_len = ceil(signal_len * up / down)
  const index_t out_len = (signal_len * up + down - 1) / down;

  auto in     = make_tensor<ValueType>({signal_len});
  auto filter = make_tensor<ValueType>({filter_len});
  auto out    = make_tensor<ValueType>({out_len});

  // Populate inputs with deterministic random data so first-run timing
  // doesn't get charged for page faults on the unified-memory fast path.
  (in     = random<ValueType>({signal_len}, NORMAL)).run(exec);
  (filter = random<ValueType>({filter_len}, NORMAL)).run(exec);

  in.PrefetchDevice(0);
  filter.PrefetchDevice(0);
  out.PrefetchDevice(0);

  // Warmup.
  (out = resample_poly(in, filter, up, down)).run(exec);
  exec.sync();

  MATX_NVTX_START_RANGE("resample_poly_1d", matx_nvxtLogLevels::MATX_NVTX_LOG_ALL, 1)
  state.exec(
      [&out, &in, &filter, up, down](nvbench::launch &launch) {
        (out = resample_poly(in, filter, up, down))
            .run(cudaExecutor(launch.get_stream()));
      });
  MATX_NVTX_END_RANGE(1)

  // Show the effective filter length (after the 0=auto sentinel resolves) and
  // throughput in input-samples/sec as native columns, so downstream tooling
  // can read them straight off the nvbench table/CSV without re-deriving.
  auto seconds = state.get_summary("Batch GPU").get_float64("value");

  auto &eff_fl = state.add_summary("matx/resample_poly/effective_filter_size");
  eff_fl.set_string("name", "Eff. Filter");
  eff_fl.set_string("description", "Effective filter length (resolves 0=auto)");
  eff_fl.set_int64("value", filter_len);

  auto &thr = state.add_summary("matx/resample_poly/in_msamp_per_sec");
  thr.set_string("name", "In Msamp/s");
  thr.set_string("hint", "item_rate");
  thr.set_string("description", "Million input samples per second");
  thr.set_float64("value", static_cast<double>(signal_len) / seconds / 1e6);
}
NVBENCH_BENCH_TYPES(resample_poly_1d, NVBENCH_TYPE_AXES(resample_poly_types))
    .add_int64_axis("Up",          {4})
    .add_int64_axis("Down",        {5})
    .add_int64_axis("Signal Size", {512000})
    // Filter Size = 0 (or any value <= 0) is a sentinel meaning "auto-compute
    // from up/down via 2*10*max(up,down)+1". The actual filter length used
    // shows up in the "Eff. Filter" summary column on every row.
    .add_int64_axis("Filter Size", {0});
