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
#include <algorithm>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

using namespace matx;

/**
 * Window Functions Example
 *
 * Generates every window function MatX provides, computes their frequency responses with a zero-padded
 * FFT, and prints each window's peak sidelobe level, 3 dB mainlobe width, and equivalent noise bandwidth.
 * A rectangular window (no windowing) is included as a baseline.
 *
 * When MatX is built with MATX_EN_VISUALIZATION=ON (which requires the plotly Python package), the
 * responses are also written to an interactive plot, window_functions.html, with one panel per window.
 */

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  const index_t win_len = 64;
  // Zero padding interpolates the response between the window's DFT bins so the sidelobes are resolved
  const index_t nfft = 4096;
  // The windows are real, so the FFT returns only the nfft / 2 + 1 non-negative frequency bins. Bin k is at
  // k / nfft cycles/sample, which reaches Nyquist (0.5) only when nfft is even.
  const index_t nfreq = nfft / 2 + 1;
  cudaExecutor exec{};

  // Generate each window into its own tensor, so adding a window takes only another add_window call
  std::vector<std::string> names;
  std::vector<tensor_t<float, 1>> generated;
  auto add_window = [&](const std::string &name, const auto &op) {
    names.push_back(name);
    generated.push_back(make_tensor<float>({win_len}));
    (generated.back() = op).run(exec);
  };

  add_window("Rectangular", ones<float>({win_len}));
  add_window("Bartlett", bartlett<0>({win_len}));
  add_window("Hanning", hanning<0>({win_len}));
  add_window("Hamming", hamming<0>({win_len}));
  add_window("Blackman", blackman<0>({win_len}));
  add_window("Flat top", flattop<0>({win_len}));
  // nbar is chosen so that nbar >= 2*A^2 + 0.5 with A = acosh(10^(sll/20)) / pi, which keeps each taper monotonic
  add_window("Taylor (nbar = 4, sll = 30 dB)", taylor<0>({win_len}, {.nbar = 4, .sll = 30}));
  add_window("Taylor (nbar = 9, sll = 50 dB)", taylor<0>({win_len}, {.nbar = 9, .sll = 50}));

  // Gather the windows into the rows of one tensor so the FFT and reductions below run batched
  const index_t num_windows = static_cast<index_t>(names.size());
  auto windows = make_tensor<float>({num_windows, win_len});
  for (index_t i = 0; i < num_windows; i++) {
    (slice<1>(windows, {i, 0}, {matxDropDim, matxEnd}) = generated[static_cast<size_t>(i)]).run(exec);
  }

  auto spectrum = make_tensor<cuda::std::complex<float>>({num_windows, nfreq});
  auto dc_gain = make_tensor<float>({num_windows});
  auto energy = make_tensor<float>({num_windows});
  auto enbw = make_tensor<float>({num_windows});
  auto peak = make_tensor<float>({num_windows});
  auto response_db = make_tensor<float>({num_windows, nfreq});
  auto freqs = make_tensor<float>({nfreq});

  (spectrum = fft(windows, nfft)).run(exec);
  // The response at DC is the sum of the window's samples
  (dc_gain = sum(windows, {1})).run(exec);
  (energy = sum(windows * windows, {1})).run(exec);
  // Equivalent noise bandwidth in DFT bins: the width of a rectangular filter that passes the same noise power.
  // It is defined with the DC gain, so it also holds for windows with negative samples.
  (enbw = static_cast<float>(win_len) * energy / (dc_gain * dc_gain)).run(exec);

  // Normalize each response to 0 dB at its peak. A window with no negative samples peaks at DC, but one with
  // negative samples, such as the flat top, can peak slightly away from DC.
  (peak = max(abs(spectrum), {1})).run(exec);
  // Clamp at -240 dB (a magnitude ratio of 1e-12) so exact nulls stay finite without biasing any other value
  (response_db = 20.0f * log10(max(abs(spectrum) / clone<2>(peak, {matxKeepDim, nfreq}), 1e-12f))).run(exec);
  (freqs = range<0>({nfreq}, 0.0f, 1.0f / static_cast<float>(nfft))).run(exec);
  exec.sync();

  printf("%-32s %18s %18s %12s\n", "Window", "Peak sidelobe (dB)", "3 dB width (bins)", "ENBW (bins)");
  std::vector<float> peak_sidelobe(static_cast<size_t>(num_windows));
  for (index_t i = 0; i < num_windows; i++) {
    // Move past the top of the mainlobe (flat-top windows ripple there), walk down to the first null, then
    // take the largest response beyond it as the peak sidelobe level
    index_t k = 0;
    while (k + 1 < nfreq && response_db(i, k) > -6.0f) {
      k++;
    }
    while (k + 1 < nfreq && response_db(i, k + 1) < response_db(i, k)) {
      k++;
    }
    float psl = response_db(i, k);
    for (index_t j = k; j < nfreq; j++) {
      psl = std::max(psl, response_db(i, j));
    }
    peak_sidelobe[static_cast<size_t>(i)] = psl;

    index_t half_power = 0;
    while (half_power < nfreq && response_db(i, half_power) > -3.0103f) {
      half_power++;
    }
    // Two-sided width of the -3 dB mainlobe in units of the window's own DFT bin spacing (1 / win_len cycles/sample)
    const double width_bins = 2.0 * static_cast<double>(half_power) * static_cast<double>(win_len) / static_cast<double>(nfft);

    printf("%-32s %18.2f %18.2f %12.2f\n", names[static_cast<size_t>(i)].c_str(), static_cast<double>(psl),
           width_bins, static_cast<double>(enbw(i)));
  }

#if MATX_ENABLE_VIZ
  {
    using namespace pybind11::literals;
    // One panel per window on shared axes: overlaying this many responses would hide their sidelobes.
    // The figure is built with plotly directly since viz::line plots against the sample index.
    const int cols = 4;
    const int rows = static_cast<int>((num_windows + cols - 1) / cols);
    const char *surface = "#fcfcfb";
    const char *text_primary = "#0b0b0b";
    const char *text_secondary = "#52514e";
    const char *gridline = "#e4e3df";
    const char *series = "#2a78d6";

    auto pb = std::make_unique<detail::MatXPybind>();
    auto go = pybind11::module_::import("plotly.graph_objects");
    auto subplots = pybind11::module_::import("plotly.subplots");

    pybind11::list titles;
    for (index_t i = 0; i < num_windows; i++) {
      titles.append(std::format("{}<br><span style='font-size:12px;color:{}'>peak sidelobe {:.1f} dB</span>",
                                names[static_cast<size_t>(i)], text_secondary, peak_sidelobe[static_cast<size_t>(i)]));
    }

    auto fig = subplots.attr("make_subplots")(
        "rows"_a = rows, "cols"_a = cols, "shared_xaxes"_a = true, "shared_yaxes"_a = true,
        "subplot_titles"_a = titles, "horizontal_spacing"_a = 0.025, "vertical_spacing"_a = 0.14,
        "x_title"_a = "Normalized frequency (cycles/sample)", "y_title"_a = "Magnitude (dB, normalized to peak)");

    auto np_freqs = pb->TensorViewToNumpy(freqs);
    for (index_t i = 0; i < num_windows; i++) {
      auto row = slice<1>(response_db, {i, 0}, {matxDropDim, matxEnd});
      fig.attr("add_trace")(
          go.attr("Scatter")("x"_a = np_freqs, "y"_a = pb->TensorViewToNumpy(row), "mode"_a = "lines",
                             "name"_a = names[static_cast<size_t>(i)],
                             "line"_a = pybind11::dict("color"_a = series, "width"_a = 2),
                             "hovertemplate"_a = "%{x:.3f} cycles/sample<br>%{y:.1f} dB<extra></extra>"),
          "row"_a = i / cols + 1, "col"_a = i % cols + 1);
    }

    fig.attr("update_xaxes")("range"_a = pybind11::make_tuple(0, 0.5), "gridcolor"_a = gridline, "gridwidth"_a = 1,
                             "zeroline"_a = false, "linecolor"_a = gridline, "tickfont"_a = pybind11::dict("color"_a = text_secondary));
    fig.attr("update_yaxes")("range"_a = pybind11::make_tuple(-120, 5), "dtick"_a = 20, "gridcolor"_a = gridline,
                             "gridwidth"_a = 1, "zeroline"_a = false, "linecolor"_a = gridline,
                             "tickfont"_a = pybind11::dict("color"_a = text_secondary));
    fig.attr("update_layout")(
        "title"_a = std::format("Window frequency responses ({} samples, {}-point FFT)", win_len, nfft),
        "showlegend"_a = false, "height"_a = 300 * rows + 160, "hovermode"_a = "x",
        "margin"_a = pybind11::dict("t"_a = 130),
        "plot_bgcolor"_a = surface, "paper_bgcolor"_a = surface,
        "font"_a = pybind11::dict("color"_a = text_primary, "size"_a = 13));
    fig.attr("write_html")("window_functions.html");
    printf("\nWrote window_functions.html\n");
  }
#else
  printf("\nNot writing a plot since visualizations are disabled (configure with MATX_EN_VISUALIZATION=ON)\n");
#endif

  MATX_CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
  return 0;
}
