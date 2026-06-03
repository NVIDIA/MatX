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

/**
 * @file sarbp.cu
 * @brief SAR backprojection example using pre-processed CPHD data.
 *
 * Reads a .sarbp binary file produced by cphd_to_sarbp.py and runs the MatX
 * sar_bp operator to form a SAR image.  The output is written as a raw
 * single-precision complex binary file (interleaved real/imag, row-major).
 *
 * .sarbp file format (version 2)
 * ------------------------------
 * File header (256 bytes, little-endian):
 *   Offset  Size  Type      Field
 *   0       8     char[8]   magic  ("SARBP\x02\x00\x00")
 *   8       4     uint32    num_pulses
 *   12      4     uint32    num_range_bins
 *   16      4     uint32    image_width
 *   20      4     uint32    image_height
 *   24      8     float64   center_frequency  [Hz]
 *   32      8     float64   del_r             [m]
 *   40      8     float64   bandwidth         [Hz]
 *   48      8     float64   pixel_spacing     [m]
 *   56      8     float64   voxel_start_x     [m]  (first pixel centre, East)
 *   64      8     float64   voxel_start_y     [m]  (first pixel centre, North)
 *   72      8     float64   voxel_start_z     [m]  (Up, typically 0)
 *   80      8     float64   voxel_stride_x    [m]
 *   88      8     float64   voxel_stride_y    [m]
 *   96      4     uint32    flags  (bit 0: 1=FX domain, 0=range compressed;
 *                                     bit 1: 1=int16 samples, 0=complex64 samples)
 *   100     4     int32     sgn    (phase sign convention: -1 or +1)
 *   104     4     uint32    num_samples_raw   (original FX sample count)
 *   108     4     uint32    pulse_header_size (48 for complex64, 56 for int16)
 *   112     8     float64   prf               [Hz]
 *   120     8     float64   grazing_angle     [deg]
 *   128     128   -         reserved (zero-filled)
 *
 * Per-pulse record (repeated num_pulses times):
 *   Pulse header (48 or 56 bytes):
 *     0     8     float64   platform_pos_x  [m]
 *     8     8     float64   platform_pos_y  [m]
 *     16    8     float64   platform_pos_z  [m]
 *     24    8     float64   range_to_mcp    [m]
 *     32    8     float64   toa1            [s]  (differential one-way TOA, near edge)
 *     40    8     float64   toa2            [s]  (differential one-way TOA, far edge)
 *     48    8     float64   sample_scale    (int16 mode only; multiply int16 by
 *                                            this to recover complex float value)
 *   Samples:
 *     complex64 mode: num_range_bins * 8 bytes (interleaved float32 real/imag)
 *     int16 mode:     num_range_bins * 4 bytes (interleaved int16 real/imag)
 *
 * Usage:
 *   sarbp <input.sarbp> [output_file] [-u upsample_factor] [-w window]
 */

#include "matx.h"
#include <cuda/std/complex>
#include <cuda/cmath>
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_profiler_api.h>

using namespace matx;

using complex_t = cuda::std::complex<float>;

// Use up to this fraction of L2 for range profiles and phase lookup table
static constexpr double SARBP_AUTO_L2_TARGET_MULTIPLIER = 0.8;
static constexpr index_t SARBP_AUTO_BLOCK_GRANULARITY = 256;
static constexpr index_t SARBP_AUTO_MIN_BLOCK_SIZE = 256;

enum class BlockSizeMode {
  Auto,
  All,
  Manual
};

struct BlockSizeSelection {
  BlockSizeMode mode{BlockSizeMode::Auto};
  index_t manual_size{0};
};

static bool parse_index_arg(const std::string &arg, index_t &value, index_t min_value)
{
  index_t parsed{};
  const auto *begin = arg.data();
  const auto *end = arg.data() + arg.size();
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (arg.empty() || ec != std::errc{} || ptr != end || parsed < min_value) {
    return false;
  }

  value = parsed;
  return true;
}

static bool parse_block_size_arg(const std::string &arg, BlockSizeSelection &selection)
{
  if (arg == "auto") {
    selection = BlockSizeSelection{BlockSizeMode::Auto, 0};
    return true;
  }
  if (arg == "all") {
    selection = BlockSizeSelection{BlockSizeMode::All, 0};
    return true;
  }

  index_t parsed{};
  if (!parse_index_arg(arg, parsed, 0)) {
    return false;
  }

  if (parsed == 0) {
    // Preserve the old "-b 0" behavior as an alias for all pulses.
    selection = BlockSizeSelection{BlockSizeMode::All, 0};
  } else {
    selection = BlockSizeSelection{BlockSizeMode::Manual, parsed};
  }
  return true;
}

static index_t round_down_to_multiple(index_t value, index_t multiple)
{
  if (multiple <= 1) {
    return value;
  }
  return (value / multiple) * multiple;
}

static size_t get_phase_lut_bytes(index_t output_range_bins, const SarBpParams &params)
{
  if (!has_feature(params.features, SarBpFeature::PhaseLUTOptimization)) {
    return 0;
  }

  const size_t elem_size = (params.compute_type == SarBpComputeType::Double)
      ? sizeof(cuda::std::complex<double>)
      : sizeof(cuda::std::complex<float>);
  return static_cast<size_t>(output_range_bins) * elem_size;
}

static index_t choose_auto_block_size(index_t num_pulses, index_t output_range_bins,
                                      const SarBpParams &params,
                                      const cudaDeviceProp &device_prop)
{
  const size_t profile_bytes_per_pulse =
      static_cast<size_t>(output_range_bins) * sizeof(complex_t);
  if (num_pulses <= 0 || profile_bytes_per_pulse == 0 ||
      device_prop.l2CacheSize <= 0) {
    return num_pulses;
  }

  const size_t phase_lut_bytes = get_phase_lut_bytes(output_range_bins, params);
  const double l2_target_bytes =
      static_cast<double>(device_prop.l2CacheSize) * SARBP_AUTO_L2_TARGET_MULTIPLIER;
  double profile_budget_bytes = l2_target_bytes - static_cast<double>(phase_lut_bytes);
  const double min_profile_budget =
      static_cast<double>(profile_bytes_per_pulse) *
      static_cast<double>(SARBP_AUTO_MIN_BLOCK_SIZE);
  if (profile_budget_bytes < min_profile_budget) {
    profile_budget_bytes = min_profile_budget;
  }

  index_t block_size =
      static_cast<index_t>(profile_budget_bytes / static_cast<double>(profile_bytes_per_pulse));
  block_size = round_down_to_multiple(block_size, SARBP_AUTO_BLOCK_GRANULARITY);
  block_size = std::max(block_size, SARBP_AUTO_MIN_BLOCK_SIZE);
  return std::min(block_size, num_pulses);
}

// Aggregate of non-tensor state needed by run_bp_device(). Kept separate from
// the tensor and host-buffer parameters because most members are scalar
// configuration values whose call-site reads cleanly as a brace-initializer.
struct BpRunCtx {
  cudaStream_t stream;
  cudaExecutor &exec;
  index_t block_size;
  index_t num_pulses;
  index_t num_blocks;
  index_t output_range_bins;
  index_t num_samples_raw;
  index_t fft_size;
  index_t num_range_bins;
  index_t ifft_shift;
  index_t image_width;
  index_t image_height;
  index_t image_tiles;
  bool is_fx_domain;
  bool is_int16_mode;
  bool apply_window;
  bool taylor_fast_add_third_order;
  SarBpPixelZMode pixel_z_mode;  // compile-time pixel-z assumption to apply
  int sgn;
  bool do_warmup;
  std::string output_file;
  double del_r;
  SarBpParams params;
  double3 *h_positions;
  double *h_range_to_mcp;
  complex_t *h_range_profiles;
  int16_t *h_range_profiles_i16;
  float *h_ampsf;
};

// Run all device-side work (allocate block buffers, optional warmup, the
// main pulse-block BP loop, copy result out, print timings, write the
// output file). Templated on the platform-positions and range-to-mcp
// tensor types so callers can pass either tensor<double3> / tensor<double>
// or tensor<fltflt> / tensor<fltflt> (rank-2 / rank-1 for the fltflt path).
template <typename PosTensor, typename RtmTensor, typename VoxLocOp>
static int run_bp_device(PosTensor blk_positions, RtmTensor blk_rtm,
                         const VoxLocOp &voxel_locations, const BpRunCtx &ctx)
{
  using PosT = typename std::decay_t<decltype(blk_positions)>::value_type;
  constexpr bool pos_is_fltflt = std::is_same_v<PosT, matx::fltflt>;

  // The host buffers ctx.h_positions / ctx.h_range_to_mcp were declared with
  // double element types and either hold doubles (standard paths) or
  // fltflt-encoded bytes (after the in-place conversion done in main when
  // precision is fltflt). Since the byte sizes are identical between the two
  // representations, the cudaMemcpyAsync calls below are just byte copies
  // regardless of which device tensor type is being filled.
  static_assert(3 * sizeof(matx::fltflt) == sizeof(double3),
                "fltflt[3] must match double3 byte size for the in-place conversion to work");
  static_assert(sizeof(matx::fltflt) == sizeof(double),
                "fltflt must match double byte size for the in-place conversion to work");

  // Slice helper: 2D for fltflt path (shape [npulses, 3]) vs 1D for double3.
  auto pos_slice = [&](index_t npulses) {
    if constexpr (pos_is_fltflt) {
      return matx::slice(blk_positions, {0, 0}, {npulses, 3});
    } else {
      return matx::slice(blk_positions, {0},    {npulses});
    }
  };

  auto upload_positions = [&](auto &cur_positions, index_t p0, index_t npulses) {
    MATX_CUDA_CHECK(cudaMemcpyAsync(
        cur_positions.Data(),
        reinterpret_cast<const uint8_t *>(ctx.h_positions) + p0 * sizeof(double3),
        static_cast<size_t>(npulses) * sizeof(double3),
        cudaMemcpyHostToDevice, ctx.stream));
  };

  auto upload_rtm = [&](auto &cur_rtm, index_t p0, index_t npulses) {
    MATX_CUDA_CHECK(cudaMemcpyAsync(
        cur_rtm.Data(),
        reinterpret_cast<const uint8_t *>(ctx.h_range_to_mcp) + p0 * sizeof(double),
        static_cast<size_t>(npulses) * sizeof(double),
        cudaMemcpyHostToDevice, ctx.stream));
  };

  // Pre-allocate GPU block buffers (sized for largest block)
  auto blk_profiles = make_tensor<complex_t>({ctx.block_size, ctx.output_range_bins}, matx::MATX_DEVICE_MEMORY);
  auto blk_compressed = ctx.is_fx_domain
      ? make_tensor<complex_t>({ctx.block_size, ctx.fft_size}, matx::MATX_DEVICE_MEMORY)
      : make_tensor<complex_t>({1, 1});
  auto blk_fx = ctx.is_fx_domain
      ? make_tensor<complex_t>({ctx.block_size, ctx.num_samples_raw}, matx::MATX_DEVICE_MEMORY)
      : make_tensor<complex_t>({1, 1});
  // int16 interleaved I/Q samples and per-pulse scale (int16 mode only)
  auto blk_fx_i16 = (ctx.is_int16_mode && ctx.is_fx_domain)
      ? make_tensor<int16_t>({ctx.block_size, ctx.num_samples_raw * 2}, matx::MATX_DEVICE_MEMORY)
      : make_tensor<int16_t>({1, 1});
  auto blk_ampsf = ctx.is_int16_mode
      ? make_tensor<float>({ctx.block_size}, matx::MATX_DEVICE_MEMORY)
      : make_tensor<float>({1});

  // Image tensor --zeroed before first block
  auto image = make_tensor<complex_t>({ctx.image_height, ctx.image_width}, matx::MATX_DEVICE_MEMORY);
  (image = matx::zeros<complex_t>({ctx.image_height, ctx.image_width})).run(ctx.exec);

  auto run_bp_tiles = [&](auto &cur_profiles, auto &cur_positions, auto &cur_rtm) {
    for (index_t tile_y = 0; tile_y < ctx.image_tiles; tile_y++) {
      const index_t y0 = (ctx.image_height * tile_y) / ctx.image_tiles;
      const index_t y1 = (ctx.image_height * (tile_y + 1)) / ctx.image_tiles;
      for (index_t tile_x = 0; tile_x < ctx.image_tiles; tile_x++) {
        const index_t x0 = (ctx.image_width * tile_x) / ctx.image_tiles;
        const index_t x1 = (ctx.image_width * (tile_x + 1)) / ctx.image_tiles;

        auto cur_image = matx::slice(image, {y0, x0}, {y1, x1});
        auto cur_voxel_locations = matx::slice(voxel_locations, {y0, x0}, {y1, x1});
        auto bp = matx::experimental::sar_bp(
            cur_image, cur_profiles, cur_positions, cur_voxel_locations, cur_rtm, ctx.params);
        // Cross product of the optional TaylorFast third-order term and the
        // compile-time pixel-z assumption (variable / zero / fixed).
        auto run_with_z = [&](auto bp_props) {
          switch (ctx.pixel_z_mode) {
            case SarBpPixelZMode::Zero:
              (cur_image = bp_props.template props<matx::PropSarBpPixelZIsZero>()).run(ctx.exec);
              break;
            case SarBpPixelZMode::Fixed:
              (cur_image = bp_props.template props<matx::PropSarBpPixelZIsFixed>()).run(ctx.exec);
              break;
            case SarBpPixelZMode::Variable:
              (cur_image = bp_props).run(ctx.exec);
              break;
          }
        };
        if (ctx.taylor_fast_add_third_order) {
          run_with_z(bp.template props<matx::PropSarBpTaylorFastAddThirdOrder>());
        } else {
          run_with_z(bp);
        }
      }
    }
  };

  // Warmup: run kernels with correct tensor sizes to initialize FFT plans,
  // load kernels, etc. so that the timed run reflects steady-state performance.
  if (ctx.do_warmup) {
    std::cout << "Warming up kernels..." << std::flush;

    auto warmup_block = [&](index_t npulses) {
      auto cur_profiles  = matx::slice(blk_profiles,  {0, 0}, {npulses, ctx.output_range_bins});
      auto cur_positions = pos_slice(npulses);
      auto cur_rtm       = matx::slice(blk_rtm,       {0},    {npulses});

      (cur_profiles = matx::zeros<complex_t>({npulses, ctx.output_range_bins})).run(ctx.exec);
      // Generic zero-init via cudaMemset -- both layouts are 24 bytes/pulse
      // for positions (double3 == 3 * fltflt) and 8 bytes/pulse for rtm.
      MATX_CUDA_CHECK(cudaMemsetAsync(cur_positions.Data(), 0,
                      static_cast<size_t>(npulses) * sizeof(double3), ctx.stream));
      MATX_CUDA_CHECK(cudaMemsetAsync(cur_rtm.Data(), 0,
                      static_cast<size_t>(npulses) * sizeof(double), ctx.stream));

      if (ctx.is_fx_domain) {
        auto cur_fx = matx::slice(blk_fx, {0, 0}, {npulses, ctx.num_samples_raw});
        if (ctx.is_int16_mode) {
          // Warmup int16 -> complex<float> conversion kernel
          auto cur_fx_i16 = matx::slice(blk_fx_i16, {0, 0}, {npulses, ctx.num_samples_raw * 2});
          auto cur_ampsf = matx::slice(blk_ampsf, {0}, {npulses});
          (cur_fx_i16 = static_cast<int16_t>(0)).run(ctx.exec);
          (cur_ampsf = 1.0f).run(ctx.exec);
          auto i_vals = matx::slice(cur_fx_i16, {0, 0},
                                    {npulses, ctx.num_samples_raw * 2}, {1, 2});
          auto q_vals = matx::slice(cur_fx_i16, {0, 1},
                                    {npulses, ctx.num_samples_raw * 2}, {1, 2});
          auto ampsf_b = matx::clone<2>(cur_ampsf, {matxKeepDim, ctx.num_samples_raw});
          (cur_fx = matx::as_complex_float(
              matx::as_float(i_vals) * ampsf_b,
              matx::as_float(q_vals) * ampsf_b)).run(ctx.exec);
        }
        if (ctx.apply_window) {
          (cur_fx = cur_fx * matx::hamming<1>({npulses, ctx.num_samples_raw})).run(ctx.exec);
        }
        auto cur_compressed = matx::slice(blk_compressed, {0, 0}, {npulses, ctx.fft_size});
        if (ctx.sgn == -1) {
          (cur_compressed = matx::ifft(cur_profiles)).run(ctx.exec);
        } else {
          (cur_compressed = matx::fft(cur_profiles)).run(ctx.exec);
        }
        (cur_profiles = matx::fftshift1D(cur_compressed)).run(ctx.exec);
      }

      run_bp_tiles(cur_profiles, cur_positions, cur_rtm);
    };

    // Warmup with primary block size
    warmup_block(ctx.block_size);

    // Warmup with final block size if it differs (different FFT plan)
    const index_t last_block_size = ctx.num_pulses - (ctx.num_blocks - 1) * ctx.block_size;
    if (ctx.num_blocks > 1 && last_block_size != ctx.block_size) {
      warmup_block(last_block_size);
    }

    // Re-zero image after warmup
    (image = matx::zeros<complex_t>({ctx.image_height, ctx.image_width})).run(ctx.exec);
    ctx.exec.sync();
    std::cout << " done" << std::endl;
  }

  // Pre-allocate pinned host buffer for image output.
  const size_t num_pixels =
      static_cast<size_t>(ctx.image_height) * static_cast<size_t>(ctx.image_width);
  const size_t image_bytes = num_pixels * sizeof(complex_t);
  complex_t *h_image = nullptr;
  MATX_CUDA_CHECK(cudaHostAlloc(&h_image, image_bytes, cudaHostAllocDefault));

  std::cout << "Running backprojection (" << ctx.output_range_bins << " range bins, del_r="
            << ctx.del_r << " m)..." << std::endl;

  cudaEvent_t ev_start, ev_stop;
  MATX_CUDA_CHECK(cudaEventCreate(&ev_start));
  MATX_CUDA_CHECK(cudaEventCreate(&ev_stop));

  std::vector<cudaEvent_t> ev_bp_start(ctx.num_blocks);
  std::vector<cudaEvent_t> ev_bp_stop(ctx.num_blocks);
  for (index_t blk = 0; blk < ctx.num_blocks; blk++) {
    MATX_CUDA_CHECK(cudaEventCreate(&ev_bp_start[blk]));
    MATX_CUDA_CHECK(cudaEventCreate(&ev_bp_stop[blk]));
  }

  cudaProfilerStart();
  MATX_CUDA_CHECK(cudaEventRecord(ev_start, ctx.stream));

  for (index_t blk = 0; blk < ctx.num_blocks; blk++) {
    const index_t p0 = blk * ctx.block_size;
    const index_t p1 = std::min(p0 + ctx.block_size, ctx.num_pulses);
    const index_t npulses = p1 - p0;

    // Views into pre-allocated buffers (handle last block being smaller)
    auto cur_profiles  = matx::slice(blk_profiles,  {0, 0}, {npulses, ctx.output_range_bins});
    auto cur_positions = pos_slice(npulses);
    auto cur_rtm       = matx::slice(blk_rtm,       {0},    {npulses});

    // Upload platform positions and range_to_mcp for this block
    upload_positions(cur_positions, p0, npulses);
    upload_rtm(cur_rtm, p0, npulses);

    if (ctx.is_fx_domain) {
      auto cur_fx = matx::slice(blk_fx, {0, 0}, {npulses, ctx.num_samples_raw});

      if (ctx.is_int16_mode) {
        // Upload int16 I/Q pairs and per-pulse scale
        auto cur_fx_i16 = matx::slice(blk_fx_i16, {0, 0}, {npulses, ctx.num_samples_raw * 2});
        auto cur_ampsf = matx::slice(blk_ampsf, {0}, {npulses});
        MATX_CUDA_CHECK(cudaMemcpyAsync(cur_fx_i16.Data(),
                        ctx.h_range_profiles_i16 + p0 * ctx.num_samples_raw * 2,
                        static_cast<size_t>(npulses) * static_cast<size_t>(ctx.num_samples_raw) * 2 * sizeof(int16_t),
                        cudaMemcpyHostToDevice, ctx.stream));
        MATX_CUDA_CHECK(cudaMemcpyAsync(cur_ampsf.Data(),
                        ctx.h_ampsf + p0,
                        static_cast<size_t>(npulses) * sizeof(float),
                        cudaMemcpyHostToDevice, ctx.stream));

        // Strided views: even indices = I, odd indices = Q
        auto i_vals = matx::slice(cur_fx_i16, {0, 0},
                                  {npulses, ctx.num_samples_raw * 2}, {1, 2});
        auto q_vals = matx::slice(cur_fx_i16, {0, 1},
                                  {npulses, ctx.num_samples_raw * 2}, {1, 2});
        // Broadcast ampsf across range samples
        auto ampsf_b = matx::clone<2>(cur_ampsf, {matxKeepDim, ctx.num_samples_raw});

        // Convert int16 -> float, scale by AmpSF, combine to complex<float>
        if (ctx.apply_window) {
          auto win = matx::hamming<1>({npulses, ctx.num_samples_raw});
          (cur_fx = matx::as_complex_float(
              matx::as_float(i_vals) * ampsf_b * win,
              matx::as_float(q_vals) * ampsf_b * win)).run(ctx.exec);
        } else {
          (cur_fx = matx::as_complex_float(
              matx::as_float(i_vals) * ampsf_b,
              matx::as_float(q_vals) * ampsf_b)).run(ctx.exec);
        }
      } else {
        // Upload complex64 samples directly
        MATX_CUDA_CHECK(cudaMemcpyAsync(cur_fx.Data(),
                        ctx.h_range_profiles + p0 * ctx.num_samples_raw,
                        static_cast<size_t>(npulses) * static_cast<size_t>(ctx.num_samples_raw) * sizeof(complex_t),
                        cudaMemcpyHostToDevice, ctx.stream));

        if (ctx.apply_window) {
          (cur_fx = cur_fx * matx::hamming<1>({npulses, ctx.num_samples_raw})).run(ctx.exec);
          MATX_CUDA_CHECK(cudaGetLastError());
        }
      }

      // Zero only the padding region (middle of each row after ifftshift)
      const index_t pad_size = ctx.fft_size - ctx.num_samples_raw;
      if (pad_size > 0) {
        MATX_CUDA_CHECK(cudaMemset2DAsync(
            cur_profiles.Data() + (ctx.num_samples_raw - ctx.ifft_shift),
            static_cast<size_t>(ctx.fft_size) * sizeof(complex_t),
            0,
            static_cast<size_t>(pad_size) * sizeof(complex_t),
            static_cast<size_t>(npulses),
            ctx.stream));
      }

      // Second half of spectrum (indices shift..N-1) -> start of padded row
      MATX_CUDA_CHECK(cudaMemcpy2DAsync(
          cur_profiles.Data(),
          static_cast<size_t>(ctx.fft_size) * sizeof(complex_t),
          cur_fx.Data() + ctx.ifft_shift,
          static_cast<size_t>(ctx.num_samples_raw) * sizeof(complex_t),
          static_cast<size_t>(ctx.num_samples_raw - ctx.ifft_shift) * sizeof(complex_t),
          static_cast<size_t>(npulses),
          cudaMemcpyDeviceToDevice, ctx.stream));

      // First half of spectrum (indices 0..shift-1) -> end of padded row
      MATX_CUDA_CHECK(cudaMemcpy2DAsync(
          cur_profiles.Data() + (ctx.fft_size - ctx.ifft_shift),
          static_cast<size_t>(ctx.fft_size) * sizeof(complex_t),
          cur_fx.Data(),
          static_cast<size_t>(ctx.num_samples_raw) * sizeof(complex_t),
          static_cast<size_t>(ctx.ifft_shift) * sizeof(complex_t),
          static_cast<size_t>(npulses),
          cudaMemcpyDeviceToDevice, ctx.stream));

      // IFFT (SGN=-1) or FFT (SGN=+1) for range compression
      auto cur_compressed = matx::slice(blk_compressed, {0, 0}, {npulses, ctx.fft_size});
      if (ctx.sgn == -1) {
        (cur_compressed = matx::ifft(cur_profiles)).run(ctx.exec);
      } else {
        (cur_compressed = matx::fft(cur_profiles)).run(ctx.exec);
      }

      // fftshift to centre the zero-range bin (write result back to cur_profiles)
      (cur_profiles = matx::fftshift1D(cur_compressed)).run(ctx.exec);
    } else {
      // Pre-compressed: simple copy
      MATX_CUDA_CHECK(cudaMemcpyAsync(cur_profiles.Data(),
                      ctx.h_range_profiles + p0 * ctx.num_range_bins,
                      static_cast<size_t>(npulses) * static_cast<size_t>(ctx.num_range_bins) * sizeof(complex_t),
                      cudaMemcpyHostToDevice, ctx.stream));
    }

    // Backprojection - accumulates this block's pulses into image
    MATX_CUDA_CHECK(cudaEventRecord(ev_bp_start[blk], ctx.stream));
    run_bp_tiles(cur_profiles, cur_positions, cur_rtm);
    MATX_CUDA_CHECK(cudaEventRecord(ev_bp_stop[blk], ctx.stream));

    if (ctx.num_blocks > 1) {
      std::cout << "\r  Block " << (blk + 1) << " / " << ctx.num_blocks << std::flush;
    }
  }

  // Copy result to host buffer (included in timed region)
  MATX_CUDA_CHECK(cudaMemcpyAsync(h_image, image.Data(), image_bytes,
             cudaMemcpyDeviceToHost, ctx.stream));

  MATX_CUDA_CHECK(cudaEventRecord(ev_stop, ctx.stream));

  ctx.exec.sync();
  cudaProfilerStop();

  float elapsed_ms = 0;
  MATX_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
  if (ctx.num_blocks > 1) std::cout << std::endl;

  float bp_elapsed_ms = 0;
  for (index_t blk = 0; blk < ctx.num_blocks; blk++) {
    float blk_ms = 0;
    MATX_CUDA_CHECK(cudaEventElapsedTime(&blk_ms, ev_bp_start[blk], ev_bp_stop[blk]));
    bp_elapsed_ms += blk_ms;
  }

  const double total_backprojections =
      static_cast<double>(ctx.image_width) *
      static_cast<double>(ctx.image_height) *
      static_cast<double>(ctx.num_pulses);
  const double bp_gbp_per_sec = total_backprojections / (bp_elapsed_ms * 1e6);

  std::cout << "Full reconstruction completed in " << elapsed_ms / 1000.0f << " s (incl. data transfers and iffts, if applied)"
            << std::endl;
  std::cout << "  Giga Backprojections: " << total_backprojections / 1e9
            << " (num_pixels * num_pulses)" << std::endl;
  std::cout << "BP kernels only      : " << bp_elapsed_ms / 1000.0f << " s (summed over "
            << ctx.num_blocks << " block" << (ctx.num_blocks > 1 ? "s" : "") << ")" << std::endl;
  std::cout << "  Rate                : " << bp_gbp_per_sec << " Gbp/s" << std::endl;

  MATX_CUDA_CHECK(cudaEventDestroy(ev_start));
  MATX_CUDA_CHECK(cudaEventDestroy(ev_stop));
  for (index_t blk = 0; blk < ctx.num_blocks; blk++) {
    MATX_CUDA_CHECK(cudaEventDestroy(ev_bp_start[blk]));
    MATX_CUDA_CHECK(cudaEventDestroy(ev_bp_stop[blk]));
  }

  // Write raw binary file
  std::ofstream out(ctx.output_file, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "ERROR: cannot open " << ctx.output_file << " for writing" << std::endl;
    MATX_CUDA_CHECK(cudaFreeHost(h_image));
    return 1;
  }
  out.write(reinterpret_cast<const char *>(h_image),
            static_cast<std::streamsize>(image_bytes));
  out.close();

  std::cout << "Wrote " << ctx.image_height << " x " << ctx.image_width
            << " complex<float> image to " << ctx.output_file << std::endl;

  MATX_CUDA_CHECK(cudaFreeHost(h_image));
  return 0;
}

// ---------------------------------------------------------------------------
// .sarbp file header (matches Python writer)
// ---------------------------------------------------------------------------

static constexpr size_t SARBP_HEADER_SIZE = 256;
static constexpr char SARBP_MAGIC[8] = {'S', 'A', 'R', 'B', 'P', '\x02', '\x00', '\x00'};

struct SarbpFileHeader {
  uint32_t num_pulses;
  uint32_t num_range_bins;
  uint32_t image_width;
  uint32_t image_height;
  double   center_frequency;
  double   del_r;
  double   bandwidth;
  double   pixel_spacing;
  double   voxel_start_x;
  double   voxel_start_y;
  double   voxel_start_z;
  double   voxel_stride_x;
  double   voxel_stride_y;
  uint32_t flags;              // bit 0: 1=FX domain, 0=range compressed
  int32_t  sgn;                // phase sign convention (-1 or +1)
  uint32_t num_samples_raw;    // original FX sample count before processing
  uint32_t pulse_header_size;  // bytes per pulse header (48 for v2)
  double   prf;                // pulse repetition frequency [Hz]
  double   grazing_angle;      // grazing angle [deg]
};

static SarbpFileHeader read_sarbp_header(std::ifstream &f) {
  char header_buf[SARBP_HEADER_SIZE];
  f.read(header_buf, SARBP_HEADER_SIZE);
  if (!f) {
    std::cerr << "ERROR: failed to read .sarbp file header" << std::endl;
    std::exit(1);
  }

  // Verify magic
  if (std::memcmp(header_buf, SARBP_MAGIC, 8) != 0) {
    std::cerr << "ERROR: invalid .sarbp magic bytes" << std::endl;
    std::exit(1);
  }

  SarbpFileHeader h;
  std::memcpy(&h.num_pulses,       header_buf +  8, 4);
  std::memcpy(&h.num_range_bins,   header_buf + 12, 4);
  std::memcpy(&h.image_width,      header_buf + 16, 4);
  std::memcpy(&h.image_height,     header_buf + 20, 4);
  std::memcpy(&h.center_frequency, header_buf + 24, 8);
  std::memcpy(&h.del_r,            header_buf + 32, 8);
  std::memcpy(&h.bandwidth,        header_buf + 40, 8);
  std::memcpy(&h.pixel_spacing,    header_buf + 48, 8);
  std::memcpy(&h.voxel_start_x,    header_buf + 56, 8);
  std::memcpy(&h.voxel_start_y,    header_buf + 64, 8);
  std::memcpy(&h.voxel_start_z,    header_buf + 72, 8);
  std::memcpy(&h.voxel_stride_x,   header_buf + 80, 8);
  std::memcpy(&h.voxel_stride_y,   header_buf + 88, 8);
  std::memcpy(&h.flags,             header_buf + 96, 4);
  std::memcpy(&h.sgn,               header_buf + 100, 4);
  std::memcpy(&h.num_samples_raw,   header_buf + 104, 4);
  std::memcpy(&h.pulse_header_size, header_buf + 108, 4);
  std::memcpy(&h.prf,               header_buf + 112, 8);
  std::memcpy(&h.grazing_angle,     header_buf + 120, 8);
  return h;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  auto print_usage = []() {
    std::cerr
        << "Usage: sarbp <input.sarbp> [options]\n"
        << "\n"
        << "  <input.sarbp>          .sarbp input file from cphd_to_sarbp_input.py\n"
        << "  -o, --output <file>    Output file (default: input path with .raw extension)\n"
        << "  -u, --upsample <N>     Range upsample factor via zero-padding (default: 1)\n"
        << "  -w, --window <type>    Window for range compression: hamming, none (default: hamming)\n"
        << "  -b, --block-size <N|0|auto|all>\n"
        << "                          Pulses per block; 0/all use all pulses, auto uses an L2-cache heuristic (default: auto)\n"
        << "  --image-tiles <N>      Process image as N x N tiles (default: 1)\n"
        << "  --taylor-fast-third-order\n"
        << "                          Add the third-order term for --precision taylor_fast\n"
        << "  --warmup               Warmup GPU kernels and FFT plans before timed run\n"
        << "  --precision <type>     Compute precision: double, float, fltflt, mixed, taylor_fast (default: mixed)\n"
        << "  --pixel-z <mode>       Compile-time pixel-z assumption: variable, zero, fixed (default: variable)\n"
        << "  -h, --help             Print this help message and exit\n";
  };

  if (argc < 2) {
    print_usage();
    return 1;
  }

  std::string input_file;
  std::string output_file;
  int upsample_factor = 1;
  std::string window_type = "hamming";
  std::string block_size_arg = "auto";
  index_t image_tiles = 1;
  bool do_warmup = false;
  bool taylor_fast_add_third_order = false;
  std::string precision_type = "mixed";
  std::string pixel_z_arg = "variable";
  SarBpPixelZMode pixel_z_mode = SarBpPixelZMode::Variable;

  auto needs_value = [&](int i) -> bool {
    if (i + 1 >= argc) {
      std::cerr << "ERROR: option " << argv[i] << " requires a value" << std::endl;
      print_usage();
      return false;
    }
    return true;
  };

  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      print_usage();
      return 0;
    } else if (std::strcmp(argv[i], "-u") == 0 || std::strcmp(argv[i], "--upsample") == 0) {
      if (!needs_value(i)) return 1;
      upsample_factor = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-w") == 0 || std::strcmp(argv[i], "--window") == 0) {
      if (!needs_value(i)) return 1;
      window_type = argv[++i];
    } else if (std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) {
      if (!needs_value(i)) return 1;
      output_file = argv[++i];
    } else if (std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--block-size") == 0) {
      if (!needs_value(i)) return 1;
      block_size_arg = argv[++i];
    } else if (std::strcmp(argv[i], "--image-tiles") == 0) {
      if (!needs_value(i)) return 1;
      if (!parse_index_arg(argv[++i], image_tiles, 1)) {
        std::cerr << "ERROR: invalid image tile count '" << argv[i]
                  << "' (use a positive integer)" << std::endl;
        print_usage();
        return 1;
      }
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      do_warmup = true;
    } else if (std::strcmp(argv[i], "--taylor-fast-third-order") == 0) {
      taylor_fast_add_third_order = true;
    } else if (std::strcmp(argv[i], "--precision") == 0) {
      if (!needs_value(i)) return 1;
      precision_type = argv[++i];
    } else if (std::strcmp(argv[i], "--pixel-z") == 0) {
      if (!needs_value(i)) return 1;
      pixel_z_arg = argv[++i];
    } else if (argv[i][0] == '-') {
      std::cerr << "ERROR: unknown option '" << argv[i] << "'" << std::endl;
      print_usage();
      return 1;
    } else if (input_file.empty()) {
      input_file = argv[i];
    } else if (output_file.empty()) {
      output_file = argv[i];
    } else {
      std::cerr << "ERROR: unexpected positional argument '" << argv[i] << "'" << std::endl;
      print_usage();
      return 1;
    }
  }

  if (input_file.empty()) {
    std::cerr << "ERROR: missing required <input.sarbp> argument" << std::endl;
    print_usage();
    return 1;
  }

  BlockSizeSelection block_size_selection;
  if (!parse_block_size_arg(block_size_arg, block_size_selection)) {
    std::cerr << "ERROR: invalid block size '" << block_size_arg
              << "' (use a positive integer, 0, auto, or all)" << std::endl;
    print_usage();
    return 1;
  }

  if (output_file.empty()) {
    auto dot = input_file.rfind('.');
    output_file = (dot != std::string::npos ? input_file.substr(0, dot) : input_file) + ".raw";
  }

  if (taylor_fast_add_third_order && precision_type != "taylor_fast") {
    std::cerr << "ERROR: --taylor-fast-third-order requires --precision taylor_fast" << std::endl;
    print_usage();
    return 1;
  }

  if (pixel_z_arg == "variable") {
    pixel_z_mode = SarBpPixelZMode::Variable;
  } else if (pixel_z_arg == "zero") {
    pixel_z_mode = SarBpPixelZMode::Zero;
  } else if (pixel_z_arg == "fixed") {
    pixel_z_mode = SarBpPixelZMode::Fixed;
  } else {
    std::cerr << "ERROR: invalid --pixel-z '" << pixel_z_arg
              << "' (use variable, zero, or fixed)" << std::endl;
    print_usage();
    return 1;
  }

  // -------------------------------------------------------------------
  // Read .sarbp file header
  // -------------------------------------------------------------------
  std::ifstream fin(input_file, std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "ERROR: cannot open " << input_file << std::endl;
    return 1;
  }

  auto hdr = read_sarbp_header(fin);

  const auto num_pulses     = static_cast<index_t>(hdr.num_pulses);
  const auto num_range_bins = static_cast<index_t>(hdr.num_range_bins);
  const auto image_width    = static_cast<index_t>(hdr.image_width);
  const auto image_height   = static_cast<index_t>(hdr.image_height);
  const bool is_fx_domain   = (hdr.flags & 0x1) != 0;
  const bool is_int16_mode  = (hdr.flags & 0x2) != 0;
  const int sgn             = hdr.sgn;

  if (num_pulses <= 0 || num_range_bins <= 0 ||
      image_width <= 0 || image_height <= 0) {
    std::cerr << "ERROR: invalid .sarbp dimensions: pulses=" << num_pulses
              << ", samples=" << num_range_bins
              << ", image=" << image_height << " x " << image_width
              << std::endl;
    return 1;
  }

  if (image_tiles > image_width || image_tiles > image_height) {
    std::cerr << "ERROR: --image-tiles " << image_tiles
              << " exceeds image dimensions " << image_height << " x "
              << image_width << std::endl;
    return 1;
  }

  std::cout << "Input file       : " << input_file << std::endl;
  std::cout << "Pulses           : " << num_pulses << std::endl;
  std::cout << "Sample format    : " << (is_int16_mode ? "int16" : "complex64") << std::endl;
  std::cout << "Samples          : " << num_range_bins
            << (is_fx_domain ? " (FX domain)" : " (range compressed)") << std::endl;
  std::cout << "Image size       : " << image_height << " x " << image_width << std::endl;
  std::cout << "Center frequency : " << hdr.center_frequency / 1e9 << " GHz" << std::endl;
  std::cout << "del_r            : " << hdr.del_r << " m" << std::endl;
  std::cout << "Pixel spacing    : " << hdr.pixel_spacing << " m" << std::endl;
  std::cout << "Voxel start      : (" << hdr.voxel_start_x << ", "
            << hdr.voxel_start_y << ", " << hdr.voxel_start_z << ") m" << std::endl;
  std::cout << "Voxel stride     : (" << hdr.voxel_stride_x << ", "
            << hdr.voxel_stride_y << ") m" << std::endl;
  if (hdr.prf > 0) {
    std::cout << "PRF              : " << hdr.prf << " Hz" << std::endl;
  }
  if (hdr.grazing_angle > 0) {
    std::cout << "Grazing angle    : " << hdr.grazing_angle << " deg" << std::endl;
  }
  std::cout << "Output file      : " << output_file << std::endl;

  if (is_int16_mode && !is_fx_domain) {
    std::cerr << "ERROR: int16 sample mode is only supported for FX-domain data"
              << std::endl;
    return 1;
  }

  MATX_ENTER_HANDLER();

  // -------------------------------------------------------------------
  // Read per-pulse data (platform positions, range_to_mcp, samples)
  // -------------------------------------------------------------------
  // complex_t is declared at file scope so the run_bp_device helper can use it.

  double3 *h_positions = nullptr;
  double *h_range_to_mcp = nullptr;
  float *h_ampsf = nullptr;  // per-pulse scale (int16 mode only)
  complex_t *h_range_profiles = nullptr;
  int16_t *h_range_profiles_i16 = nullptr;

  MATX_CUDA_CHECK(cudaHostAlloc(&h_positions,
      static_cast<size_t>(num_pulses) * sizeof(double3), cudaHostAllocDefault));
  MATX_CUDA_CHECK(cudaHostAlloc(&h_range_to_mcp,
      static_cast<size_t>(num_pulses) * sizeof(double), cudaHostAllocDefault));

  if (is_int16_mode) {
    MATX_CUDA_CHECK(cudaHostAlloc(&h_ampsf,
        static_cast<size_t>(num_pulses) * sizeof(float), cudaHostAllocDefault));
    // int16 pairs: num_range_bins * 2 int16 per pulse
    MATX_CUDA_CHECK(cudaHostAlloc(&h_range_profiles_i16,
        static_cast<size_t>(num_pulses) * static_cast<size_t>(num_range_bins) * 2 * sizeof(int16_t),
        cudaHostAllocDefault));
  } else {
    MATX_CUDA_CHECK(cudaHostAlloc(&h_range_profiles,
        static_cast<size_t>(num_pulses) * static_cast<size_t>(num_range_bins) * sizeof(complex_t),
        cudaHostAllocDefault));
  }

  const size_t pulse_hdr_size = hdr.pulse_header_size > 0 ? hdr.pulse_header_size : 48;
  // The spec allows only 48 (complex64) or 56 (int16) bytes. Any other value
  // is a malformed / crafted file. The per-pulse stack buffer below is
  // sized for the 7-double maximum, so an unchecked larger value would
  // overflow the buffer on the fin.read() that follows.
  constexpr size_t MAX_PULSE_HDR_BYTES = sizeof(double) * 7;  // == 56
  if (pulse_hdr_size != 48 && pulse_hdr_size != MAX_PULSE_HDR_BYTES) {
    std::cerr << "ERROR: invalid pulse_header_size " << pulse_hdr_size
              << " (expected 48 or 56)" << std::endl;
    return 1;
  }
  const size_t samples_bytes = is_int16_mode
      ? static_cast<size_t>(num_range_bins) * 2 * sizeof(int16_t)
      : static_cast<size_t>(num_range_bins) * sizeof(complex_t);

  for (index_t i = 0; i < num_pulses; i++) {
    // Pulse header: pos (3), range_to_mcp, toa1, toa2, [ampsf if int16 mode]
    double pulse_hdr[7] = {};
    fin.read(reinterpret_cast<char *>(pulse_hdr),
             static_cast<std::streamsize>(pulse_hdr_size));
    if (!fin) {
      std::cerr << "ERROR: unexpected end of file at pulse " << i << std::endl;
      return 1;
    }
    h_positions[i] = double3{pulse_hdr[0], pulse_hdr[1], pulse_hdr[2]};
    h_range_to_mcp[i] = pulse_hdr[3];
    // pulse_hdr[4] = toa1, pulse_hdr[5] = toa2

    if (is_int16_mode) {
      h_ampsf[i] = static_cast<float>(
          (pulse_hdr_size >= 56) ? pulse_hdr[6] : 1.0);
      fin.read(reinterpret_cast<char *>(
                   h_range_profiles_i16 + i * num_range_bins * 2),
               static_cast<std::streamsize>(samples_bytes));
    } else {
      fin.read(reinterpret_cast<char *>(&h_range_profiles[i * num_range_bins]),
               static_cast<std::streamsize>(samples_bytes));
    }
    if (!fin) {
      std::cerr << "ERROR: unexpected end of file reading samples for pulse " << i << std::endl;
      return 1;
    }
  }
  fin.close();
  std::cout << "Loaded " << num_pulses << " pulses from .sarbp file" << std::endl;

  // If the user selected the fltflt precision, convert the platform positions
  // and range_to_mcp values in-place from double to fltflt. Both types are
  // 8 bytes per scalar (double = 8 bytes, fltflt = 2 * 4 bytes); see the
  // static_asserts in run_bp_device(). We perform the type-punning via
  // unsigned char* + std::memcpy rather than reinterpret_cast through
  // double*/fltflt*, because the latter would alias the same storage as two
  // incompatible types and is undefined behaviour under strict aliasing.
  // unsigned char* may alias any object type, and std::memcpy of
  // trivially-copyable types is the standards-blessed way to bit-cast.
  const bool use_fltflt_platform_inputs =
      precision_type == "fltflt";
  if (use_fltflt_platform_inputs) {
    auto *pos_bytes = reinterpret_cast<unsigned char *>(h_positions);
    for (size_t i = 0; i < static_cast<size_t>(num_pulses) * 3; i++) {
      unsigned char *slot = pos_bytes + i * sizeof(double);
      double d;
      std::memcpy(&d, slot, sizeof(double));      // read double
      const matx::fltflt ff(d);
      std::memcpy(slot, &ff, sizeof(matx::fltflt)); // overwrite as fltflt
    }
    auto *rtm_bytes = reinterpret_cast<unsigned char *>(h_range_to_mcp);
    for (size_t i = 0; i < static_cast<size_t>(num_pulses); i++) {
      unsigned char *slot = rtm_bytes + i * sizeof(double);
      double d;
      std::memcpy(&d, slot, sizeof(double));
      const matx::fltflt ff(d);
      std::memcpy(slot, &ff, sizeof(matx::fltflt));
    }
  }

  // -------------------------------------------------------------------
  // Compute range compression / upsampling parameters
  // -------------------------------------------------------------------
  const index_t num_samples_raw = hdr.num_samples_raw > 0
      ? static_cast<index_t>(hdr.num_samples_raw) : num_range_bins;
  const index_t fft_size = is_fx_domain
      ? (upsample_factor > 1
          ? cuda::next_power_of_two(static_cast<index_t>(num_samples_raw) * static_cast<index_t>(upsample_factor))
          : num_samples_raw)
      : num_range_bins;
  const index_t output_range_bins = is_fx_domain ? fft_size : num_range_bins;
  const double del_r = is_fx_domain
      ? hdr.del_r * static_cast<double>(num_samples_raw) / static_cast<double>(fft_size)
      : hdr.del_r;

  if (is_fx_domain) {
    std::cout << "Range compression:" << std::endl;
    std::cout << "  FX samples     : " << num_samples_raw << std::endl;
    std::cout << "  Upsample factor: " << upsample_factor << std::endl;
    std::cout << "  FFT size       : " << fft_size << std::endl;
    std::cout << "  SGN            : " << sgn << std::endl;
    std::cout << "  Window         : " << window_type << std::endl;
    std::cout << "  Output bins    : " << fft_size << std::endl;
    std::cout << "  del_r          : " << hdr.del_r << " -> " << del_r << " m" << std::endl;
  } else if (upsample_factor > 1) {
    std::cout << "WARNING: upsample factor ignored for pre-compressed range data" << std::endl;
  }

  // -------------------------------------------------------------------
  // Construct voxel grid
  // -------------------------------------------------------------------
  const auto voxel_end_x = hdr.voxel_start_x + hdr.voxel_stride_x * static_cast<double>(image_width - 1);
  const auto voxel_end_y = hdr.voxel_start_y + hdr.voxel_stride_y * static_cast<double>(image_height - 1);

  auto pix_coords_x = matx::linspace<float>(
      static_cast<float>(hdr.voxel_start_x),
      static_cast<float>(voxel_end_x),
      image_width);
  auto pix_coords_y = matx::linspace<float>(
      static_cast<float>(hdr.voxel_start_y),
      static_cast<float>(voxel_end_y),
      image_height);

  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matxKeepDim});
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matxKeepDim, image_width});

  auto voxel_locations = matx::zipvec(
      pix_coords_xclone, pix_coords_yclone,
      matx::zeros<float>({image_height, image_width}));

  // -------------------------------------------------------------------
  // Configure sar_bp parameters
  // -------------------------------------------------------------------
  SarBpParams params;
  if (precision_type == "double") {
    params.compute_type = SarBpComputeType::Double;
  } else if (precision_type == "float") {
    params.compute_type = SarBpComputeType::Float;
  } else if (precision_type == "fltflt") {
    params.compute_type = SarBpComputeType::FloatFloat;
  } else if (precision_type == "mixed") {
    params.compute_type = SarBpComputeType::Mixed;
  } else if (precision_type == "taylor_fast") {
    params.compute_type = SarBpComputeType::TaylorFast;
  } else {
    std::cerr << "ERROR: unknown precision type '" << precision_type
              << "' (use double, float, fltflt, mixed, or taylor_fast)" << std::endl;
    return 1;
  }
  if (params.compute_type != SarBpComputeType::Double) {
    params.features = SarBpFeature::PhaseLUTOptimization;
  }
  params.center_frequency = (sgn >= 0) ? -hdr.center_frequency : hdr.center_frequency;
  params.del_r = del_r;

  // -------------------------------------------------------------------
  // Block processing: range compression (if FX) + backprojection
  // -------------------------------------------------------------------
  size_t l2_cache_bytes = 0;
  const size_t profile_bytes_per_pulse =
      static_cast<size_t>(output_range_bins) * sizeof(complex_t);
  const size_t phase_lut_bytes = get_phase_lut_bytes(output_range_bins, params);

  index_t block_size = num_pulses;
  if (block_size_selection.mode == BlockSizeMode::Auto) {
    int device = 0;
    cudaDeviceProp device_prop{};
    MATX_CUDA_CHECK(cudaGetDevice(&device));
    MATX_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device));
    l2_cache_bytes = static_cast<size_t>(device_prop.l2CacheSize);
    block_size = choose_auto_block_size(num_pulses, output_range_bins, params, device_prop);
  } else if (block_size_selection.mode == BlockSizeMode::Manual) {
    block_size = std::min(block_size_selection.manual_size, num_pulses);
  }
  const index_t num_blocks = (num_pulses + block_size - 1) / block_size;

  std::cout << "Block size       : " << block_size << " pulses ";
  if (block_size_selection.mode == BlockSizeMode::Auto) {
    std::cout << "(auto, ";
  } else if (block_size_selection.mode == BlockSizeMode::All) {
    std::cout << "(all, ";
  } else {
    std::cout << "(manual, ";
  }
  std::cout << num_blocks << " block" << (num_blocks > 1 ? "s" : "") << ")" << std::endl;
  if (block_size_selection.mode == BlockSizeMode::Auto) {
    std::cout << "  Auto heuristic : L2 "
              << static_cast<double>(l2_cache_bytes) / (1024.0 * 1024.0)
              << " MiB, target " << SARBP_AUTO_L2_TARGET_MULTIPLIER
              << "x L2, profiles "
              << static_cast<double>(profile_bytes_per_pulse) / 1024.0
              << " KiB/pulse, phase LUT "
              << static_cast<double>(phase_lut_bytes) / (1024.0 * 1024.0)
              << " MiB" << std::endl;
  }
  std::cout << "BP precision     : " << precision_type;
  if (precision_type == "taylor_fast") {
    std::cout << (taylor_fast_add_third_order ? " (third order)" : " (second order)");
  }
  std::cout << std::endl;
  std::cout << "Image tiles      : " << image_tiles << " x " << image_tiles
            << std::endl;

  cudaStream_t stream;
  MATX_CUDA_CHECK(cudaStreamCreate(&stream));
  cudaExecutor exec{stream};

  const bool apply_window = is_fx_domain && window_type != "none";
  const index_t ifft_shift = num_samples_raw / 2;

  // Bundle non-tensor state for run_bp_device(). The platform-positions and
  // range-to-mcp tensors are allocated below with a type that matches
  // `use_fltflt_platform_inputs`: `tensor<double3>` / `tensor<double>` for the
  // standard paths, or `tensor<fltflt>` / `tensor<fltflt>` for the FloatFloat
  // path. In the fltflt case we already converted the pinned host buffers
  // in-place from double to fltflt right after the file read above (same byte
  // layout, no extra storage). The kernel
  // template (`SarBp<...>`) already handles both rank-1 (vector-of-double3
  // / -fltflt3-style) and rank-2 (matrix of [pulses, 3]) layouts for
  // platform_positions, so run_bp_device() branches internally only on the
  // slice and cudaMemcpyAsync source casts.
  const BpRunCtx ctx{
      .stream                = stream,
      .exec                  = exec,
      .block_size            = block_size,
      .num_pulses            = num_pulses,
      .num_blocks            = num_blocks,
      .output_range_bins     = output_range_bins,
      .num_samples_raw       = num_samples_raw,
      .fft_size              = fft_size,
      .num_range_bins        = num_range_bins,
      .ifft_shift            = ifft_shift,
      .image_width           = image_width,
      .image_height          = image_height,
      .image_tiles           = image_tiles,
      .is_fx_domain          = is_fx_domain,
      .is_int16_mode         = is_int16_mode,
      .apply_window          = apply_window,
      .taylor_fast_add_third_order = taylor_fast_add_third_order,
      .pixel_z_mode          = pixel_z_mode,
      .sgn                   = sgn,
      .do_warmup             = do_warmup,
      .output_file           = output_file,
      .del_r                 = del_r,
      .params                = params,
      .h_positions           = h_positions,
      .h_range_to_mcp        = h_range_to_mcp,
      .h_range_profiles      = h_range_profiles,
      .h_range_profiles_i16  = h_range_profiles_i16,
      .h_ampsf               = h_ampsf,
  };

  int dev_status;
  if (use_fltflt_platform_inputs) {
    auto blk_positions = make_tensor<matx::fltflt>({block_size, 3}, matx::MATX_DEVICE_MEMORY);
    auto blk_rtm       = make_tensor<matx::fltflt>({block_size},    matx::MATX_DEVICE_MEMORY);
    dev_status = run_bp_device(blk_positions, blk_rtm, voxel_locations, ctx);
  } else {
    auto blk_positions = make_tensor<double3>({block_size}, matx::MATX_DEVICE_MEMORY);
    auto blk_rtm       = make_tensor<double>({block_size},  matx::MATX_DEVICE_MEMORY);
    dev_status = run_bp_device(blk_positions, blk_rtm, voxel_locations, ctx);
  }
  if (dev_status != 0) return dev_status;


  if (h_positions) MATX_CUDA_CHECK(cudaFreeHost(h_positions));
  if (h_range_to_mcp) MATX_CUDA_CHECK(cudaFreeHost(h_range_to_mcp));
  if (h_range_profiles) MATX_CUDA_CHECK(cudaFreeHost(h_range_profiles));
  if (h_range_profiles_i16) MATX_CUDA_CHECK(cudaFreeHost(h_range_profiles_i16));
  if (h_ampsf) MATX_CUDA_CHECK(cudaFreeHost(h_ampsf));
  matx::ClearCachesAndAllocations();
  MATX_CUDA_CHECK(cudaStreamDestroy(stream));
  MATX_EXIT_HANDLER();

  return 0;
}
