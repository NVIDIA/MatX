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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_profiler_api.h>

using namespace matx;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "   \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// .sarbp file header (matches Python writer)
// ---------------------------------------------------------------------------

static constexpr size_t SARBP_HEADER_SIZE = 256;
static constexpr char SARBP_MAGIC[8] = {'S', 'A', 'R', 'B', 'P', '\x02', '\x00', '\x00'};

static index_t next_pow2(index_t n) {
  index_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

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
        << "  <input.sarbp>          .sarbp file from cphd_to_sarbp.py\n"
        << "  -o, --output <file>    Output file (default: input path with .raw extension)\n"
        << "  -u, --upsample <N>     Range upsample factor via zero-padding (default: 1)\n"
        << "  -w, --window <type>    Window for range compression: hamming, none (default: hamming)\n"
        << "  -b, --block-size <N>   Pulses per block for reduced GPU memory (default: all)\n"
        << "  --warmup               Warmup GPU kernels and FFT plans before timed run\n"
        << "  --precision <type>     Compute precision: double, float, fltflt, mixed (default: mixed)\n"
        << "  -h, --help             Print this help message and exit\n";
  };

  if (argc < 2) {
    print_usage();
    return 1;
  }

  if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
    print_usage();
    return 0;
  }

  const std::string input_file = argv[1];
  std::string output_file;
  int upsample_factor = 1;
  std::string window_type = "hamming";
  int block_size_arg = 0;  // 0 = all pulses in one block
  bool do_warmup = false;
  std::string precision_type = "mixed";

  for (int i = 2; i < argc; i++) {
    if ((std::strcmp(argv[i], "-u") == 0 || std::strcmp(argv[i], "--upsample") == 0) && i + 1 < argc) {
      upsample_factor = std::atoi(argv[++i]);
    } else if ((std::strcmp(argv[i], "-w") == 0 || std::strcmp(argv[i], "--window") == 0) && i + 1 < argc) {
      window_type = argv[++i];
    } else if ((std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
      output_file = argv[++i];
    } else if ((std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--block-size") == 0) && i + 1 < argc) {
      block_size_arg = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      do_warmup = true;
    } else if (std::strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
      precision_type = argv[++i];
    } else if (output_file.empty()) {
      output_file = argv[i];
    }
  }

  if (output_file.empty()) {
    auto dot = input_file.rfind('.');
    output_file = (dot != std::string::npos ? input_file.substr(0, dot) : input_file) + ".raw";
  }

  // -------------------------------------------------------------------
  // 1. Read .sarbp file header
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
  // 2. Read per-pulse data (platform positions, range_to_mcp, samples)
  // -------------------------------------------------------------------
  using complex_t = cuda::std::complex<float>;

  double3 *h_positions = nullptr;
  double *h_range_to_mcp = nullptr;
  float *h_ampsf = nullptr;  // per-pulse scale (int16 mode only)
  complex_t *h_range_profiles = nullptr;
  int16_t *h_range_profiles_i16 = nullptr;

  CUDA_CHECK(cudaHostAlloc(&h_positions,
      static_cast<size_t>(num_pulses) * sizeof(double3), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&h_range_to_mcp,
      static_cast<size_t>(num_pulses) * sizeof(double), cudaHostAllocDefault));

  if (is_int16_mode) {
    CUDA_CHECK(cudaHostAlloc(&h_ampsf,
        static_cast<size_t>(num_pulses) * sizeof(float), cudaHostAllocDefault));
    // int16 pairs: num_range_bins * 2 int16 per pulse
    CUDA_CHECK(cudaHostAlloc(&h_range_profiles_i16,
        static_cast<size_t>(num_pulses) * static_cast<size_t>(num_range_bins) * 2 * sizeof(int16_t),
        cudaHostAllocDefault));
  } else {
    CUDA_CHECK(cudaHostAlloc(&h_range_profiles,
        static_cast<size_t>(num_pulses) * static_cast<size_t>(num_range_bins) * sizeof(complex_t),
        cudaHostAllocDefault));
  }

  const size_t pulse_hdr_size = hdr.pulse_header_size > 0 ? hdr.pulse_header_size : 32;
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

  // -------------------------------------------------------------------
  // 3. Compute range compression / upsampling parameters
  // -------------------------------------------------------------------
  const index_t num_samples_raw = hdr.num_samples_raw > 0
      ? static_cast<index_t>(hdr.num_samples_raw) : num_range_bins;
  const index_t fft_size = is_fx_domain
      ? (upsample_factor > 1
          ? next_pow2(static_cast<index_t>(num_samples_raw) * static_cast<index_t>(upsample_factor))
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
  // 4. Construct voxel grid
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
  // 6. Configure sar_bp parameters
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
  } else {
    std::cerr << "ERROR: unknown precision type '" << precision_type
              << "' (use double, float, fltflt, or mixed)" << std::endl;
    return 1;
  }
  params.features = SarBpFeature::PhaseLUTOptimization;
  params.center_frequency = (sgn >= 0) ? -hdr.center_frequency : hdr.center_frequency;
  params.del_r = del_r;

  // -------------------------------------------------------------------
  // 7. Block processing: range compression (if FX) + backprojection
  // -------------------------------------------------------------------
  const index_t block_size = (block_size_arg > 0)
      ? std::min(static_cast<index_t>(block_size_arg), num_pulses)
      : num_pulses;
  const index_t num_blocks = (num_pulses + block_size - 1) / block_size;

  std::cout << "Block size       : " << block_size << " pulses (" << num_blocks
            << " block" << (num_blocks > 1 ? "s" : "") << ")" << std::endl;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaExecutor exec{stream};

  // Pre-allocate GPU block buffers (sized for largest block)
  auto blk_profiles = make_tensor<complex_t>({block_size, output_range_bins}, matx::MATX_DEVICE_MEMORY);
  auto blk_positions = make_tensor<double3>({block_size}, matx::MATX_DEVICE_MEMORY);
  auto blk_rtm = make_tensor<double>({block_size}, matx::MATX_DEVICE_MEMORY);
  auto blk_compressed = is_fx_domain
      ? make_tensor<complex_t>({block_size, fft_size}, matx::MATX_DEVICE_MEMORY)
      : make_tensor<complex_t>({1, 1});
  auto blk_fx = is_fx_domain
      ? make_tensor<complex_t>({block_size, num_samples_raw}, matx::MATX_DEVICE_MEMORY)
      : make_tensor<complex_t>({1, 1});
  // int16 interleaved I/Q samples and per-pulse scale (int16 mode only)
  auto blk_fx_i16 = (is_int16_mode && is_fx_domain)
      ? make_tensor<int16_t>({block_size, num_samples_raw * 2}, matx::MATX_DEVICE_MEMORY)
      : make_tensor<int16_t>({1, 1});
  auto blk_ampsf = is_int16_mode
      ? make_tensor<float>({block_size}, matx::MATX_DEVICE_MEMORY)
      : make_tensor<float>({1});
  const bool apply_window = is_fx_domain && window_type != "none";

  // Image tensor --zeroed before first block
  auto image = make_tensor<complex_t>({image_height, image_width}, matx::MATX_DEVICE_MEMORY);
  (image = matx::zeros<complex_t>({image_height, image_width})).run(exec);

  const index_t ifft_shift = num_samples_raw / 2;

  // Warmup: run kernels with correct tensor sizes to initialize FFT plans,
  // load kernels, etc. so that the timed run reflects steady-state performance.
  if (do_warmup) {
    std::cout << "Warming up kernels..." << std::flush;

    auto warmup_block = [&](index_t npulses) {
      auto cur_profiles  = matx::slice(blk_profiles,  {0, 0}, {npulses, output_range_bins});
      auto cur_positions = matx::slice(blk_positions,  {0},    {npulses});
      auto cur_rtm       = matx::slice(blk_rtm,       {0},    {npulses});

      (cur_profiles = matx::zeros<complex_t>({npulses, output_range_bins})).run(exec);
      (cur_positions = double3{0.0, 0.0, 0.0}).run(exec);
      (cur_rtm = matx::zeros<double>({npulses})).run(exec);

      if (is_fx_domain) {
        auto cur_fx = matx::slice(blk_fx, {0, 0}, {npulses, num_samples_raw});
        if (is_int16_mode) {
          // Warmup int16 -> complex<float> conversion kernel
          auto cur_fx_i16 = matx::slice(blk_fx_i16, {0, 0}, {npulses, num_samples_raw * 2});
          auto cur_ampsf = matx::slice(blk_ampsf, {0}, {npulses});
          (cur_fx_i16 = 0).run(exec);
          (cur_ampsf = 1.0f).run(exec);
          auto i_vals = matx::slice(cur_fx_i16, {0, 0},
                                    {npulses, num_samples_raw * 2}, {1, 2});
          auto q_vals = matx::slice(cur_fx_i16, {0, 1},
                                    {npulses, num_samples_raw * 2}, {1, 2});
          auto ampsf_b = matx::clone<2>(cur_ampsf, {matxKeepDim, num_samples_raw});
          (cur_fx = matx::as_complex_float(
              matx::as_float(i_vals) * ampsf_b,
              matx::as_float(q_vals) * ampsf_b)).run(exec);
        }
        if (apply_window) {
          (cur_fx = cur_fx * matx::hamming<1>({npulses, num_samples_raw})).run(exec);
        }
        auto cur_compressed = matx::slice(blk_compressed, {0, 0}, {npulses, fft_size});
        if (sgn == -1) {
          (cur_compressed = matx::ifft(cur_profiles)).run(exec);
        } else {
          (cur_compressed = matx::fft(cur_profiles)).run(exec);
        }
        (cur_profiles = matx::fftshift1D(cur_compressed)).run(exec);
      }

      (image = matx::experimental::sar_bp(
          matx::zeros<complex_t>({image_height, image_width}),
          cur_profiles, cur_positions, voxel_locations, cur_rtm, params))
          .run(exec);
    };

    // Warmup with primary block size
    warmup_block(block_size);

    // Warmup with final block size if it differs (different FFT plan)
    const index_t last_block_size = num_pulses - (num_blocks - 1) * block_size;
    if (num_blocks > 1 && last_block_size != block_size) {
      warmup_block(last_block_size);
    }

    // Re-zero image after warmup
    (image = matx::zeros<complex_t>({image_height, image_width})).run(exec);
    exec.sync();
    std::cout << " done" << std::endl;
  }

  // Pre-allocate pinned host buffer for image output
  const size_t num_pixels =
      static_cast<size_t>(image_height) * static_cast<size_t>(image_width);
  complex_t *h_image = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_image, num_pixels * sizeof(complex_t), cudaHostAllocDefault));

  std::cout << "Running backprojection (" << output_range_bins << " range bins, del_r="
            << del_r << " m)..." << std::endl;

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  cudaProfilerStart();
  CUDA_CHECK(cudaEventRecord(ev_start, stream));

  for (index_t blk = 0; blk < num_blocks; blk++) {
    const index_t p0 = blk * block_size;
    const index_t p1 = std::min(p0 + block_size, num_pulses);
    const index_t npulses = p1 - p0;

    // Views into pre-allocated buffers (handle last block being smaller)
    auto cur_profiles  = matx::slice(blk_profiles,  {0, 0}, {npulses, output_range_bins});
    auto cur_positions = matx::slice(blk_positions,  {0},    {npulses});
    auto cur_rtm       = matx::slice(blk_rtm,       {0},    {npulses});

    // Upload platform positions and range_to_mcp for this block
    CUDA_CHECK(cudaMemcpyAsync(cur_positions.Data(),
                    h_positions + p0,
                    static_cast<size_t>(npulses) * sizeof(double3),
                    cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(cur_rtm.Data(),
                    h_range_to_mcp + p0,
                    static_cast<size_t>(npulses) * sizeof(double),
                    cudaMemcpyHostToDevice, stream));

    if (is_fx_domain) {
      auto cur_fx = matx::slice(blk_fx, {0, 0}, {npulses, num_samples_raw});

      if (is_int16_mode) {
        // Upload int16 I/Q pairs and per-pulse scale
        auto cur_fx_i16 = matx::slice(blk_fx_i16, {0, 0}, {npulses, num_samples_raw * 2});
        auto cur_ampsf = matx::slice(blk_ampsf, {0}, {npulses});
        CUDA_CHECK(cudaMemcpyAsync(cur_fx_i16.Data(),
                        h_range_profiles_i16 + p0 * num_samples_raw * 2,
                        static_cast<size_t>(npulses) * static_cast<size_t>(num_samples_raw) * 2 * sizeof(int16_t),
                        cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cur_ampsf.Data(),
                        h_ampsf + p0,
                        static_cast<size_t>(npulses) * sizeof(float),
                        cudaMemcpyHostToDevice, stream));

        // Strided views: even indices = I, odd indices = Q
        auto i_vals = matx::slice(cur_fx_i16, {0, 0},
                                  {npulses, num_samples_raw * 2}, {1, 2});
        auto q_vals = matx::slice(cur_fx_i16, {0, 1},
                                  {npulses, num_samples_raw * 2}, {1, 2});
        // Broadcast ampsf across range samples
        auto ampsf_b = matx::clone<2>(cur_ampsf, {matxKeepDim, num_samples_raw});

        // Convert int16 -> float, scale by AmpSF, combine to complex<float>
        if (apply_window) {
          auto win = matx::hamming<1>({npulses, num_samples_raw});
          (cur_fx = matx::as_complex_float(
              matx::as_float(i_vals) * ampsf_b * win,
              matx::as_float(q_vals) * ampsf_b * win)).run(exec);
        } else {
          (cur_fx = matx::as_complex_float(
              matx::as_float(i_vals) * ampsf_b,
              matx::as_float(q_vals) * ampsf_b)).run(exec);
        }
      } else {
        // Upload complex64 samples directly
        CUDA_CHECK(cudaMemcpyAsync(cur_fx.Data(),
                        h_range_profiles + p0 * num_samples_raw,
                        static_cast<size_t>(npulses) * static_cast<size_t>(num_samples_raw) * sizeof(complex_t),
                        cudaMemcpyHostToDevice, stream));

        if (apply_window) {
          (cur_fx = cur_fx * matx::hamming<1>({npulses, num_samples_raw})).run(exec);
          CUDA_CHECK(cudaGetLastError());
        }
      }

      // Zero only the padding region (middle of each row after ifftshift)
      const index_t pad_size = fft_size - num_samples_raw;
      if (pad_size > 0) {
        CUDA_CHECK(cudaMemset2DAsync(
            cur_profiles.Data() + (num_samples_raw - ifft_shift),
            static_cast<size_t>(fft_size) * sizeof(complex_t),
            0,
            static_cast<size_t>(pad_size) * sizeof(complex_t),
            static_cast<size_t>(npulses),
            stream));
      }

      // Second half of spectrum (indices shift..N-1) -> start of padded row
      CUDA_CHECK(cudaMemcpy2DAsync(
          cur_profiles.Data(),
          static_cast<size_t>(fft_size) * sizeof(complex_t),
          cur_fx.Data() + ifft_shift,
          static_cast<size_t>(num_samples_raw) * sizeof(complex_t),
          static_cast<size_t>(num_samples_raw - ifft_shift) * sizeof(complex_t),
          static_cast<size_t>(npulses),
          cudaMemcpyDeviceToDevice, stream));

      // First half of spectrum (indices 0..shift-1) -> end of padded row
      CUDA_CHECK(cudaMemcpy2DAsync(
          cur_profiles.Data() + (fft_size - ifft_shift),
          static_cast<size_t>(fft_size) * sizeof(complex_t),
          cur_fx.Data(),
          static_cast<size_t>(num_samples_raw) * sizeof(complex_t),
          static_cast<size_t>(ifft_shift) * sizeof(complex_t),
          static_cast<size_t>(npulses),
          cudaMemcpyDeviceToDevice, stream));

      // IFFT (SGN=-1) or FFT (SGN=+1) for range compression
      auto cur_compressed = matx::slice(blk_compressed, {0, 0}, {npulses, fft_size});
      if (sgn == -1) {
        (cur_compressed = matx::ifft(cur_profiles)).run(exec);
      } else {
        (cur_compressed = matx::fft(cur_profiles)).run(exec);
      }

      // fftshift to centre the zero-range bin (write result back to cur_profiles)
      (cur_profiles = matx::fftshift1D(cur_compressed)).run(exec);
    } else {
      // Pre-compressed: simple copy
      CUDA_CHECK(cudaMemcpyAsync(cur_profiles.Data(),
                      h_range_profiles + p0 * num_range_bins,
                      static_cast<size_t>(npulses) * static_cast<size_t>(num_range_bins) * sizeof(complex_t),
                      cudaMemcpyHostToDevice, stream));
    }

    // Backprojection - accumulates this block's pulses into image
    (image = matx::experimental::sar_bp(image, cur_profiles,
                                         cur_positions, voxel_locations,
                                         cur_rtm, params))
        .run(exec);

    if (num_blocks > 1) {
      std::cout << "\r  Block " << (blk + 1) << " / " << num_blocks << std::flush;
    }
  }

  // Copy result to pinned host buffer (included in timed region)
  CUDA_CHECK(cudaMemcpyAsync(h_image, image.Data(), num_pixels * sizeof(complex_t),
             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaEventRecord(ev_stop, stream));
  exec.sync();
  cudaProfilerStop();

  float elapsed_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
  if (num_blocks > 1) std::cout << std::endl;
  std::cout << "Backprojection complete in " << elapsed_ms / 1000.0f << " s"
            << std::endl;
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  // -------------------------------------------------------------------
  // 8. Write raw binary file
  // -------------------------------------------------------------------

  std::ofstream out(output_file, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "ERROR: cannot open " << output_file << " for writing"
              << std::endl;
    return 1;
  }
  out.write(reinterpret_cast<const char *>(h_image),
            static_cast<std::streamsize>(num_pixels * sizeof(complex_t)));
  out.close();

  std::cout << "Wrote " << image_height << " x " << image_width
            << " complex<float> image to " << output_file << std::endl;

  CUDA_CHECK(cudaFreeHost(h_positions));
  CUDA_CHECK(cudaFreeHost(h_range_to_mcp));
  if (h_range_profiles) CUDA_CHECK(cudaFreeHost(h_range_profiles));
  if (h_range_profiles_i16) CUDA_CHECK(cudaFreeHost(h_range_profiles_i16));
  if (h_ampsf) CUDA_CHECK(cudaFreeHost(h_ampsf));
  CUDA_CHECK(cudaFreeHost(h_image));
  matx::ClearCachesAndAllocations();
  CUDA_CHECK(cudaStreamDestroy(stream));
  MATX_EXIT_HANDLER();

  return 0;
}
