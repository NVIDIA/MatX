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

using namespace matx;

// SAR backprojection benchmarks for different precision modes

const std::vector<ssize_t> PROBLEM_SIZES = {2000};

/* Float precision benchmark */
void sarbp_float(nvbench::state &state)
{
  // Get current parameters from single "Problem Size" axis
  const index_t problem_size = static_cast<index_t>(state.get_int64("Problem Size"));
  const index_t num_range_bins = problem_size;
  const index_t num_pulses = problem_size;
  const index_t image_width = problem_size;
  const index_t image_height = problem_size;

  using compute_t = float;
  using complex_t = cuda::std::complex<compute_t>;
  using apc_t = float3;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  const compute_t min_x = -10.0f;
  const compute_t max_x = 10.0f;
  const compute_t min_y = -10.0f;
  const compute_t max_y = 10.0f;
  auto pix_coords_x = matx::linspace<compute_t>(min_x, max_x, image_width);
  auto pix_coords_y = matx::linspace<compute_t>(min_y, max_y, image_height);
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
  auto voxel_locations = matx::zipvec(
    pix_coords_xclone, pix_coords_yclone, matx::zeros<compute_t>({image_height, image_width}));

  auto range_profiles = matx::make_tensor<complex_t>({num_pulses, num_range_bins});
  auto range_to_mcp = matx::make_tensor<compute_t>({num_pulses});
  auto platform_positions = matx::make_tensor<apc_t>({num_pulses});
  auto image = matx::make_tensor<complex_t>({image_height, image_width});

  const compute_t plat_dx = static_cast<compute_t>((max_x - min_x) / static_cast<compute_t>(num_pulses));
  const compute_t plat_y = -1000.0f;
  const compute_t plat_z = 1000.0f;
  for (index_t i = 0; i < num_pulses; i++) {
    const compute_t plat_x = min_x + static_cast<compute_t>(i) * plat_dx;
    platform_positions(i) = apc_t{plat_x, plat_y, plat_z};
    range_to_mcp(i) = ::sqrtf(plat_x*plat_x + plat_y*plat_y + plat_z*plat_z);
  }

  // Initialize range_profiles with random data to simulate realistic usage
  cudaExecutor exec{};
  (range_profiles = random<complex_t>(range_profiles.Shape(), UNIFORM)).run(exec);
  exec.sync();

  SarBpParams params;
  params.compute_type = SarBpComputeType::Float;
  params.features = SarBpFeature::PhaseLUTOptimization;
  params.center_frequency = 10.0e9;
  params.del_r = static_cast<compute_t>((max_x - min_x) / static_cast<compute_t>(num_range_bins));

  // Prefetch data to device
  range_profiles.PrefetchDevice(0);
  range_to_mcp.PrefetchDevice(0);
  platform_positions.PrefetchDevice(0);
  image.PrefetchDevice(0);
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions,
                                         voxel_locations, range_to_mcp, params))
        .run(cudaExecutor(launch.get_stream()));
  });

  matx::ClearCachesAndAllocations();
}

NVBENCH_BENCH(sarbp_float)
    .add_int64_axis("Problem Size", PROBLEM_SIZES);

/* Double precision benchmark */
void sarbp_double(nvbench::state &state)
{
  // Get current parameters from single "Problem Size" axis
  const index_t problem_size = static_cast<index_t>(state.get_int64("Problem Size"));
  const index_t num_range_bins = problem_size;
  const index_t num_pulses = problem_size;
  const index_t image_width = problem_size;
  const index_t image_height = problem_size;

  using compute_t = double;
  using complex_t = cuda::std::complex<compute_t>;
  using apc_t = double3;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  const compute_t min_x = -10.0;
  const compute_t max_x = 10.0;
  const compute_t min_y = -10.0;
  const compute_t max_y = 10.0;
  auto pix_coords_x = matx::linspace<compute_t>(min_x, max_x, image_width);
  auto pix_coords_y = matx::linspace<compute_t>(min_y, max_y, image_height);
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
  auto voxel_locations = matx::zipvec(
    pix_coords_xclone, pix_coords_yclone, matx::zeros<compute_t>({image_height, image_width}));

  auto range_profiles = matx::make_tensor<complex_t>({num_pulses, num_range_bins});
  auto range_to_mcp = matx::make_tensor<compute_t>({num_pulses});
  auto platform_positions = matx::make_tensor<apc_t>({num_pulses});
  auto image = matx::make_tensor<complex_t>({image_height, image_width});

  const compute_t plat_dx = static_cast<compute_t>((max_x - min_x) / static_cast<compute_t>(num_pulses));
  const compute_t plat_y = -1000.0;
  const compute_t plat_z = 1000.0;
  for (index_t i = 0; i < num_pulses; i++) {
    const compute_t plat_x = min_x + static_cast<compute_t>(i) * plat_dx;
    platform_positions(i) = apc_t{plat_x, plat_y, plat_z};
    range_to_mcp(i) = ::sqrt(plat_x*plat_x + plat_y*plat_y + plat_z*plat_z);
  }

  // Initialize range_profiles with random data to simulate realistic usage
  cudaExecutor exec{};
  (range_profiles = random<complex_t>(range_profiles.Shape(), UNIFORM)).run(exec);
  exec.sync();

  SarBpParams params;
  params.compute_type = SarBpComputeType::Double;
  params.features = SarBpFeature::PhaseLUTOptimization;
  params.center_frequency = 10.0e9;
  params.del_r = static_cast<compute_t>((max_x - min_x) / static_cast<compute_t>(num_range_bins));

  // Prefetch data to device
  range_profiles.PrefetchDevice(0);
  range_to_mcp.PrefetchDevice(0);
  platform_positions.PrefetchDevice(0);
  image.PrefetchDevice(0);
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions,
                                         voxel_locations, range_to_mcp, params))
        .run(cudaExecutor(launch.get_stream()));
  });

  matx::ClearCachesAndAllocations();
}

NVBENCH_BENCH(sarbp_double)
    .add_int64_axis("Problem Size", PROBLEM_SIZES);

/* Mixed precision benchmark - uses fp32 for image/range profiles, fp64 for platform positions */
void sarbp_mixed(nvbench::state &state)
{
  // Get current parameters from single "Problem Size" axis
  const index_t problem_size = static_cast<index_t>(state.get_int64("Problem Size"));
  const index_t num_range_bins = problem_size;
  const index_t num_pulses = problem_size;
  const index_t image_width = problem_size;
  const index_t image_height = problem_size;

  using complex_t = cuda::std::complex<float>;
  using loose_compute_t = float;
  using strict_compute_t = double;
  using apc_t = double3;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  const loose_compute_t min_x = -10.0f;
  const loose_compute_t max_x = 10.0f;
  const loose_compute_t min_y = -10.0f;
  const loose_compute_t max_y = 10.0f;
  auto pix_coords_x = matx::linspace<loose_compute_t>(min_x, max_x, image_width);
  auto pix_coords_y = matx::linspace<loose_compute_t>(min_y, max_y, image_height);
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
  auto voxel_locations = matx::zipvec(
    pix_coords_xclone, pix_coords_yclone, matx::zeros<loose_compute_t>({image_height, image_width}));

  auto range_profiles = matx::make_tensor<complex_t>({num_pulses, num_range_bins});
  auto range_to_mcp = matx::make_tensor<strict_compute_t>({num_pulses});
  auto platform_positions = matx::make_tensor<apc_t>({num_pulses});
  auto image = matx::make_tensor<complex_t>({image_height, image_width});

  const strict_compute_t plat_dx = static_cast<strict_compute_t>((max_x - min_x) / static_cast<strict_compute_t>(num_pulses));
  const strict_compute_t plat_y = -1000.0;
  const strict_compute_t plat_z = 1000.0;
  for (index_t i = 0; i < num_pulses; i++) {
    const strict_compute_t plat_x = min_x + static_cast<strict_compute_t>(i) * plat_dx;
    platform_positions(i) = apc_t{plat_x, plat_y, plat_z};
    range_to_mcp(i) = ::sqrt(plat_x*plat_x + plat_y*plat_y + plat_z*plat_z);
  }

  // Initialize range_profiles with random data to simulate realistic usage
  cudaExecutor exec{};
  (range_profiles = random<complex_t>(range_profiles.Shape(), UNIFORM)).run(exec);
  exec.sync();

  SarBpParams params;
  params.compute_type = SarBpComputeType::Mixed;
  params.features = SarBpFeature::PhaseLUTOptimization;
  params.center_frequency = 10.0e9;
  params.del_r = static_cast<strict_compute_t>((max_x - min_x) / static_cast<strict_compute_t>(num_range_bins));

  // Prefetch data to device
  range_profiles.PrefetchDevice(0);
  range_to_mcp.PrefetchDevice(0);
  platform_positions.PrefetchDevice(0);
  image.PrefetchDevice(0);
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions,
                                         voxel_locations, range_to_mcp, params))
        .run(cudaExecutor(launch.get_stream()));
  });

  matx::ClearCachesAndAllocations();
}

NVBENCH_BENCH(sarbp_mixed)
    .add_int64_axis("Problem Size", PROBLEM_SIZES);

/* FloatFloat precision benchmark - uses float-float arithmetic for better precision than fp32 */
void sarbp_fltflt(nvbench::state &state)
{
  // Get current parameters from single "Problem Size" axis
  const index_t problem_size = static_cast<index_t>(state.get_int64("Problem Size"));
  const index_t num_range_bins = problem_size;
  const index_t num_pulses = problem_size;
  const index_t image_width = problem_size;
  const index_t image_height = problem_size;

  using complex_t = cuda::std::complex<float>;
  using loose_compute_t = float;
  using strict_compute_t = double;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  const loose_compute_t min_x = -10.0f;
  const loose_compute_t max_x = 10.0f;
  const loose_compute_t min_y = -10.0f;
  const loose_compute_t max_y = 10.0f;
  auto pix_coords_x = matx::linspace<loose_compute_t>(min_x, max_x, image_width);
  auto pix_coords_y = matx::linspace<loose_compute_t>(min_y, max_y, image_height);
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
  auto voxel_locations = matx::zipvec(
    pix_coords_xclone, pix_coords_yclone, matx::zeros<loose_compute_t>({image_height, image_width}));

  auto range_profiles = matx::make_tensor<complex_t>({num_pulses, num_range_bins});
  auto range_to_mcp = matx::make_tensor<fltflt>({num_pulses});
  auto platform_positions = matx::make_tensor<fltflt>({num_pulses,3});
  auto image = matx::make_tensor<complex_t>({image_height, image_width});

  const strict_compute_t plat_dx = static_cast<strict_compute_t>((max_x - min_x) / static_cast<strict_compute_t>(num_pulses));
  const strict_compute_t plat_y = -1000.0;
  const strict_compute_t plat_z = 1000.0;
  for (index_t i = 0; i < num_pulses; i++) {
    const strict_compute_t plat_x = min_x + static_cast<strict_compute_t>(i) * plat_dx;
    platform_positions(i, 0) = static_cast<fltflt>(plat_x);
    platform_positions(i, 1) = static_cast<fltflt>(plat_y);
    platform_positions(i, 2) = static_cast<fltflt>(plat_z);
    range_to_mcp(i) = static_cast<fltflt>(::sqrt(plat_x*plat_x + plat_y*plat_y + plat_z*plat_z));
  }

  // Initialize range_profiles with random data to simulate realistic usage
  cudaExecutor exec{};
  (range_profiles = random<complex_t>(range_profiles.Shape(), UNIFORM)).run(exec);
  exec.sync();

  SarBpParams params;
  params.compute_type = SarBpComputeType::FloatFloat;
  params.features = SarBpFeature::PhaseLUTOptimization;
  params.center_frequency = 10.0e9;
  params.del_r = static_cast<strict_compute_t>((max_x - min_x) / static_cast<strict_compute_t>(num_range_bins));

  // Prefetch data to device
  range_profiles.PrefetchDevice(0);
  range_to_mcp.PrefetchDevice(0);
  platform_positions.PrefetchDevice(0);
  image.PrefetchDevice(0);
  exec.sync();

  state.exec([&](nvbench::launch &launch) {
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions,
                                         voxel_locations, range_to_mcp, params))
        .run(cudaExecutor(launch.get_stream()));
  });

  matx::ClearCachesAndAllocations();
}

NVBENCH_BENCH(sarbp_fltflt)
    .add_int64_axis("Problem Size", PROBLEM_SIZES);
