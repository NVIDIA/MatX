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

#include "assert.h"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"
#include <cuda/std/complex>

using namespace matx;

template <typename T>
class SarBpTest : public ::testing::Test {
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;
protected:
  void SetUp() override
  {
    CheckTestTypeSupport<GTestType>();
    pb = std::make_unique<detail::MatXPybind>();

    if constexpr (is_complex_half_v<GTestType> || is_matx_half_v<GTestType>) {
      thresh = 1.0e-1;
    } else if constexpr (std::is_same_v<GTestType, double>) {
      thresh = 1.0e-12;
    } else {
      // Revisit this tolerance. We should likely use a relative tolerance
      // rather than absolute for larger values.
      thresh = 1.0e-3;
    }
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  double thresh;
};

template <typename TensorType>
class SarBpTestNonComplexNonHalfFloatTypes
    : public SarBpTest<TensorType> {
};

template <typename TensorType>
class SarBpTestDoubleType
    : public SarBpTest<TensorType> {
};

TYPED_TEST_SUITE(SarBpTestNonComplexNonHalfFloatTypes, MatXFloatNonComplexNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(SarBpTestDoubleType, MatXDoubleOnlyTypeCUDAExec);

// Test with all values (input data, pixels, locations, etc.) having the same floating
// point precision.
TYPED_TEST(SarBpTestNonComplexNonHalfFloatTypes, NonMixedTypes)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using complex_t = cuda::std::complex<TestType>;
  using apc_t = std::conditional_t<std::is_same_v<TestType, double>, double3, float3>;

  const index_t num_range_bins = 128;
  const index_t num_pulses = 128;
  const index_t image_width = 128;
  const index_t image_height = 128;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  const TestType min_x = -10.0;
  const TestType max_x = 10.0;
  const TestType min_y = -10.0;
  const TestType max_y = 10.0;
  auto pix_coords_x = matx::linspace<TestType>(min_x, max_x, image_width);
  auto pix_coords_y = matx::linspace<TestType>(min_y, max_y, image_height);
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
  auto voxel_locations = matx::zipvec(
    pix_coords_xclone, pix_coords_yclone, matx::zeros<TestType>({image_height, image_width}));
  
  auto range_profiles = matx::ones<complex_t>({num_pulses, num_range_bins});
  auto range_to_mcp = matx::make_tensor<TestType>({num_pulses});  
  auto platform_positions = matx::make_tensor<apc_t>({num_pulses});
  auto image = matx::make_tensor<complex_t>({image_height, image_width});
  const TestType plat_dx = (max_x - min_x) / num_pulses;
  const TestType plat_y = -1000.0;
  const TestType plat_z = 1000.0;
  for (index_t i = 0; i < num_pulses; i++) {
    const TestType plat_x = min_x + static_cast<TestType>(i) * plat_dx;
    platform_positions(i) = apc_t{plat_x, plat_y, plat_z};
    range_to_mcp(i) = ::sqrt(plat_x*plat_x + plat_y*plat_y + plat_z*plat_z);
  }

  SarBpParams params;
  if constexpr (std::is_same_v<TestType, double>) {
    params.compute_type = SarBpComputeType::Double;
  } else {
    params.compute_type = SarBpComputeType::Float;
  }
  params.features = SarBpFeature::PhaseLUTOptimization;
  params.center_frequency = 10.0e9;
  params.del_r = (max_x - min_x) / num_range_bins;

  (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)).run(this->exec);
  this->exec.sync();

  MATX_EXIT_HANDLER();
}

// Test Mixed and FloatFloat precisions. These precisions are used when the user wants
// better than fp32 precision at lower cost than fp64 (especially for GPUs with reduced FP64 throughput).
TYPED_TEST(SarBpTestDoubleType, MixedPrecision)
{
  MATX_ENTER_HANDLER();

  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  // Used for image pixels and input data, which are assumed to be fp32 for
  // mixed precision variants.
  using complex_t = cuda::std::complex<float>;
  using loose_compute_t = float;
  using strict_compute_t = double;
  using apc_t = double3;  

  const index_t num_range_bins = 128;
  const index_t num_pulses = 128;
  const index_t image_width = 128;
  const index_t image_height = 128;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  const loose_compute_t min_x = -10.0;
  const loose_compute_t max_x = 10.0;
  const loose_compute_t min_y = -10.0;
  const loose_compute_t max_y = 10.0;
  auto pix_coords_x = matx::linspace<loose_compute_t>(min_x, max_x, image_width);
  auto pix_coords_y = matx::linspace<loose_compute_t>(min_y, max_y, image_height);
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
  auto voxel_locations = matx::zipvec(
    pix_coords_xclone, pix_coords_yclone, matx::zeros<loose_compute_t>({image_height, image_width}));
  
  auto range_profiles = matx::ones<complex_t>({num_pulses, num_range_bins});
  auto range_to_mcp = matx::make_tensor<strict_compute_t>({num_pulses});  
  auto platform_positions = matx::make_tensor<apc_t>({num_pulses});
  auto image = matx::make_tensor<complex_t>({image_height, image_width});
  const strict_compute_t plat_dx = (max_x - min_x) / num_pulses;
  const strict_compute_t plat_y = -1000.0;
  const strict_compute_t plat_z = 1000.0;
  for (index_t i = 0; i < num_pulses; i++) {
    const strict_compute_t plat_x = min_x + static_cast<strict_compute_t>(i) * plat_dx;
    platform_positions(i) = apc_t{plat_x, plat_y, plat_z};
    range_to_mcp(i) = ::sqrt(plat_x*plat_x + plat_y*plat_y + plat_z*plat_z);
  }

  SarBpParams params;
  params.center_frequency = 10.0e9;
  params.del_r = (max_x - min_x) / num_range_bins;

  this->exec.sync();

  // Run with Mixed precision and no additional optimizations
  {
    params.compute_type = SarBpComputeType::Mixed;
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)).run(this->exec);
    this->exec.sync();
  }

  // Enable PhaseLUTOptimization
  {
    params.features = SarBpFeature::PhaseLUTOptimization;
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)).run(this->exec);
    this->exec.sync();
  }

  // Enable FloatFloat optimizations. This is only an optimization on GPUs with reduced FP64 throughput.
  // On 100 class GPUs with full FP64 throughput, Mixed precision is significantly faster than FloatFloat.
  {
    params.compute_type = SarBpComputeType::FloatFloat;
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)).run(this->exec);
    this->exec.sync();
  }

  MATX_EXIT_HANDLER();
}

// Test with a simplified point target. In this case, there is a single reflector at a known position.
// The range profile data will be populated with the negation of the phase model applied in the backprojector
// for the two range bins that will be interpolated by the backprojector. Other range bins will be populated with 0.
// The expected result is an image with the pixel center at the target location having a value of approximately
// the number of pulses.
// better than fp32 precision at lower cost than fp64 (especially for GPUs with reduced FP64 throughput).
TYPED_TEST(SarBpTestDoubleType, PointTarget)
{
  MATX_ENTER_HANDLER();

  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  constexpr matx::index_t image_width = 128;
  constexpr matx::index_t image_height = 128;
  const matx::index_t num_pulses = 128;
  const matx::index_t num_range_bins = 128;

  const double pix_dx = 0.5;
  const double pix_x_min = -1.0 * image_width / 2.0 + pix_dx;
  const double pix_dy = 0.5;
  const double pix_y_min = -1.0 * image_height / 2.0 + pix_dy;
  const double plat_x_min = 1.0 * image_width / 2.0 - 10.0;
  const double plat_dx = (plat_x_min - pix_x_min) / num_pulses;
  const double plat_y = -7000.0;
  const double plat_z = 7000.0;
  assert(image_width % 4 == 0);
  assert(image_height % 4 == 0);
  const double target_x = pix_x_min + pix_dx * (image_width / 4.0);
  const double target_y = pix_y_min + pix_dy * (image_height / 4.0);
  // Target range relative to mocomp point (0,0,0)
  const double target_R = ::sqrt(target_x * target_x + target_y * target_y);

  matx::SarBpParams bp_params;
  bp_params.del_r = ::sqrt((image_width * pix_dx) * (image_width * pix_dx) + (image_height * pix_dy) * (image_height * pix_dy)) / num_range_bins;
  bp_params.center_frequency = 10.0e9;
  const double bin_offset = 0.5 * (num_range_bins - 1);

  std::vector<cuda::std::complex<double>> h_range_profiles;
  std::vector<double3> h_antenna_phase_centers;
  std::vector<double> h_range_to_mcp;
  h_range_profiles.resize(num_pulses * num_range_bins);
  h_antenna_phase_centers.resize(num_pulses);
  h_range_to_mcp.resize(num_pulses);

  for (matx::index_t i = 0; i < num_pulses; i++) {
    h_antenna_phase_centers[i] = double3{
      plat_x_min + 
      plat_dx * static_cast<double>(i),
      plat_y,
      plat_z
    };
    h_range_to_mcp[i] = ::sqrt(
      h_antenna_phase_centers[i].x * h_antenna_phase_centers[i].x +
      h_antenna_phase_centers[i].y * h_antenna_phase_centers[i].y +
      h_antenna_phase_centers[i].z * h_antenna_phase_centers[i].z);
    const double R_to_target = ::sqrt(
      (h_antenna_phase_centers[i].x - target_x) * (h_antenna_phase_centers[i].x - target_x) +
      (h_antenna_phase_centers[i].y - target_y) * (h_antenna_phase_centers[i].y - target_y) +
      (h_antenna_phase_centers[i].z) * (h_antenna_phase_centers[i].z));
    const double ideal_dR = R_to_target - h_range_to_mcp[i];
    const double ideal_bin = ideal_dR / bp_params.del_r + bin_offset;
    const double ideal_phase = -4.0 * M_PI * ideal_dR * (bp_params.center_frequency / SPEED_OF_LIGHT);
    double sinx, cosx;
    ::sincos(ideal_phase, &sinx, &cosx);
    const cuda::std::complex<double> sample = cuda::std::complex<double>{cosx, sinx};
    const int ideal_bin_floor = static_cast<int>(floor(ideal_bin));
    if (ideal_bin_floor >= 0 && ideal_bin_floor < num_range_bins) {
      h_range_profiles[i * num_range_bins + ideal_bin_floor] = sample;
    }
    if (ideal_bin_floor + 1 >= 0 && ideal_bin_floor + 1 < num_range_bins) {
      h_range_profiles[i * num_range_bins + ideal_bin_floor + 1] = sample;
    }
  }

  // re_thresh and im_thresh are the thresholds for the real and imaginary parts of the pixel
  // centered at the target location, respectively.

  auto validate = [image_height, image_width, num_pulses](auto &image, double re_thresh, [[maybe_unused]] double im_thresh) {
    for (matx::index_t i = 0; i < image_height; i++) {
      for (matx::index_t j = 0; j < image_width; j++) {
        if (i == image_height / 4 && j == image_width / 4) {
          const double expected = num_pulses;
          const double actual = image(i, j).real();
          ASSERT_NEAR(expected, actual, re_thresh);
          ASSERT_NEAR(0.0, image(i, j).imag(), im_thresh);
        } else if (i < image_height / 8 || i >= 3 * image_height / 8) {
          // Away from the scatterer in range, we should not have any non-zero values.
          ASSERT_EQ(abs(image(i, j).real()), 0);
          ASSERT_EQ(abs(image(i, j).imag()), 0);
        } else {
          ASSERT_LT(abs(image(i, j).real()), num_pulses);
          ASSERT_LT(abs(image(i, j).imag()), num_pulses);
        }
      }
    }
  };

  // Start with fully fp64 backprojector. The range profiles and all position values are double precision.
  {
    auto pix_coords_x = matx::linspace<double>(
      static_cast<double>(pix_x_min), static_cast<double>(pix_x_min + (image_width - 1) * pix_dx), image_width);
    auto pix_coords_y = matx::linspace<double>(
      static_cast<double>(pix_y_min), static_cast<double>(pix_y_min + (image_height - 1) * pix_dy), image_height);
    auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
    auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
    auto voxel_locations = matx::zipvec(pix_coords_xclone, pix_coords_yclone, matx::zeros<double>({image_height, image_width}));

    auto range_profiles = matx::make_tensor<cuda::std::complex<double>>({num_pulses, num_range_bins});
    auto platform_positions = matx::make_tensor<double3>({num_pulses});
    auto range_to_mcp = matx::make_tensor<double>({num_pulses});
    MATX_CUDA_CHECK(cudaMemcpy(range_profiles.Data(), h_range_profiles.data(), h_range_profiles.size() * sizeof(cuda::std::complex<double>), cudaMemcpyHostToDevice));
    MATX_CUDA_CHECK(cudaMemcpy(platform_positions.Data(), h_antenna_phase_centers.data(), h_antenna_phase_centers.size() * sizeof(double3), cudaMemcpyHostToDevice));
    MATX_CUDA_CHECK(cudaMemcpy(range_to_mcp.Data(), h_range_to_mcp.data(), h_range_to_mcp.size() * sizeof(double), cudaMemcpyHostToDevice));
    auto zero_image = matx::zeros<cuda::std::complex<double>>({image_height, image_width});
    auto image = matx::make_tensor<cuda::std::complex<double>>({image_height, image_width});

    bp_params.compute_type = matx::SarBpComputeType::Double;
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
    this->exec.sync();

    validate(image, 1.0e-10, 1.0e-10);
  }

  // The mixed precision and single precision backprojectors take single-precision values for range profiles,
  // voxel positions, and image voxels. For mixed precision, platform positions and range to the mcp are
  // still in double precision.
  auto pix_coords_x = matx::linspace<float>(
    static_cast<float>(pix_x_min), static_cast<float>(pix_x_min + (image_width - 1) * pix_dx), image_width);
  auto pix_coords_y = matx::linspace<float>(
    static_cast<float>(pix_y_min), static_cast<float>(pix_y_min + (image_height - 1) * pix_dy), image_height);
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
  auto voxel_locations = matx::zipvec(pix_coords_xclone, pix_coords_yclone, matx::zeros<float>({image_height, image_width}));

  auto range_profiles = matx::make_tensor<cuda::std::complex<float>>({num_pulses, num_range_bins});
  std::vector<cuda::std::complex<float>> h_range_profiles_f32;
  h_range_profiles_f32.resize(num_pulses * num_range_bins);
  for (matx::index_t i = 0; i < num_pulses * num_range_bins; i++) {
    h_range_profiles_f32[i] = static_cast<cuda::std::complex<float>>(h_range_profiles[i]);
  }
  MATX_CUDA_CHECK(cudaMemcpy(range_profiles.Data(), h_range_profiles_f32.data(), h_range_profiles_f32.size() * sizeof(cuda::std::complex<float>), cudaMemcpyHostToDevice));

  auto zero_image = matx::zeros<cuda::std::complex<float>>({image_height, image_width});
  auto image = matx::make_tensor<cuda::std::complex<float>>({image_height, image_width});

  // Fully fp32 version of the backprojector
  {
    std::vector<float3> h_antenna_phase_centers_f32;
    std::vector<float> h_range_to_mcp_f32;
    h_antenna_phase_centers_f32.resize(num_pulses);
    h_range_to_mcp_f32.resize(num_pulses);
    for (matx::index_t i = 0; i < num_pulses; i++) {
      h_antenna_phase_centers_f32[i] = float3{
        static_cast<float>(h_antenna_phase_centers[i].x),
        static_cast<float>(h_antenna_phase_centers[i].y),
        static_cast<float>(h_antenna_phase_centers[i].z)
      };
      h_range_to_mcp_f32[i] = static_cast<float>(h_range_to_mcp[i]);
    }
    auto platform_positions = matx::make_tensor<float3>({num_pulses});
    auto range_to_mcp = matx::make_tensor<float>({num_pulses});
    MATX_CUDA_CHECK(cudaMemcpy(
      platform_positions.Data(), h_antenna_phase_centers_f32.data(), h_antenna_phase_centers_f32.size() * sizeof(float3), cudaMemcpyHostToDevice));
    MATX_CUDA_CHECK(cudaMemcpy(range_to_mcp.Data(), h_range_to_mcp_f32.data(), h_range_to_mcp_f32.size() * sizeof(float), cudaMemcpyHostToDevice));

    bp_params.compute_type = matx::SarBpComputeType::Float;
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
    this->exec.sync();

    for (matx::index_t i = 0; i < image_height; i++) {
      for (matx::index_t j = 0; j < image_width; j++) {
        if (i == image_height / 4 && j == image_width / 4) {
          const float expected = 0.98f * num_pulses;
          const float actual = image(i, j).real();
          ASSERT_GT(actual, expected);
          ASSERT_LT(std::abs(image(i, j).imag()), 1.0f);
        } else if (i < image_height / 8 || i >= 3 * image_height / 8) {
          // Away from the scatterer in range, we should not have any non-zero values.
          ASSERT_EQ(abs(image(i, j).real()), 0.0f);
          ASSERT_EQ(abs(image(i, j).imag()), 0.0f);
        } else {
          const float expected = 0.9f * num_pulses;
          ASSERT_LT(abs(image(i, j).real()), expected);
          ASSERT_LT(abs(image(i, j).imag()), expected);
        }
      }
    }
  }

  // Mixed and FloatFloat versions of the backprojector. For these, we need double-precision platform positions
  // and range to the mcp.
  {
    bp_params.features = matx::SarBpFeature::PhaseLUTOptimization;
  
    auto platform_positions = matx::make_tensor<double3>({num_pulses});
    auto range_to_mcp = matx::make_tensor<double>({num_pulses});
    MATX_CUDA_CHECK(cudaMemcpy(
      platform_positions.Data(), h_antenna_phase_centers.data(), h_antenna_phase_centers.size() * sizeof(double3), cudaMemcpyHostToDevice));
    MATX_CUDA_CHECK(cudaMemcpy(range_to_mcp.Data(), h_range_to_mcp.data(), h_range_to_mcp.size() * sizeof(double), cudaMemcpyHostToDevice));

    bp_params.compute_type = matx::SarBpComputeType::Mixed;
    // example-begin sar-bp-1
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
    // example-end sar-bp-1
    this->exec.sync();  

    validate(image, 1.0e-10, 1.0e-2);

    bp_params.compute_type = matx::SarBpComputeType::FloatFloat;
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
    this->exec.sync();  

    validate(image, 1.0e-10, 1.0e-2);
  }

  MATX_EXIT_HANDLER();
}
