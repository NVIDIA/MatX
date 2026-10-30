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
#include "prerun_tester.h"
#include <cuda/std/complex>
#include <cuda/std/numbers>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

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
    } else if constexpr (cuda::std::is_same_v<GTestType, double>) {
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

// Verify the num_range_bins boundary at 2^24 for compute types that compute
// the per-pulse bin_offset in fp32 (Float, Mixed, FloatFloat, TaylorFast). fp32 can
// exactly represent all integers in [-2^24, 2^24] but not above, so those
// paths accept num_range_bins == 2^24 and reject anything larger.
//
// To avoid a 128 MB phase-LUT allocation in the positive case, we use Mixed
// compute with SarBpFeature::None, which routes through the no-LUT code path.
// All bulk inputs use operator generators (matx::ones / matx::zeros) so the
// only allocated memory is the tiny output image and a 1-element
// platform_positions tensor.
//
// Also verify the lower bound, which the host checks before dispatching on the
// compute type: linear interpolation reads two adjacent bins, so fewer than two
// range bins must be rejected. The kernel's single unsigned bounds check relies
// on this, since with zero bins num_range_bins - 1 would wrap to UINT32_MAX.
TYPED_TEST(SarBpTestDoubleType, RangeBinsLimitFp32Path)
{
  MATX_ENTER_HANDLER();

  using complex_t = cuda::std::complex<float>;

  constexpr index_t MAX_RANGE_BINS = static_cast<index_t>(1) << 24;
  const index_t num_pulses = 1;
  const index_t image_width = 4;
  const index_t image_height = 4;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  auto voxel_locations = matx::zipvec(
    matx::zeros<float>({image_height, image_width}),
    matx::zeros<float>({image_height, image_width}),
    matx::zeros<float>({image_height, image_width}));
  auto range_to_mcp = matx::zeros<double>({num_pulses});
  auto platform_positions = matx::make_tensor<double3>({num_pulses});
  platform_positions(0) = double3{0.0, 0.0, 1000.0};
  auto image = matx::make_tensor<complex_t>({image_height, image_width});

  SarBpParams params;
  params.compute_type = SarBpComputeType::Mixed;
  params.features = SarBpFeature::None;  // skip phase-LUT alloc
  params.center_frequency = 10.0e9;
  params.del_r = 1.0;

  auto run = [&](index_t num_range_bins) {
    auto range_profiles = matx::ones<complex_t>({num_pulses, num_range_bins});
    (image = matx::experimental::sar_bp(
      zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)).run(this->exec);
    this->exec.sync();
  };

  // num_range_bins == 2^24 is the inclusive upper bound and must not throw.
  EXPECT_NO_THROW(run(MAX_RANGE_BINS));

  // num_range_bins == 2^24 + 1 must throw matxException.
  ASSERT_THROW(run(MAX_RANGE_BINS + 1), matx::detail::matxException);

  // Two bins is the smallest valid profile.
  ASSERT_THROW(run(0), matx::detail::matxException);
  ASSERT_THROW(run(1), matx::detail::matxException);
  EXPECT_NO_THROW(run(2));

  MATX_EXIT_HANDLER();
}

// Verify that the Double compute type is exempt from the 2^24 cap. Double
// uses loose_compute_t = double throughout, so bin_offset and the bin floor
// are fp64 and the fp32 mantissa limit does not apply. Only the int32_t
// indexing cap (num_range_bins <= INT32_MAX) constrains this path.
TYPED_TEST(SarBpTestDoubleType, RangeBinsLimitDoublePath)
{
  MATX_ENTER_HANDLER();

  using complex_t = cuda::std::complex<double>;

  constexpr index_t MAX_RANGE_BINS = static_cast<index_t>(1) << 24;
  const index_t num_pulses = 1;
  const index_t image_width = 4;
  const index_t image_height = 4;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  auto voxel_locations = matx::zipvec(
    matx::zeros<double>({image_height, image_width}),
    matx::zeros<double>({image_height, image_width}),
    matx::zeros<double>({image_height, image_width}));
  auto range_to_mcp = matx::zeros<double>({num_pulses});
  auto platform_positions = matx::make_tensor<double3>({num_pulses});
  platform_positions(0) = double3{0.0, 0.0, 1000.0};
  auto image = matx::make_tensor<complex_t>({image_height, image_width});

  SarBpParams params;
  params.compute_type = SarBpComputeType::Double;
  params.features = SarBpFeature::None;  // skip phase-LUT alloc
  params.center_frequency = 10.0e9;
  params.del_r = 1.0;

  auto range_profiles = matx::ones<complex_t>({num_pulses, MAX_RANGE_BINS + 1});
  EXPECT_NO_THROW({
    (image = matx::experimental::sar_bp(
      zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)).run(this->exec);
    this->exec.sync();
  });

  MATX_EXIT_HANDLER();
}

// Verify the FFT-shift range-profile convention directly: zero differential
// range maps to floor(N/2), for both even and odd profile lengths. The even
// case distinguishes this from the former geometric-center offset (N-1)/2.
TYPED_TEST(SarBpTestDoubleType, ZeroDifferentialRangeUsesFftShiftBin)
{
  MATX_ENTER_HANDLER();

  using complex_t = cuda::std::complex<double>;
  constexpr complex_t SAMPLE{0.6, 0.8};
  constexpr index_t NUM_PULSES = 1;

  auto zero_image = matx::zeros<complex_t>({1, 1});
  auto voxel_locations = matx::zipvec(
    matx::zeros<double>({1, 1}),
    matx::zeros<double>({1, 1}),
    matx::zeros<double>({1, 1}));
  auto platform_positions = matx::make_tensor<double3>({NUM_PULSES});
  auto range_to_mcp = matx::make_tensor<double>({NUM_PULSES});
  auto image = matx::make_tensor<complex_t>({1, 1});
  platform_positions(0) = double3{0.0, 0.0, 1000.0};
  range_to_mcp(0) = 1000.0;

  SarBpParams params;
  params.compute_type = SarBpComputeType::Double;
  params.center_frequency = 9.6e9;
  params.del_r = 1.0;

  for (const index_t num_range_bins : {index_t{8}, index_t{9}}) {
    for (const SarBpFeature features :
         {SarBpFeature::None, SarBpFeature::PhaseLUTOptimization}) {
      SCOPED_TRACE(::testing::Message()
                   << "num_range_bins=" << num_range_bins
                   << ", phase_lut="
                   << (features == SarBpFeature::PhaseLUTOptimization));
      auto range_profiles = matx::make_tensor<complex_t>(
          {NUM_PULSES, num_range_bins});
      (range_profiles = matx::zeros<complex_t>(
          {NUM_PULSES, num_range_bins})).run(this->exec);
      this->exec.sync();
      range_profiles(0, num_range_bins / 2) = SAMPLE;

      params.features = features;
      (image = matx::experimental::sar_bp(
          zero_image, range_profiles, platform_positions, voxel_locations,
          range_to_mcp, params)).run(this->exec);
      this->exec.sync();

      EXPECT_NEAR(image(0, 0).real(), SAMPLE.real(), 1.0e-12);
      EXPECT_NEAR(image(0, 0).imag(), SAMPLE.imag(), 1.0e-12);
    }
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

  // Divisible by 4 for target-location checks, but not by the 16x16 CUDA block
  // size, so TaylorFast exercises partial edge CTAs.
  constexpr matx::index_t image_width = 132;
  constexpr matx::index_t image_height = 132;
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
  // Target range relative to mocomp point (0,0,0) — used for parameter setup
  [[maybe_unused]] const double target_R = ::sqrt(target_x * target_x + target_y * target_y);

  matx::SarBpParams bp_params{};
  bp_params.del_r = ::sqrt((image_width * pix_dx) * (image_width * pix_dx) + (image_height * pix_dy) * (image_height * pix_dy)) / num_range_bins;
  bp_params.center_frequency = 10.0e9;
  const double bin_offset = static_cast<double>(num_range_bins / 2);

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
    const double ideal_phase =
        -4.0 * cuda::std::numbers::pi * ideal_dR *
        (bp_params.center_frequency / matx::constants::speed_of_light);
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

  auto validate = [num_pulses](auto &image, double re_thresh, [[maybe_unused]] double im_thresh) {
    for (matx::index_t i = 0; i < image_height; i++) {
      for (matx::index_t j = 0; j < image_width; j++) {
        if (i == image_height / 4 && j == image_width / 4) {
          const double expected = num_pulses;
          const double actual = image(i, j).real();
          ASSERT_NEAR(expected, actual, re_thresh);
          ASSERT_NEAR(0.0, image(i, j).imag(), im_thresh);
        } else if (i < image_height / 8 || i >= 3 * image_height / 8) {
          // Away from the scatterer in range, we should not have any non-zero values.
          ASSERT_EQ(cuda::std::abs(image(i, j).real()), 0);
          ASSERT_EQ(cuda::std::abs(image(i, j).imag()), 0);
        } else {
          ASSERT_LT(cuda::std::abs(image(i, j).real()), num_pulses);
          ASSERT_LT(cuda::std::abs(image(i, j).imag()), num_pulses);
        }
      }
    }
  };
  // Allow a little under two fp32 epsilons per pulse at the focused target.
  // The precise accumulation error depends on the interpolated-bin phases.
  const double f32_accum_re_thresh = 2.0e-7 * static_cast<double>(num_pulses);

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

    validate(image, 1.05e-10, 1.05e-10);

    // The pixel z coordinate is 0 for this grid, so the PropSarBpPixelZIsZero
    // and PropSarBpPixelZIsFixed compile-time assumptions both hold and must
    // reproduce the focused image.
    {
      SCOPED_TRACE("Double PropSarBpPixelZIsZero");
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
        .props<matx::PropSarBpPixelZIsZero>()).run(this->exec);
      this->exec.sync();
      validate(image, 1.05e-10, 1.05e-10);
    }
    {
      SCOPED_TRACE("Double PropSarBpPixelZIsFixed");
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
        .props<matx::PropSarBpPixelZIsFixed>()).run(this->exec);
      this->exec.sync();
      validate(image, 1.05e-10, 1.05e-10);
    }
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

    auto validate_float = [&](auto &img) {
      for (matx::index_t i = 0; i < image_height; i++) {
        for (matx::index_t j = 0; j < image_width; j++) {
          if (i == image_height / 4 && j == image_width / 4) {
            const float expected = 0.98f * num_pulses;
            const float actual = img(i, j).real();
            ASSERT_GT(actual, expected);
            ASSERT_LT(cuda::std::abs(img(i, j).imag()), 3.0f);
          } else if (i < image_height / 8 || i >= 3 * image_height / 8) {
            // Away from the scatterer in range, we should not have any non-zero values.
            ASSERT_EQ(cuda::std::abs(img(i, j).real()), 0.0f);
            ASSERT_EQ(cuda::std::abs(img(i, j).imag()), 0.0f);
          } else {
            const float expected = 0.91f * num_pulses;
            ASSERT_LT(cuda::std::abs(img(i, j).real()), expected);
            ASSERT_LT(cuda::std::abs(img(i, j).imag()), expected);
          }
        }
      }
    };

    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
    this->exec.sync();
    validate_float(image);

    // Pixel z is 0, so both pixel-z assumptions hold and must focus identically.
    {
      SCOPED_TRACE("Float PropSarBpPixelZIsZero");
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
        .props<matx::PropSarBpPixelZIsZero>()).run(this->exec);
      this->exec.sync();
      validate_float(image);
    }
    {
      SCOPED_TRACE("Float PropSarBpPixelZIsFixed");
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
        .props<matx::PropSarBpPixelZIsFixed>()).run(this->exec);
      this->exec.sync();
      validate_float(image);
    }
  }

  // Mixed, FloatFloat, and TaylorFast versions of the backprojector. For these, we need double-precision
  // platform positions and range to the mcp.
  {
    auto platform_positions = matx::make_tensor<double3>({num_pulses});
    auto range_to_mcp = matx::make_tensor<double>({num_pulses});
    MATX_CUDA_CHECK(cudaMemcpy(
      platform_positions.Data(), h_antenna_phase_centers.data(), h_antenna_phase_centers.size() * sizeof(double3), cudaMemcpyHostToDevice));
    MATX_CUDA_CHECK(cudaMemcpy(range_to_mcp.Data(), h_range_to_mcp.data(), h_range_to_mcp.size() * sizeof(double), cudaMemcpyHostToDevice));

    {
      SCOPED_TRACE("Mixed no PhaseLUT");

      // example-begin sar-bp-1
      bp_params.compute_type = matx::SarBpComputeType::Mixed;
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
      // example-end sar-bp-1
      this->exec.sync();

      validate(image, f32_accum_re_thresh, 1.0e-7);
    }

    bp_params.features = matx::SarBpFeature::PhaseLUTOptimization;

    {
      SCOPED_TRACE("Mixed PhaseLUT");

      bp_params.compute_type = matx::SarBpComputeType::Mixed;
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
      this->exec.sync();

      validate(image, f32_accum_re_thresh, 1.0e-2);
    }

    {
      SCOPED_TRACE("FloatFloat");

      bp_params.compute_type = matx::SarBpComputeType::FloatFloat;
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
      this->exec.sync();

      validate(image, f32_accum_re_thresh, 1.0e-2);
    }

    {
      SCOPED_TRACE("TaylorFast second order");
      bp_params.compute_type = matx::SarBpComputeType::TaylorFast;
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)).run(this->exec);
      this->exec.sync();

      validate(image, f32_accum_re_thresh, 1.8e-2);
    }

    {
      SCOPED_TRACE("TaylorFast third order");
      // example-begin sar-bp-2
      bp_params.compute_type = matx::SarBpComputeType::TaylorFast;
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
        .props<matx::PropSarBpTaylorFastAddThirdOrder>()).run(this->exec);
      // example-end sar-bp-2
      this->exec.sync();

      validate(image, f32_accum_re_thresh, 1.8e-2);
    }

    // The pixel z coordinate is 0 for this grid, so PropSarBpPixelZIsZero and
    // PropSarBpPixelZIsFixed both hold and must reproduce the focused image for
    // each of these compute types (and combined with the third-order Taylor
    // property). The variadic prop_tags pack lets a single helper instantiate
    // any combination of properties.
    {
      auto run_props = [&](const char *tag, double im_thresh, [[maybe_unused]] auto... prop_tags) {
        SCOPED_TRACE(tag);
        (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
          .template props<decltype(prop_tags)...>()).run(this->exec);
        this->exec.sync();
        validate(image, f32_accum_re_thresh, im_thresh);
      };

      {
        SCOPED_TRACE("Mixed PropSarBpPixelZIsZero");
        // example-begin sar-bp-3
        bp_params.compute_type = matx::SarBpComputeType::Mixed;
        bp_params.features = matx::SarBpFeature::PhaseLUTOptimization;

        (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
          .props<matx::PropSarBpPixelZIsZero>()).run(this->exec);
        // example-end sar-bp-3
        this->exec.sync();
        validate(image, f32_accum_re_thresh, 1.0e-2);
      }
      run_props("Mixed PropSarBpPixelZIsFixed", 1.0e-2, matx::PropSarBpPixelZIsFixed{});

      bp_params.compute_type = matx::SarBpComputeType::FloatFloat;
      run_props("FloatFloat PropSarBpPixelZIsZero", 1.0e-2, matx::PropSarBpPixelZIsZero{});
      run_props("FloatFloat PropSarBpPixelZIsFixed", 1.0e-2, matx::PropSarBpPixelZIsFixed{});

      bp_params.compute_type = matx::SarBpComputeType::TaylorFast;
      run_props("TaylorFast PropSarBpPixelZIsZero", 1.8e-2, matx::PropSarBpPixelZIsZero{});
      run_props("TaylorFast PropSarBpPixelZIsFixed", 1.8e-2, matx::PropSarBpPixelZIsFixed{});
      run_props("TaylorFast third order PropSarBpPixelZIsZero", 1.8e-2,
        matx::PropSarBpTaylorFastAddThirdOrder{}, matx::PropSarBpPixelZIsZero{});
      run_props("TaylorFast third order PropSarBpPixelZIsFixed", 1.8e-2,
        matx::PropSarBpTaylorFastAddThirdOrder{}, matx::PropSarBpPixelZIsFixed{});
    }
  }

  MATX_EXIT_HANDLER();
}

// Exercise PropSarBpPixelZIsFixed with a genuinely non-zero constant pixel
// height. Unlike the z == 0 grids above, this drives the cached dz^2 path with
// a non-trivial value. We verify that, with the assumption holding, the property
// still produces a correctly focused point-target image for every compute type.
TYPED_TEST(SarBpTestDoubleType, PixelZIsFixedNonZeroHeight)
{
  MATX_ENTER_HANDLER();

  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  constexpr matx::index_t image_width = 132;
  constexpr matx::index_t image_height = 132;
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
  const double target_x = pix_x_min + pix_dx * (image_width / 4.0);
  const double target_y = pix_y_min + pix_dy * (image_height / 4.0);
  // A single focal-plane height shared by EVERY image pixel (and the target).
  // This is exactly the condition PropSarBpPixelZIsFixed asserts -- the height
  // is constant across the whole image, not merely within a thread block. It is
  // kept small enough that the target's range bins stay within the range window
  // (each meter of height shifts the differential range by ~0.7 m and the
  // window spans ~93 m), while being clearly non-zero so the fixed-z value that
  // flows through the kernel is non-trivial (z == 0 would not exercise it).
  const double pixel_z = 10.0;

  matx::SarBpParams bp_params{};
  bp_params.del_r = ::sqrt((image_width * pix_dx) * (image_width * pix_dx) +
                           (image_height * pix_dy) * (image_height * pix_dy)) / num_range_bins;
  bp_params.center_frequency = 10.0e9;
  bp_params.features = matx::SarBpFeature::PhaseLUTOptimization;
  const double bin_offset = static_cast<double>(num_range_bins / 2);

  // Ideal point-target range profiles for the target at (target_x, target_y, pixel_z).
  std::vector<double3> h_apc(num_pulses);
  std::vector<double> h_rtm(num_pulses);
  std::vector<cuda::std::complex<double>> h_rp(num_pulses * num_range_bins, cuda::std::complex<double>{0.0, 0.0});
  for (matx::index_t i = 0; i < num_pulses; i++) {
    h_apc[i] = double3{plat_x_min + plat_dx * static_cast<double>(i), plat_y, plat_z};
    h_rtm[i] = ::sqrt(h_apc[i].x * h_apc[i].x + h_apc[i].y * h_apc[i].y + h_apc[i].z * h_apc[i].z);
    const double R = ::sqrt(
      (h_apc[i].x - target_x) * (h_apc[i].x - target_x) +
      (h_apc[i].y - target_y) * (h_apc[i].y - target_y) +
      (h_apc[i].z - pixel_z) * (h_apc[i].z - pixel_z));
    const double dR = R - h_rtm[i];
    const double ideal_bin = dR / bp_params.del_r + bin_offset;
    const double ideal_phase =
        -4.0 * cuda::std::numbers::pi * dR *
        (bp_params.center_frequency / matx::constants::speed_of_light);
    double sinx, cosx;
    ::sincos(ideal_phase, &sinx, &cosx);
    const cuda::std::complex<double> sample{cosx, sinx};
    const int b = static_cast<int>(floor(ideal_bin));
    if (b >= 0 && b < num_range_bins) h_rp[i * num_range_bins + b] = sample;
    if (b + 1 >= 0 && b + 1 < num_range_bins) h_rp[i * num_range_bins + b + 1] = sample;
  }

  auto platform_positions = matx::make_tensor<double3>({num_pulses});
  auto range_to_mcp = matx::make_tensor<double>({num_pulses});
  MATX_CUDA_CHECK(cudaMemcpy(platform_positions.Data(), h_apc.data(), num_pulses * sizeof(double3), cudaMemcpyHostToDevice));
  MATX_CUDA_CHECK(cudaMemcpy(range_to_mcp.Data(), h_rtm.data(), num_pulses * sizeof(double), cudaMemcpyHostToDevice));

  // Allow a little under two fp32 epsilons per pulse at the focused target.
  const double f32_accum_re_thresh = 2.0e-7 * static_cast<double>(num_pulses);

  // Confirm the property-enabled reconstruction focuses the point target and is
  // free of spurious energy away from it -- the same structure used by the
  // PointTarget test.
  auto validate = [num_pulses](auto &image, double re_thresh, double im_thresh) {
    for (matx::index_t i = 0; i < image_height; i++) {
      for (matx::index_t j = 0; j < image_width; j++) {
        if (i == image_height / 4 && j == image_width / 4) {
          ASSERT_NEAR(static_cast<double>(num_pulses), static_cast<double>(image(i, j).real()), re_thresh);
          ASSERT_NEAR(0.0, static_cast<double>(image(i, j).imag()), im_thresh);
        } else if (i < image_height / 8 || i >= 3 * image_height / 8) {
          ASSERT_EQ(cuda::std::abs(image(i, j).real()), 0);
          ASSERT_EQ(cuda::std::abs(image(i, j).imag()), 0);
        } else {
          ASSERT_LT(cuda::std::abs(image(i, j).real()), num_pulses);
          ASSERT_LT(cuda::std::abs(image(i, j).imag()), num_pulses);
        }
      }
    }
  };

  // Double compute type (complex<double> profiles / image).
  {
    SCOPED_TRACE("Double");
    auto range_profiles = matx::make_tensor<cuda::std::complex<double>>({num_pulses, num_range_bins});
    MATX_CUDA_CHECK(cudaMemcpy(range_profiles.Data(), h_rp.data(), h_rp.size() * sizeof(cuda::std::complex<double>), cudaMemcpyHostToDevice));
    auto px = matx::linspace<double>(pix_x_min, pix_x_min + (image_width - 1) * pix_dx, image_width);
    auto py = matx::linspace<double>(pix_y_min, pix_y_min + (image_height - 1) * pix_dy, image_height);
    auto voxel_locations = matx::zipvec(matx::clone<2>(px, {image_height, matx::matxKeepDim}),
                                        matx::clone<2>(py, {matx::matxKeepDim, image_width}),
                                        pixel_z * matx::ones<double>({image_height, image_width}));
    auto zero_image = matx::zeros<cuda::std::complex<double>>({image_height, image_width});
    auto image = matx::make_tensor<cuda::std::complex<double>>({image_height, image_width});

    bp_params.compute_type = matx::SarBpComputeType::Double;
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
      .props<matx::PropSarBpPixelZIsFixed>()).run(this->exec);
    this->exec.sync();
    validate(image, 1.0e-9, 1.0e-9);
  }

  // complex<float> profiles / image, used by Float, Mixed, FloatFloat, and TaylorFast.
  std::vector<cuda::std::complex<float>> h_rp_f32(h_rp.size());
  for (size_t k = 0; k < h_rp.size(); k++) {
    h_rp_f32[k] = static_cast<cuda::std::complex<float>>(h_rp[k]);
  }
  auto range_profiles_f32 = matx::make_tensor<cuda::std::complex<float>>({num_pulses, num_range_bins});
  MATX_CUDA_CHECK(cudaMemcpy(range_profiles_f32.Data(), h_rp_f32.data(), h_rp_f32.size() * sizeof(cuda::std::complex<float>), cudaMemcpyHostToDevice));
  auto px_f = matx::linspace<float>(static_cast<float>(pix_x_min), static_cast<float>(pix_x_min + (image_width - 1) * pix_dx), image_width);
  auto py_f = matx::linspace<float>(static_cast<float>(pix_y_min), static_cast<float>(pix_y_min + (image_height - 1) * pix_dy), image_height);
  auto voxel_locations_f32 = matx::zipvec(matx::clone<2>(px_f, {image_height, matx::matxKeepDim}),
                                          matx::clone<2>(py_f, {matx::matxKeepDim, image_width}),
                                          static_cast<float>(pixel_z) * matx::ones<float>({image_height, image_width}));
  auto zero_image_f32 = matx::zeros<cuda::std::complex<float>>({image_height, image_width});
  auto image_f32 = matx::make_tensor<cuda::std::complex<float>>({image_height, image_width});

  // Float compute type (fp32 platform positions / range-to-mcp).
  {
    SCOPED_TRACE("Float");
    std::vector<float3> h_apc_f32(num_pulses);
    std::vector<float> h_rtm_f32(num_pulses);
    for (matx::index_t i = 0; i < num_pulses; i++) {
      h_apc_f32[i] = float3{static_cast<float>(h_apc[i].x), static_cast<float>(h_apc[i].y), static_cast<float>(h_apc[i].z)};
      h_rtm_f32[i] = static_cast<float>(h_rtm[i]);
    }
    auto platform_positions_f32 = matx::make_tensor<float3>({num_pulses});
    auto range_to_mcp_f32 = matx::make_tensor<float>({num_pulses});
    MATX_CUDA_CHECK(cudaMemcpy(platform_positions_f32.Data(), h_apc_f32.data(), num_pulses * sizeof(float3), cudaMemcpyHostToDevice));
    MATX_CUDA_CHECK(cudaMemcpy(range_to_mcp_f32.Data(), h_rtm_f32.data(), num_pulses * sizeof(float), cudaMemcpyHostToDevice));

    bp_params.compute_type = matx::SarBpComputeType::Float;
    (image_f32 = matx::experimental::sar_bp(zero_image_f32, range_profiles_f32, platform_positions_f32, voxel_locations_f32, range_to_mcp_f32, bp_params)
      .props<matx::PropSarBpPixelZIsFixed>()).run(this->exec);
    this->exec.sync();

    // Float has lower precision; only require the target to be substantially focused.
    ASSERT_GT(image_f32(image_height / 4, image_width / 4).real(), 0.98f * num_pulses);
    ASSERT_LT(cuda::std::abs(image_f32(image_height / 4, image_width / 4).imag()), 3.0f);
  }

  // Mixed, FloatFloat, and TaylorFast (fp64 platform positions / range-to-mcp).
  // Mixed and FloatFloat compute the range exactly (in fp64 / fltflt), so the
  // focused real part is limited only by fp32 accumulation. TaylorFast uses a
  // local Taylor approximation of the range, so it carries a small additional
  // approximation error and gets a correspondingly looser (but still tight,
  // ~1e-5 relative) real threshold.
  {
    for (auto ct : {matx::SarBpComputeType::Mixed, matx::SarBpComputeType::FloatFloat, matx::SarBpComputeType::TaylorFast}) {
      const bool is_taylor = (ct == matx::SarBpComputeType::TaylorFast);
      SCOPED_TRACE(ct == matx::SarBpComputeType::Mixed ? "Mixed" :
                   (ct == matx::SarBpComputeType::FloatFloat ? "FloatFloat" : "TaylorFast"));
      bp_params.compute_type = ct;
      const double re_thresh = is_taylor ? 1.0e-3 : f32_accum_re_thresh;
      const double im_thresh = is_taylor ? 1.8e-2 : 1.0e-2;
      (image_f32 = matx::experimental::sar_bp(zero_image_f32, range_profiles_f32, platform_positions, voxel_locations_f32, range_to_mcp, bp_params)
        .props<matx::PropSarBpPixelZIsFixed>()).run(this->exec);
      this->exec.sync();
      validate(image_f32, re_thresh, im_thresh);
    }
  }

  MATX_EXIT_HANDLER();
}

// Verify sar_bp forwards PreRun/PostRun to ALL of its operands by wrapping each
// in a lifecycle probe.
TYPED_TEST(SarBpTestNonComplexNonHalfFloatTypes, RangeToMcpOperatorInput)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using complex_t = cuda::std::complex<TestType>;
  using apc_t = cuda::std::conditional_t<cuda::std::is_same_v<TestType, double>, double3, float3>;

  const index_t num_range_bins = 64;
  const index_t num_pulses = 64;
  const index_t image_width = 64;
  const index_t image_height = 64;

  auto zero_image = matx::zeros<complex_t>({image_height, image_width});
  const TestType min_x = -10.0;
  const TestType max_x = 10.0;
  auto pix_coords_x = matx::linspace<TestType>(min_x, max_x, image_width);
  auto pix_coords_y = matx::linspace<TestType>(min_x, max_x, image_height);
  auto pix_coords_yclone = matx::clone<2>(pix_coords_y, {matx::matxKeepDim, image_width});
  auto pix_coords_xclone = matx::clone<2>(pix_coords_x, {image_height, matx::matxKeepDim});
  auto voxel_locations = matx::zipvec(
    pix_coords_xclone, pix_coords_yclone, matx::zeros<TestType>({image_height, image_width}));

  auto range_profiles = matx::ones<complex_t>({num_pulses, num_range_bins});
  auto platform_positions = matx::make_tensor<apc_t>({num_pulses});
  auto range_to_mcp = matx::make_tensor<TestType>({num_pulses});
  const TestType plat_dx = (max_x - min_x) / num_pulses;
  const TestType plat_y = -1000.0;
  const TestType plat_z = 1000.0;
  for (index_t i = 0; i < num_pulses; i++) {
    const TestType plat_x = min_x + static_cast<TestType>(i) * plat_dx;
    platform_positions(i) = apc_t{plat_x, plat_y, plat_z};
    range_to_mcp(i) = ::sqrt(plat_x*plat_x + plat_y*plat_y + plat_z*plat_z);
  }

  SarBpParams params;
  if constexpr (cuda::std::is_same_v<TestType, double>) {
    params.compute_type = SarBpComputeType::Double;
  } else {
    params.compute_type = SarBpComputeType::Float;
  }
  params.features = SarBpFeature::PhaseLUTOptimization;
  params.center_frequency = 10.0e9;
  params.del_r = (max_x - min_x) / num_range_bins;

  // Reference from the raw operands.
  auto image_ref = matx::make_tensor<complex_t>({image_height, image_width});
  (image_ref = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)).run(this->exec);

  // Wrap every operand in a lifecycle probe.
  test::PreRunLifecycle s_img, s_rp, s_pp, s_vl, s_rtm;
  auto image_test = matx::make_tensor<complex_t>({image_height, image_width});
  (image_test = matx::experimental::sar_bp(
      test::make_prerun_tester(zero_image, s_img),
      test::make_prerun_tester(range_profiles, s_rp),
      test::make_prerun_tester(platform_positions, s_pp),
      test::make_prerun_tester(voxel_locations, s_vl),
      test::make_prerun_tester(range_to_mcp, s_rtm), params)).run(this->exec);

  this->exec.sync();

  test::ExpectLifecycleClean(s_img, "initial_image");
  test::ExpectLifecycleClean(s_rp,  "range_profiles");
  test::ExpectLifecycleClean(s_pp,  "platform_positions");
  test::ExpectLifecycleClean(s_vl,  "voxel_locations");
  test::ExpectLifecycleClean(s_rtm, "range_to_mcp");

  for (index_t i = 0; i < image_height; i++) {
    for (index_t j = 0; j < image_width; j++) {
      ASSERT_NEAR(image_test(i, j).real(), image_ref(i, j).real(), this->thresh) << "real mismatch at (" << i << "," << j << ")";
      ASSERT_NEAR(image_test(i, j).imag(), image_ref(i, j).imag(), this->thresh) << "imag mismatch at (" << i << "," << j << ")";
    }
  }

  MATX_EXIT_HANDLER();
}

// TaylorFast reference-relative phase synthesis. These tests call the
// per-sample helpers directly and compare the synthesized phasor with
// e^{i*alpha*(J + n + w - bin_offset)} evaluated in double for the same
// (J, n, w).
namespace {

struct SarBpPhaseSynthSample {
  int32_t J;
  int32_t n;
  float w;
};

// As in the sar_bp transform, the phase correction is formed on the host and
// passed to the kernel by value.
__global__ void SarBpPhaseSynthTestKernel(const SarBpPhaseSynthSample *samples, int count,
                                          SarBpTaylorFastPhaseCorrection c, int32_t bin_offset,
                                          cuda::std::complex<float> *out)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    const uint32_t ref_ticks = SarBpRefPhaseTicks(c, samples[i].J, bin_offset);
    out[i] = SarBpSynthesizePhase(c, ref_ticks, samples[i].n, samples[i].w);
  }
}

// Largest |P - e^{i*phi}| over the samples, which for unit phasors is the
// phase error in radians to first order, or the first non-finite error.
double SarBpPhaseSynthMaxError(const std::vector<SarBpPhaseSynthSample> &h_samples, double alpha, int32_t bin_offset)
{
  const int count = static_cast<int>(h_samples.size());
  auto samples = make_tensor<SarBpPhaseSynthSample>({count});
  auto out = make_tensor<cuda::std::complex<float>>({count});
  MATX_CUDA_CHECK(cudaMemcpy(samples.Data(), h_samples.data(), count * sizeof(SarBpPhaseSynthSample), cudaMemcpyHostToDevice));
  SarBpPhaseSynthTestKernel<<<(count + 255) / 256, 256>>>(
      samples.Data(), count, SarBpTaylorFastPhaseCorrection::Make(alpha), bin_offset, out.Data());
  MATX_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<cuda::std::complex<float>> h_out(count);
  MATX_CUDA_CHECK(cudaMemcpy(h_out.data(), out.Data(), count * sizeof(cuda::std::complex<float>), cudaMemcpyDeviceToHost));
  double max_err = 0.0;
  for (int i = 0; i < count; i++) {
    const SarBpPhaseSynthSample &s = h_samples[i];
    // Reduce the integer-bin part exactly before forming the angle so that
    // the reference itself is accurate for |J| up to 2^24.
    const double rev = alpha / (2.0 * cuda::std::numbers::pi);
    const double k = static_cast<double>(s.J) + static_cast<double>(s.n) - static_cast<double>(bin_offset);
    const double int_rev = std::fmod(std::fmod(rev, 1.0) * k, 1.0);
    const double phi = 2.0 * cuda::std::numbers::pi * int_rev + alpha * static_cast<double>(s.w);
    const double dr = static_cast<double>(h_out[i].real()) - std::cos(phi);
    const double di = static_cast<double>(h_out[i].imag()) - std::sin(phi);
    const double err = std::sqrt(dr * dr + di * di);
    // std::max would drop a NaN error, so return a non-finite error at once;
    // it then fails every tolerance check.
    if (!std::isfinite(err)) {
      return err;
    }
    max_err = std::max(max_err, err);
  }
  return max_err;
}

std::vector<SarBpPhaseSynthSample> SarBpPhaseSynthSamples(float w_lo, float w_hi, int count)
{
  std::mt19937 gen(1234);
  std::uniform_int_distribution<int32_t> J_dist(-(1 << 23), (1 << 24));
  std::uniform_int_distribution<int32_t> n_dist(-256, 256);
  std::uniform_real_distribution<float> w_dist(w_lo, w_hi);
  std::vector<SarBpPhaseSynthSample> samples(count);
  for (auto &s : samples) {
    s = SarBpPhaseSynthSample{J_dist(gen), n_dist(gen), w_dist(gen)};
  }
  // Include the weight endpoints and an exact zero offset.
  if (samples.size() >= 3) {
    samples[0].w = 0.0f;
    samples[1].w = std::nextafter(1.0f, 0.0f);
    samples[2].n = 0;
  }
  return samples;
}

} // namespace

TEST(SarBpPhaseSynthesis, IncrementWrapsAtOneRevolution)
{
  // A fractional revolution per bin within half a tick of one rounds to
  // 2^32, which must narrow to the phase-equivalent increment 0.
  const SarBpTaylorFastPhaseCorrection c = SarBpTaylorFastPhaseCorrection::Make(
      2.0 * cuda::std::numbers::pi * (23.0 + (1.0 - 0x1p-40)));
  EXPECT_EQ(c.Q, 0u);

  const SarBpTaylorFastPhaseCorrection c2 = SarBpTaylorFastPhaseCorrection::Make(
      2.0 * cuda::std::numbers::pi * 23.25);
  EXPECT_EQ(c2.Q, 0x40000000u);
  EXPECT_EQ(c2.Q64, 0x4000000000000000ull);

  // For a tiny negative alpha, frac(alpha / 2pi) rounds to exactly 1, which
  // must also wrap to a zero increment in both widths.
  const SarBpTaylorFastPhaseCorrection c3 = SarBpTaylorFastPhaseCorrection::Make(-1.0e-300);
  EXPECT_EQ(c3.Q64, 0ull);
  EXPECT_EQ(c3.Q, 0u);

  // A non-finite alpha has no meaningful increment; Make must still produce a
  // defined value rather than convert NaN to an integer.
  for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
    const SarBpTaylorFastPhaseCorrection c4 = SarBpTaylorFastPhaseCorrection::Make(bad);
    EXPECT_EQ(c4.Q64, 0ull);
    EXPECT_EQ(c4.Q, 0u);
  }
}

TEST(SarBpPhaseSynthesis, ReferenceTicks)
{
  // A quarter revolution per bin: the reference angle for J - bin_offset = k is
  // k/4 revolutions, reduced modulo one revolution for negative k as well.
  constexpr int32_t bin_offset = 1 << 23;
  const SarBpTaylorFastPhaseCorrection quarter = SarBpTaylorFastPhaseCorrection::Make(
      2.0 * cuda::std::numbers::pi * 23.25);
  EXPECT_EQ(SarBpRefPhaseTicks(quarter, bin_offset, bin_offset), 0u);
  EXPECT_EQ(SarBpRefPhaseTicks(quarter, bin_offset + 2, bin_offset), 0x80000000u);
  EXPECT_EQ(SarBpRefPhaseTicks(quarter, bin_offset + 3, bin_offset), 0xC0000000u);
  EXPECT_EQ(SarBpRefPhaseTicks(quarter, bin_offset - 1, bin_offset), 0xC0000000u);
  EXPECT_EQ(SarBpRefPhaseTicks(quarter, -(1 << 23), bin_offset), 0u);

  // SarBpRefPhaseTicks forms bits 63..32 of Q64*k from 32-bit pieces. Check it
  // against the 64-bit product, including nonzero low words of Q64, which the
  // sign correction for negative k depends on. With Q64 = 1 and k = -1 the
  // product is 2^64 - 1, whose upper word is all ones only with the correction.
  auto expected = [](uint64_t q64, int32_t ref_bin_int, int32_t offset) {
    const uint64_t k = static_cast<uint64_t>(static_cast<int64_t>(ref_bin_int) - offset);
    return static_cast<uint32_t>((q64 * k) >> 32);
  };
  const SarBpTaylorFastPhaseCorrection one{1ull, 0.0f, 0u};
  EXPECT_EQ(SarBpRefPhaseTicks(one, bin_offset - 1, bin_offset), 0xFFFFFFFFu);

  std::mt19937_64 gen(5678);
  std::uniform_int_distribution<int32_t> k_dist(-(1 << 24), 1 << 24);
  for (int i = 0; i < 100000; i++) {
    const SarBpTaylorFastPhaseCorrection c{gen(), 0.0f, 0u};
    const int32_t J = bin_offset + k_dist(gen);
    ASSERT_EQ(SarBpRefPhaseTicks(c, J, bin_offset), expected(c.Q64, J, bin_offset))
        << "Q64=" << c.Q64 << " J=" << J;
  }
}

TEST(SarBpPhaseSynthesis, MatchesDoubleReference)
{
  MATX_ENTER_HANDLER();
  // X band with 0.36 m bins: about 23 revolutions per range bin, so the
  // fractional-bin term spans up to ~146 rad. The error is dominated by
  // __sincosf at that argument magnitude and by rounding alpha*w in fp32.
  // The kernel forms w separately from n, so w can land outside [0, 1) by up
  // to ulp(bin_loc)/2; the samples extend 1e-4 past both ends to cover that.
  const double alpha = 4.0 * cuda::std::numbers::pi * 0.3623 * 9.6e9 / matx::constants::speed_of_light;
  constexpr int32_t bin_offset = (1 << 24) / 2;
  constexpr float excursion = 1.0e-4f;
  const auto samples = SarBpPhaseSynthSamples(-excursion, 1.0f + excursion, 1 << 16);
  EXPECT_LT(SarBpPhaseSynthMaxError(samples, alpha, bin_offset), 1.0e-4);

  // A whole number of revolutions plus a quarter per bin makes Q*n exercise
  // the integer wraparound with no fractional-tick rounding.
  const double alpha_q = 2.0 * cuda::std::numbers::pi * 23.25;
  EXPECT_LT(SarBpPhaseSynthMaxError(samples, alpha_q, bin_offset), 1.0e-4);
  MATX_EXIT_HANDLER();
}

// A phase LUT is used exactly when PhaseLUTOptimization is set and the compute
// type reads a table, which excludes TaylorFast. In particular Double uses one
// only with the feature.
TEST(SarBpPhaseLUT, UsesPhaseLUTFollowsFeatureAndComputeType)
{
  constexpr SarBpParams taylor_fast{SarBpComputeType::TaylorFast, SarBpFeature::PhaseLUTOptimization};
  static_assert(!SarBpUsesPhaseLUT(taylor_fast));

  for (const SarBpComputeType compute_type : {SarBpComputeType::Double, SarBpComputeType::Float,
                                              SarBpComputeType::Mixed, SarBpComputeType::FloatFloat,
                                              SarBpComputeType::TaylorFast}) {
    SCOPED_TRACE(static_cast<int>(compute_type));
    SarBpParams params;
    params.compute_type = compute_type;
    params.features = SarBpFeature::None;
    EXPECT_FALSE(SarBpUsesPhaseLUT(params));
    params.features = SarBpFeature::PhaseLUTOptimization;
    EXPECT_EQ(SarBpUsesPhaseLUT(params), compute_type != SarBpComputeType::TaylorFast);
  }
}

namespace {

// Inputs for the range-profile addressing tests: a 48 x 48 grid at z = 0 whose
// 24 m swath edges fall inside the image, 601 pulses (three 256-pulse blocks,
// the last partial), and range profiles held as a column slice of a wider
// tensor. Each row is padded on both sides with a sentinel that no in-range
// sample read can reach, so the profile view has a padded row stride and an
// offset base pointer.
struct SarBpAddressingInputs {
  using complex_t = cuda::std::complex<float>;
  static constexpr index_t IMAGE = 48;
  static constexpr index_t NUM_PULSES = 601;
  static constexpr index_t NUM_BINS = 48;
  static constexpr index_t PAD = 5;
  static constexpr double DEL_R = 0.5;
  static constexpr float SENTINEL = 1.0e30f;

  tensor_t<double, 2> px = make_tensor<double>({IMAGE, IMAGE});
  tensor_t<double, 2> py = make_tensor<double>({IMAGE, IMAGE});
  tensor_t<double, 2> pz = make_tensor<double>({IMAGE, IMAGE});
  tensor_t<double3, 1> platform_positions = make_tensor<double3>({NUM_PULSES});
  tensor_t<double, 1> range_to_mcp = make_tensor<double>({NUM_PULSES});
  tensor_t<complex_t, 2> padded = make_tensor<complex_t>({NUM_PULSES, NUM_BINS + 2 * PAD});

  SarBpAddressingInputs()
  {
    for (index_t i = 0; i < IMAGE; i++) {
      for (index_t j = 0; j < IMAGE; j++) {
        px(i, j) = static_cast<double>(j) - IMAGE / 2;
        py(i, j) = static_cast<double>(i) - IMAGE / 2;
        pz(i, j) = 0.0;
      }
    }
    for (index_t p = 0; p < NUM_PULSES; p++) {
      const double3 a{-200.0 + 400.0 * static_cast<double>(p) / NUM_PULSES, -5000.0, 5000.0};
      platform_positions(p) = a;
      range_to_mcp(p) = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    }
    std::mt19937 gen(2468);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (index_t p = 0; p < NUM_PULSES; p++) {
      for (index_t b = 0; b < NUM_BINS + 2 * PAD; b++) {
        const bool in_row = b >= PAD && b < PAD + NUM_BINS;
        padded(p, b) = in_row ? complex_t{dist(gen), dist(gen)} : complex_t{SENTINEL, SENTINEL};
      }
    }
  }

  // The profiles without the padding: unit inner stride, row stride NUM_BINS + 2*PAD.
  auto Profiles() const { return slice(padded, {0, PAD}, {matxEnd, PAD + NUM_BINS}); }
  auto Voxels() const { return zipvec(px, py, pz); }

  // Number of pixel-pulse pairs whose exact lower interpolation bin is -1 and
  // NUM_BINS - 1: the samples on either side of the swath that the bounds check
  // must skip.
  std::pair<int, int> EdgeSampleCounts() const
  {
    int below = 0, above = 0;
    for (index_t p = 0; p < NUM_PULSES; p++) {
      const double3 a = platform_positions(p);
      for (index_t i = 0; i < IMAGE; i++) {
        for (index_t j = 0; j < IMAGE; j++) {
          const double dx = px(i, j) - a.x, dy = py(i, j) - a.y, dz = pz(i, j) - a.z;
          const double bin = (std::sqrt(dx * dx + dy * dy + dz * dz) - range_to_mcp(p)) / DEL_R + NUM_BINS / 2;
          const double floor_bin = std::floor(bin);
          below += floor_bin == -1.0;
          above += floor_bin == static_cast<double>(NUM_BINS - 1);
        }
      }
    }
    return {below, above};
  }
};

constexpr SarBpComputeType kSarBpAllComputeTypes[] = {
    SarBpComputeType::Double, SarBpComputeType::Float, SarBpComputeType::Mixed,
    SarBpComputeType::FloatFloat, SarBpComputeType::TaylorFast};

} // namespace

// With PhaseLUTOptimization, unit-stride tensor views take a fast path that
// carries each pulse's row offset across the pulse loop. Compare it with the
// generic accessor path, reached by passing the same profiles through an
// identity operator (conj of conj), across several pulse blocks and with a
// padded row stride. Only the addressing differs, so the images must agree to
// within a few ulps; the two kernel instantiations may contract floating-point
// operations differently. A misaddressed sample changes a pixel by about one
// sample's magnitude, far above the tolerance.
TEST(SarBpAddressing, RowOffsetMatchesGenericAccessor)
{
  MATX_ENTER_HANDLER();
  cudaExecutor exec{};
  using complex_t = SarBpAddressingInputs::complex_t;
  constexpr index_t IMAGE = SarBpAddressingInputs::IMAGE;
  SarBpAddressingInputs in;
  auto profiles = in.Profiles();
  ASSERT_EQ(profiles.Stride(0), SarBpAddressingInputs::NUM_BINS + 2 * SarBpAddressingInputs::PAD);
  ASSERT_EQ(profiles.Stride(1), 1);

  auto zero_image = zeros<complex_t>({IMAGE, IMAGE});
  auto fast = make_tensor<complex_t>({IMAGE, IMAGE});
  auto generic = make_tensor<complex_t>({IMAGE, IMAGE});
  for (const SarBpComputeType compute_type : kSarBpAllComputeTypes) {
    SCOPED_TRACE(static_cast<int>(compute_type));
    SarBpParams params;
    params.compute_type = compute_type;
    params.features = SarBpFeature::PhaseLUTOptimization;
    params.center_frequency = 9.6e9;
    params.del_r = SarBpAddressingInputs::DEL_R;
    (fast = experimental::sar_bp(zero_image, profiles, in.platform_positions, in.Voxels(),
                                 in.range_to_mcp, params)).run(exec);
    (generic = experimental::sar_bp(zero_image, conj(conj(profiles)), in.platform_positions, in.Voxels(),
                                    in.range_to_mcp, params)).run(exec);
    exec.sync();
    double max_mag = 0.0;
    bool finite = true;
    for (index_t i = 0; i < IMAGE; i++) {
      for (index_t j = 0; j < IMAGE; j++) {
        finite = finite && std::isfinite(fast(i, j).real()) && std::isfinite(fast(i, j).imag()) &&
                 std::isfinite(generic(i, j).real()) && std::isfinite(generic(i, j).imag());
        max_mag = std::max(max_mag, static_cast<double>(cuda::std::abs(generic(i, j))));
      }
    }
    // NaN compares false against any tolerance and std::max drops it, so check
    // finiteness explicitly. The bound keeps a sentinel read, which would also
    // inflate the tolerance below, from passing unnoticed; SwathEdgesAndModes
    // checks it in detail.
    ASSERT_TRUE(finite);
    ASSERT_GT(max_mag, 0.0);
    ASSERT_LT(max_mag, 1.0e6);
    // A few float ulps at the image's peak magnitude: rounding differences in
    // the accumulation scale with the partial sums, not with a pixel's final
    // value, which can be small after cancellation.
    const double tol = 4.0 * std::numeric_limits<float>::epsilon() * max_mag;
    int mismatches = 0;
    double max_diff = 0.0;
    for (index_t i = 0; i < IMAGE; i++) {
      for (index_t j = 0; j < IMAGE; j++) {
        const double dr = std::abs(static_cast<double>(fast(i, j).real()) - generic(i, j).real());
        const double di = std::abs(static_cast<double>(fast(i, j).imag()) - generic(i, j).imag());
        mismatches += !(dr <= tol && di <= tol);
        max_diff = std::max(max_diff, std::max(dr, di));
      }
    }
    EXPECT_EQ(mismatches, 0) << "max difference " << max_diff << ", tolerance " << tol;
  }
  MATX_EXIT_HANDLER();
}

// Run every compute type, with and without PhaseLUTOptimization, in the
// variable-z and PixelZIsZero modes on the shared inputs:
// - The samples just outside the swath sit next to the sentinel padding, so
//   admitting a lower bin of -1 or NUM_BINS - 1 would read a sentinel and blow
//   up the affected pixels. The geometry is checked to produce both edge cases.
// - The grid is at z = 0, so across all pulse blocks the uniform-z kernels must
//   match the variable-z ones.
// - TaylorFast synthesizes its phase, so it accepts either feature setting and
//   produces the same image; FloatFloat without the feature throws.
TEST(SarBpAddressing, SwathEdgesAndModes)
{
  MATX_ENTER_HANDLER();
  cudaExecutor exec{};
  using complex_t = SarBpAddressingInputs::complex_t;
  constexpr index_t IMAGE = SarBpAddressingInputs::IMAGE;
  SarBpAddressingInputs in;
  const auto [below, above] = in.EdgeSampleCounts();
  ASSERT_GT(below, 0);
  ASSERT_GT(above, 0);

  auto zero_image = zeros<complex_t>({IMAGE, IMAGE});
  auto variable_z = make_tensor<complex_t>({IMAGE, IMAGE});
  auto zero_z = make_tensor<complex_t>({IMAGE, IMAGE});
  auto taylor_without_feature = make_tensor<complex_t>({IMAGE, IMAGE});
  auto run = [&](auto &image, const SarBpParams &params, bool pixel_z_is_zero) {
    auto op = experimental::sar_bp(zero_image, in.Profiles(), in.platform_positions, in.Voxels(),
                                   in.range_to_mcp, params);
    if (pixel_z_is_zero) {
      (image = op.template props<PropSarBpPixelZIsZero>()).run(exec);
    } else {
      (image = op).run(exec);
    }
    exec.sync();
  };
  // Every pixel sums at most NUM_PULSES unit-variance samples, so a sentinel
  // read shows up as a huge or non-finite pixel.
  auto expect_bounded = [&](auto &image) {
    double max_mag = 0.0;
    bool finite = true;
    for (index_t i = 0; i < IMAGE; i++) {
      for (index_t j = 0; j < IMAGE; j++) {
        finite = finite && std::isfinite(image(i, j).real()) && std::isfinite(image(i, j).imag());
        max_mag = std::max(max_mag, static_cast<double>(cuda::std::abs(image(i, j))));
      }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(max_mag, 0.0);
    EXPECT_LT(max_mag, 1.0e6);
    return max_mag;
  };
  // Propagates NaN, which std::max would drop, so a NaN difference fails the
  // comparisons below.
  auto max_abs_diff = [&](auto &a, auto &b) {
    double max_diff = 0.0;
    for (index_t i = 0; i < IMAGE; i++) {
      for (index_t j = 0; j < IMAGE; j++) {
        const double d = static_cast<double>(cuda::std::abs(a(i, j) - b(i, j)));
        if (std::isnan(d) || d > max_diff) {
          max_diff = d;
        }
      }
    }
    return max_diff;
  };

  for (const SarBpComputeType compute_type : kSarBpAllComputeTypes) {
    for (const SarBpFeature features : {SarBpFeature::None, SarBpFeature::PhaseLUTOptimization}) {
      SCOPED_TRACE(::testing::Message() << "compute_type=" << static_cast<int>(compute_type)
                                        << ", phase_lut=" << (features == SarBpFeature::PhaseLUTOptimization));
      SarBpParams params;
      params.compute_type = compute_type;
      params.features = features;
      params.center_frequency = 9.6e9;
      params.del_r = SarBpAddressingInputs::DEL_R;
      if (compute_type == SarBpComputeType::FloatFloat && features == SarBpFeature::None) {
        ASSERT_THROW(run(variable_z, params, false), matx::detail::matxException);
        continue;
      }
      run(variable_z, params, false);
      run(zero_z, params, true);
      const double max_mag = expect_bounded(variable_z);
      expect_bounded(zero_z);
      EXPECT_LT(max_abs_diff(variable_z, zero_z), 1.0e-4 * max_mag);
      if (compute_type == SarBpComputeType::TaylorFast) {
        if (features == SarBpFeature::None) {
          (taylor_without_feature = variable_z).run(exec);
          exec.sync();
        } else {
          EXPECT_EQ(max_abs_diff(variable_z, taylor_without_feature), 0.0);
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}
