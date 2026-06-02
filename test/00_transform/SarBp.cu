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

// Test Mixed, FloatFloat, and TaylorFast precisions. These precisions are used when the user wants
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

  // Enable PhaseLUTOptimization for all remaining tests
  params.features = SarBpFeature::PhaseLUTOptimization;

  {
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

  {
    params.compute_type = SarBpComputeType::TaylorFast;
    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)).run(this->exec);
    this->exec.sync();

    // We reset TaylorFast and enable PhaseLUTOptimization below to include the text in the
    // example that will be used in the documentation.
    // example-begin sar-bp-2
    params.compute_type = SarBpComputeType::TaylorFast;
    params.features = SarBpFeature::PhaseLUTOptimization;

    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)
      .props<PropSarBpTaylorFastAddThirdOrder>()).run(this->exec);
    // example-end sar-bp-2
    this->exec.sync();
  }

  // The voxel_locations grid in this test places every pixel on the z = 0 plane,
  // so the PropSarBpPixelZIsZero assumption holds and can be applied to any
  // compute type.
  {
    // example-begin sar-bp-3
    params.compute_type = SarBpComputeType::Mixed;
    params.features = SarBpFeature::PhaseLUTOptimization;

    (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params)
      .props<PropSarBpPixelZIsZero>()).run(this->exec);
    // example-end sar-bp-3
    this->exec.sync();
  }

  MATX_EXIT_HANDLER();
}

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
  // Target range relative to mocomp point (0,0,0)
  const double target_R = ::sqrt(target_x * target_x + target_y * target_y);

  matx::SarBpParams bp_params{};
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
  // Single-precision machine epsilon is ~1.1921e-7; allow one fp32 epsilon
  // per pulse of accumulation at the focused target pixel.
  const double f32_accum_re_thresh = 1.2e-7 * static_cast<double>(num_pulses);

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
            const float expected = 0.9f * num_pulses;
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
      (image = matx::experimental::sar_bp(zero_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, bp_params)
        .props<matx::PropSarBpTaylorFastAddThirdOrder>()).run(this->exec);
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

      bp_params.compute_type = matx::SarBpComputeType::Mixed;
      run_props("Mixed PropSarBpPixelZIsZero", 1.0e-2, matx::PropSarBpPixelZIsZero{});
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
  const double bin_offset = 0.5 * (num_range_bins - 1);

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
    const double ideal_phase = -4.0 * M_PI * dR * (bp_params.center_frequency / SPEED_OF_LIGHT);
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

  const double f32_accum_re_thresh = 1.2e-7 * static_cast<double>(num_pulses);

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
