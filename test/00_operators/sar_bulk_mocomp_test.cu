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
////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <cuda/std/numbers>
#include <type_traits>

#include "matx.h"
#include "operator_test_types.hpp"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

namespace {

template <typename T> using Complex = cuda::std::complex<T>;

template <typename FxOp, typename TargetRangeOp>
concept CanApplySarBulkMocomp = requires(
    const FxOp &fx, const TargetRangeOp &target_reference_range,
    const experimental::SarBulkMocompParams &params) {
  experimental::sar_bulk_mocomp(fx, target_reference_range, params);
};

template <typename FxOp, typename InitialRangeOp, typename TargetRangeOp>
concept CanChangeSarBulkMocompReference = requires(
    const FxOp &fx, const InitialRangeOp &initial_reference_range,
    const TargetRangeOp &target_reference_range,
    const experimental::SarBulkMocompParams &params) {
  experimental::sar_bulk_mocomp(fx, initial_reference_range,
                                target_reference_range, params);
};

template <typename Op, int DynamicSharedMemoryBytes>
class DynamicSharedMemoryOp
    : public BaseOp<DynamicSharedMemoryOp<Op, DynamicSharedMemoryBytes>> {
private:
  using op_type = matx::detail::base_type_t<Op>;
  op_type op_;

public:
  using matxop = bool;
  using value_type = typename op_type::value_type;

  explicit DynamicSharedMemoryOp(const Op &op) : op_(op) {}

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t
  Rank() {
    return op_type::Rank();
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t
  Size(int32_t dim) const {
    return op_.Size(dim);
  }

  template <matx::detail::OperatorCapability Cap, typename InType>
  auto get_capability(InType &in) const {
    if constexpr (Cap == matx::detail::OperatorCapability::DYN_SHM_SIZE) {
      return DynamicSharedMemoryBytes;
    } else {
      return matx::detail::get_operator_capability<Cap>(op_, in);
    }
  }
};

template <typename ExecType> constexpr matxMemorySpace_t TestMemorySpace() {
  if constexpr (is_cuda_executor_v<ExecType>) {
    return MATX_MANAGED_MEMORY;
  } else {
    return MATX_HOST_MALLOC_MEMORY;
  }
}

template <typename RangeType>
Complex<RangeType>
ExpectedCorrection(Complex<RangeType> input, RangeType range_offset,
                   index_t sample, index_t num_samples,
                   const experimental::SarBulkMocompParams &params) {
  const auto center_sample = num_samples / 2;
  const RangeType reference_phase_over_pi_per_meter = static_cast<RangeType>(
      -static_cast<double>(params.sgn) * 4.0 *
      params.phase_reference_frequency / matx::constants::speed_of_light);
  const RangeType sample_phase_over_pi_per_meter = static_cast<RangeType>(
      -static_cast<double>(params.sgn) * 4.0 * params.sample_frequency_spacing /
      matx::constants::speed_of_light);
  const RangeType phase_over_pi_per_meter =
      std::fma(static_cast<RangeType>(sample - center_sample),
               sample_phase_over_pi_per_meter,
               reference_phase_over_pi_per_meter);
  const RangeType phase_over_pi = phase_over_pi_per_meter * range_offset;
  const double phase =
      cuda::std::numbers::pi * static_cast<double>(phase_over_pi);
  const Complex<RangeType> phasor{
      static_cast<RangeType>(std::cos(phase)),
      static_cast<RangeType>(std::sin(phase))};
  return input * phasor;
}

template <typename RangeType, typename ExecType>
void RunRank2ReferenceCase(ExecType &exec, index_t num_samples, int sgn) {
  using FxType = Complex<RangeType>;
  constexpr index_t num_pulses = 3;

  constexpr auto memory_space = TestMemorySpace<ExecType>();
  auto fx = make_tensor<FxType>({num_pulses, num_samples}, memory_space);
  auto range_to_mcp = make_tensor<RangeType>({num_pulses}, memory_space);
  auto output = make_tensor<FxType>({num_pulses, num_samples}, memory_space);

  for (index_t pulse = 0; pulse < num_pulses; ++pulse) {
    const auto pulse_value = static_cast<RangeType>(pulse);
    range_to_mcp(pulse) = static_cast<RangeType>(1.25) +
                          static_cast<RangeType>(0.5) * pulse_value;
    for (index_t sample = 0; sample < num_samples; ++sample) {
      const auto sample_value = static_cast<RangeType>(sample);
      fx(pulse, sample) =
          FxType{static_cast<RangeType>(0.25) + pulse_value +
                     static_cast<RangeType>(0.125) * sample_value,
                 static_cast<RangeType>(-0.5) +
                     static_cast<RangeType>(0.25) * pulse_value -
                     static_cast<RangeType>(0.0625) * sample_value};
    }
  }

  // example-begin sar-bulk-mocomp-1
  const experimental::SarBulkMocompParams params{
      .phase_reference_frequency = 9.6e9,
      .sample_frequency_spacing = 2.5e4,
      .sgn = sgn,
  };

  auto compensated =
      experimental::sar_bulk_mocomp(fx, range_to_mcp, params);
  (output = compensated).run(exec);
  // example-end sar-bulk-mocomp-1
  exec.sync();

  const double tolerance = std::is_same_v<RangeType, float> ? 2.0e-5 : 1.0e-12;
  for (index_t pulse = 0; pulse < num_pulses; ++pulse) {
    for (index_t sample = 0; sample < num_samples; ++sample) {
      const auto expected = ExpectedCorrection(
          fx(pulse, sample), range_to_mcp(pulse), sample, num_samples, params);
      EXPECT_NEAR(output(pulse, sample).real(), expected.real(), tolerance);
      EXPECT_NEAR(output(pulse, sample).imag(), expected.imag(), tolerance);
    }
  }

  // Applying the opposite range offset in place should recover the input.
  (output = experimental::sar_bulk_mocomp(output, -range_to_mcp, params))
      .run(exec);
  exec.sync();
  for (index_t pulse = 0; pulse < num_pulses; ++pulse) {
    for (index_t sample = 0; sample < num_samples; ++sample) {
      EXPECT_NEAR(output(pulse, sample).real(), fx(pulse, sample).real(),
                  2.0 * tolerance);
      EXPECT_NEAR(output(pulse, sample).imag(), fx(pulse, sample).imag(),
                  2.0 * tolerance);
    }
  }
}

template <typename TensorType>
class SarBulkMocompTests : public ::testing::Test {};

using SarBulkMocompTestTypes =
    TupleToTypes<TypedCartesianProduct<cuda::std::tuple<float, double>,
                                       ExecutorTypesAllWithJIT>::type>::type;

TYPED_TEST_SUITE(SarBulkMocompTests, SarBulkMocompTestTypes);

template <typename Executor>
class SarBulkMocompMixedPrecisionGPUTests : public ::testing::Test {};

#ifdef MATX_EN_JIT
using SarBulkMocompMixedPrecisionGPUExecutorTypes =
    ::testing::Types<cudaExecutor, CUDAJITExecutor>;
#else
using SarBulkMocompMixedPrecisionGPUExecutorTypes =
    ::testing::Types<cudaExecutor>;
#endif

TYPED_TEST_SUITE(SarBulkMocompMixedPrecisionGPUTests,
                 SarBulkMocompMixedPrecisionGPUExecutorTypes);

} // namespace

TYPED_TEST(SarBulkMocompTests, Rank2OddAndEvenFrequencyGrids) {
  MATX_ENTER_HANDLER();
  using RangeType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};
  RunRank2ReferenceCase<RangeType>(exec, 7, -1);
  RunRank2ReferenceCase<RangeType>(exec, 8, 1);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(SarBulkMocompTests, TransformBackedRangeOffset) {
  MATX_ENTER_HANDLER();
  using RangeType = cuda::std::tuple_element_t<0, TypeParam>;
  using FxType = Complex<RangeType>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  constexpr index_t pulses = 3;
  constexpr index_t samples = 32;
  constexpr index_t range_terms = 16;
  ExecType exec{};
  constexpr auto memory_space = TestMemorySpace<ExecType>();

  auto fx = make_tensor<FxType>({pulses, samples}, memory_space);
  auto range_components =
      make_tensor<RangeType>({pulses, range_terms}, memory_space);
  auto output = make_tensor<FxType>({pulses, samples}, memory_space);

  for (index_t pulse = 0; pulse < pulses; ++pulse) {
    for (index_t term = 0; term < range_terms; ++term) {
      range_components(pulse, term) =
          static_cast<RangeType>(0.125) * static_cast<RangeType>(pulse + 1) +
          static_cast<RangeType>(0.015625) * static_cast<RangeType>(term + 1);
    }
    for (index_t sample = 0; sample < samples; ++sample) {
      fx(pulse, sample) = FxType{
          static_cast<RangeType>(1.0) +
              static_cast<RangeType>(0.25) * static_cast<RangeType>(pulse),
          static_cast<RangeType>(-0.5) +
              static_cast<RangeType>(0.03125) * static_cast<RangeType>(sample)};
    }
  }

  const experimental::SarBulkMocompParams params{2.5e6, 5.0e4, -1};
  (output =
       experimental::sar_bulk_mocomp(fx, sum(range_components, {1}), params))
      .run(exec);
  exec.sync();

  const double tolerance = std::is_same_v<RangeType, float> ? 2.0e-5 : 1.0e-12;
  for (index_t pulse = 0; pulse < pulses; ++pulse) {
    RangeType range_offset = static_cast<RangeType>(0);
    for (index_t term = 0; term < range_terms; ++term) {
      range_offset += range_components(pulse, term);
    }
    for (index_t sample = 0; sample < samples; ++sample) {
      const auto expected = ExpectedCorrection(fx(pulse, sample), range_offset,
                                               sample, samples, params);
      EXPECT_NEAR(output(pulse, sample).real(), expected.real(), tolerance);
      EXPECT_NEAR(output(pulse, sample).imag(), expected.imag(), tolerance);
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(SarBulkMocompTests, BatchedReferenceChangeAndComposition) {
  MATX_ENTER_HANDLER();
  using RangeType = cuda::std::tuple_element_t<0, TypeParam>;
  using FxType = Complex<RangeType>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  constexpr index_t batches = 2;
  constexpr index_t pulses = 3;
  constexpr index_t samples = 8;
  ExecType exec{};
  constexpr auto memory_space = TestMemorySpace<ExecType>();

  auto fx = make_tensor<FxType>({batches, pulses, samples}, memory_space);
  auto initial_ranges = make_tensor<RangeType>({batches, pulses}, memory_space);
  auto target_ranges = make_tensor<RangeType>({batches, pulses}, memory_space);
  auto output = make_tensor<FxType>({batches, pulses, samples}, memory_space);
  auto restored = make_tensor<FxType>({batches, pulses, samples}, memory_space);

  for (index_t batch = 0; batch < batches; ++batch) {
    const auto batch_value = static_cast<RangeType>(batch);
    for (index_t pulse = 0; pulse < pulses; ++pulse) {
      const auto pulse_value = static_cast<RangeType>(pulse);
      initial_ranges(batch, pulse) =
          static_cast<RangeType>(0.5) +
          static_cast<RangeType>(0.25) * batch_value +
          static_cast<RangeType>(0.125) * pulse_value;
      target_ranges(batch, pulse) = static_cast<RangeType>(2.0) +
                                    static_cast<RangeType>(0.5) * batch_value +
                                    static_cast<RangeType>(0.25) * pulse_value;
      for (index_t sample = 0; sample < samples; ++sample) {
        const auto sample_value = static_cast<RangeType>(sample);
        fx(batch, pulse, sample) =
            FxType{static_cast<RangeType>(1.0) + batch_value +
                       static_cast<RangeType>(0.25) * pulse_value,
                   static_cast<RangeType>(-0.5) +
                       static_cast<RangeType>(0.125) * sample_value};
      }
    }
  }

  const experimental::SarBulkMocompParams params{2.5e6, 5.0e4, -1};
  (output = experimental::sar_bulk_mocomp(
                fx * static_cast<RangeType>(2),
                initial_ranges, target_ranges, params))
      .run(exec);
  (restored = experimental::sar_bulk_mocomp(output, target_ranges,
                                            initial_ranges, params) /
              FxType{static_cast<RangeType>(2), static_cast<RangeType>(0)})
      .run(exec);
  exec.sync();

  const double tolerance = std::is_same_v<RangeType, float> ? 3.0e-5 : 2.0e-12;
  for (index_t batch = 0; batch < batches; ++batch) {
    for (index_t pulse = 0; pulse < pulses; ++pulse) {
      for (index_t sample = 0; sample < samples; ++sample) {
        const RangeType delta =
            target_ranges(batch, pulse) - initial_ranges(batch, pulse);
        const auto expected =
            ExpectedCorrection(fx(batch, pulse, sample), delta, sample, samples,
                               params) *
            FxType{static_cast<RangeType>(2), static_cast<RangeType>(0)};
        EXPECT_NEAR(output(batch, pulse, sample).real(), expected.real(),
                    tolerance);
        EXPECT_NEAR(output(batch, pulse, sample).imag(), expected.imag(),
                    tolerance);
        EXPECT_NEAR(restored(batch, pulse, sample).real(),
                    fx(batch, pulse, sample).real(), tolerance);
        EXPECT_NEAR(restored(batch, pulse, sample).imag(),
                    fx(batch, pulse, sample).imag(), tolerance);
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TEST(SarBulkMocompTests, PhasePrecisionFollowsRangeType) {
  MATX_ENTER_HANDLER();
  SingleThreadedHostExecutor exec{};

  auto fx_double =
      make_tensor<Complex<double>>({1, 5}, MATX_HOST_MALLOC_MEMORY);
  auto range_float = make_tensor<float>({1}, MATX_HOST_MALLOC_MEMORY);
  auto out_double =
      make_tensor<Complex<double>>({1, 5}, MATX_HOST_MALLOC_MEMORY);
  range_float(0) = 3.25f;

  auto fx_float = make_tensor<Complex<float>>({1, 5}, MATX_HOST_MALLOC_MEMORY);
  auto range_double = make_tensor<double>({1}, MATX_HOST_MALLOC_MEMORY);
  auto out_float = make_tensor<Complex<float>>({1, 5}, MATX_HOST_MALLOC_MEMORY);
  range_double(0) = 4.5;
  for (index_t sample = 0; sample < 5; ++sample) {
    fx_double(0, sample) = Complex<double>{1.0, -0.25};
    fx_float(0, sample) = Complex<float>{1.0f, 0.5f};
  }

  const experimental::SarBulkMocompParams params{9.6e9, 3.0e6, -1};
  (out_double = experimental::sar_bulk_mocomp(fx_double, range_float, params))
      .run(exec);
  (out_float = experimental::sar_bulk_mocomp(fx_float, range_double, params))
      .run(exec);
  exec.sync();

  for (index_t sample = 0; sample < 5; ++sample) {
    const auto expected_double_phase = ExpectedCorrection(
        Complex<float>{1.0f, -0.25f}, range_float(0), sample, 5, params);
    EXPECT_NEAR(out_double(0, sample).real(), expected_double_phase.real(),
                2.0e-5);
    EXPECT_NEAR(out_double(0, sample).imag(), expected_double_phase.imag(),
                2.0e-5);

    const auto expected_float_output = ExpectedCorrection(
        Complex<double>{1.0, 0.5}, range_double(0), sample, 5, params);
    EXPECT_NEAR(out_float(0, sample).real(), expected_float_output.real(),
                2.0e-6);
    EXPECT_NEAR(out_float(0, sample).imag(), expected_float_output.imag(),
                2.0e-6);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(SarBulkMocompMixedPrecisionGPUTests,
           FloatOutputPreservesDoublePhaseReduction) {
  MATX_ENTER_HANDLER();
  constexpr index_t pulses = 5;
  constexpr index_t samples = 17;
  TypeParam exec{};

  auto fx = make_tensor<Complex<float>>({pulses, samples}, MATX_MANAGED_MEMORY);
  auto ranges = make_tensor<double>({pulses}, MATX_MANAGED_MEMORY);
  auto output =
      make_tensor<Complex<float>>({pulses, samples}, MATX_MANAGED_MEMORY);

  // Cover ordinary offsets, spaceborne slant ranges, negative offsets, and a
  // very large phase whose fractional cycle would be lost if it were narrowed
  // to float before reduction.
  const double range_values[pulses] = {
      0.125, 1.25, 725000.125, -725000.375, 8000000.0625};
  for (index_t pulse = 0; pulse < pulses; ++pulse) {
    ranges(pulse) = range_values[pulse];
    for (index_t sample = 0; sample < samples; ++sample) {
      fx(pulse, sample) =
          Complex<float>{0.75f + 0.125f * static_cast<float>(pulse),
                         -0.5f + 0.03125f * static_cast<float>(sample)};
    }
  }

  const experimental::SarBulkMocompParams params{9.6e9, 27500.0, -1};
  (output = experimental::sar_bulk_mocomp(fx, ranges, params)).run(exec);
  exec.sync();

  for (index_t pulse = 0; pulse < pulses; ++pulse) {
    for (index_t sample = 0; sample < samples; ++sample) {
      const auto input = Complex<double>{fx(pulse, sample).real(),
                                         fx(pulse, sample).imag()};
      const auto expected = ExpectedCorrection(input, ranges(pulse), sample,
                                               samples, params);
      EXPECT_NEAR(output(pulse, sample).real(), expected.real(), 1.0e-6);
      EXPECT_NEAR(output(pulse, sample).imag(), expected.imag(), 1.0e-6);
    }
  }
  MATX_EXIT_HANDLER();
}

TEST(SarBulkMocompTests, DynamicSharedMemoryIsAdditive) {
  MATX_ENTER_HANDLER();
  auto fx = make_tensor<Complex<float>>({3, 8}, MATX_HOST_MALLOC_MEMORY);
  auto ranges = make_tensor<float>({3}, MATX_HOST_MALLOC_MEMORY);
  const DynamicSharedMemoryOp<decltype(fx), 4096> fx_with_shmem{fx};
  const DynamicSharedMemoryOp<decltype(ranges), 3072> ranges_with_shmem{
      ranges};

  const auto op = experimental::sar_bulk_mocomp(
      fx_with_shmem, ranges_with_shmem,
      experimental::SarBulkMocompParams{1.0e6, 1.0e4, -1});
  EXPECT_EQ(
      matx::detail::get_operator_capability<
          matx::detail::OperatorCapability::DYN_SHM_SIZE>(op),
      7168);
  MATX_EXIT_HANDLER();
}

TEST(SarBulkMocompTests, ValidatesParametersAndShape) {
  MATX_ENTER_HANDLER();
  auto fx = make_tensor<Complex<float>>({3, 8}, MATX_HOST_MALLOC_MEMORY);
  auto good_ranges = make_tensor<float>({3}, MATX_HOST_MALLOC_MEMORY);
  auto bad_ranges = make_tensor<float>({2}, MATX_HOST_MALLOC_MEMORY);

  using FxType = decltype(fx);
  using RangeType = decltype(good_ranges);
  static_assert(CanApplySarBulkMocomp<FxType, RangeType>);
  static_assert(!CanApplySarBulkMocomp<FxType, double>);
  static_assert(!CanApplySarBulkMocomp<double, RangeType>);
  static_assert(
      CanChangeSarBulkMocompReference<FxType, RangeType, RangeType>);
  static_assert(
      !CanChangeSarBulkMocompReference<FxType, double, RangeType>);
  static_assert(
      !CanChangeSarBulkMocompReference<FxType, RangeType, double>);

  EXPECT_THROW(
      (experimental::sar_bulk_mocomp(
          fx, bad_ranges, experimental::SarBulkMocompParams{1.0e6, 1.0e4, -1})),
      matx::detail::matxException);
  EXPECT_THROW(
      (experimental::sar_bulk_mocomp(
          fx, good_ranges, experimental::SarBulkMocompParams{1.0e6, 1.0e4, 0})),
      matx::detail::matxException);
  EXPECT_THROW(
      (experimental::sar_bulk_mocomp(
          fx, good_ranges, experimental::SarBulkMocompParams{1.0e6, 0.0, -1})),
      matx::detail::matxException);
  auto op = experimental::sar_bulk_mocomp(
      fx, good_ranges, experimental::SarBulkMocompParams{1.0e6, 1.0e4, -1});
#ifdef MATX_EN_JIT
  EXPECT_TRUE(jit_supported(op));
#else
  EXPECT_FALSE(jit_supported(op));
#endif
  MATX_EXIT_HANDLER();
}
