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

#pragma once

#include <complex>
#include <cuda.h>
#include <iomanip>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "matx/core/utils.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"
#include "matx/kernels/fltflt.h"
#include "matx/kernels/tensor_accessor.h"

#define PULSE_BLOCK_SIZE 256

namespace matx {

// SI-defined speed of light in m/s. The speed of light through the atmosphere will be roughly 0.03% slower
// than this, but it is assumed that any corrections for atmospheric propagation will be done elsewhere.
static constexpr double SPEED_OF_LIGHT = 299792458.0;

#ifdef __CUDACC__

// Use two iterations of Newton-Raphson to compute the square root of a double. The
// initial estimate is computed using a single-precision square root.
static __device__ __forceinline__ double NewtonRaphsonSqrt(double x) {
    const float est = sqrtf(static_cast<float>(x));
    // We perform this comparison after the sqrtf() to avoid the fp64 comparison.
    // It is rare that x will be 0, so there is generally not much advantage to an early exit.
    if (est == 0.0f) return 0.0;
    const float est_2_inv = __fdividef(1.0f, 2.0f * est);
    const double est_f64 = static_cast<double>(est);
    const double est_2_inv_f64 = static_cast<double>(est_2_inv);
    const double NR_1 = est_f64 - (est_f64 * est_f64 - x) * est_2_inv_f64;
    // Reuse est_2_inv_f64 in place of 1/(2*NR_1) for the second iteration.
    // This is an approximation, but it is accurate enough for our purposes
    // and avoids the need for a second division.
    const double NR_2 = NR_1 - (NR_1 * NR_1 - x) * est_2_inv_f64;
    return NR_2;
}

// ComputeBinToPixelFloatFloat() fuses the FloatFloat path's
// coordinate-difference, squared-norm, sqrt, and scale/add steps that would
// otherwise be a separate norm3d + fma_approx pair. The kernel only needs the
// range bin (FloatFloat requires PhaseLUT, so the bare range R is never used
// for phase), so we never materialize a canonical R -- the Newton sqrt
// correction is treated as the low part of R and fed directly into the
// scaled-add. Mathematically this is the same approximation class as
// fma_approx(fltflt_sqrt_fast(...), dr_inv, mcp_partial), but it saves the
// final fast_two_sum normalization inside sqrt_fast (~3 fp32 ops per
// pixel-pulse).
//
// Stages:
//   1. Loose dx/dy/dz via TwoSum + lo-add (no trailing renormalize). Resulting
//      pairs may be non-canonical when px ~= apx.hi (cancellation), so the
//      norm-3d stage below retains lo*lo terms.
//   2. Squared-norm accumulation with lo*lo retention -- equivalent to a
//      loose fltflt_norm3d() that accepts non-canonical inputs (see comments
//      below for the cancellation rationale).
//   3. One Newton-Raphson sqrt step on (sum_hi, sum_lo) without the final
//      fast_two_sum -- yn ~= sqrt(sum_hi), r_lo = Newton correction, with
//      (yn + r_lo) representing R to ~fast_sqrt precision.
//   4. bin = R * dr_inv + mcp_partial via fma_approx fed with the
//      non-canonical R = {yn, r_lo}. fma_approx drops the a.lo*b.lo term
//      regardless, so the only "lost" piece versus a canonical R is
//      r_lo * dr_inv.lo, which is O(ULP^2 * R) and well below the kernel's
//      error budget.
//
// Domain: at least one of dx/dy/dz must not fully cancel (sum of squared
// distances must be strictly positive) -- three-way simultaneous
// cancellation is not a meaningful SAR geometry and is not supported.
__device__ inline fltflt ComputeBinToPixelFloatFloat(
    fltflt apx, fltflt apy, fltflt apz,
    float px, float py, float pz,
    fltflt dr_inv,
    fltflt mcp_partial)
{
    // Stage 1: loose dx/dy/dz. We use the general fltflt_two_sum (6 fp32 ops)
    // for all three dimensions. When it is known that one coordinate is
    // always greater than the other (e.g., |apz.hi| >= |pz|), the faster
    // fltflt_fast_two_sum as fltflt_fast_two_sum(apz.hi, -pz) could be used
    // instead.
    fltflt dx = fltflt_two_sum(px, -apx.hi);
    fltflt dy = fltflt_two_sum(py, -apy.hi);
    fltflt dz = fltflt_two_sum(pz, -apz.hi);
    dx.lo = detail::fadd_rn(dx.lo, -apx.lo);
    dy.lo = detail::fadd_rn(dy.lo, -apy.lo);
    dz.lo = detail::fadd_rn(dz.lo, -apz.lo);

    // Stage 2: squared-norm accumulation. Same body as the canonical
    // fltflt_norm3d() but with explicit dx.lo*dx.lo / dy.lo*dy.lo /
    // dz.lo*dz.lo contributions retained. The canonical helper drops those
    // as O(eps^2 * sum_hi); for our non-canonical inputs (post-cancellation
    // |lo| can approach |hi|), they carry the cancelled dimension's full
    // contribution and must be kept. Cost: 3 fmaf_rn ops vs canonical.
    const fltflt px2 = fltflt_two_prod_fma(dx.hi, dx.hi);
    const fltflt py2 = fltflt_two_prod_fma(dy.hi, dy.hi);
    const fltflt pz2 = fltflt_two_prod_fma(dz.hi, dz.hi);
    const fltflt s = fltflt_two_sum(px2.hi, py2.hi);
    const fltflt t = fltflt_two_sum(s.hi, pz2.hi);

    float sum_lo = detail::fadd_rn(t.lo, s.lo);
    sum_lo = detail::fadd_rn(sum_lo, px2.lo);
    sum_lo = detail::fadd_rn(sum_lo, py2.lo);
    sum_lo = detail::fadd_rn(sum_lo, pz2.lo);
    sum_lo = detail::fmaf_rn(detail::fadd_rn(dx.hi, dx.hi), dx.lo, sum_lo);
    sum_lo = detail::fmaf_rn(detail::fadd_rn(dy.hi, dy.hi), dy.lo, sum_lo);
    sum_lo = detail::fmaf_rn(detail::fadd_rn(dz.hi, dz.hi), dz.lo, sum_lo);
    sum_lo = detail::fmaf_rn(dx.lo, dx.lo, sum_lo);
    sum_lo = detail::fmaf_rn(dy.lo, dy.lo, sum_lo);
    sum_lo = detail::fmaf_rn(dz.lo, dz.lo, sum_lo);
    const float sum_hi = t.hi;

    // Renormalize the (sum_hi, sum_lo) pair before sqrt. This is required for
    // the three-way fp32 cancellation corner: if all three dx.hi/dy.hi/dz.hi
    // round to zero (antenna sub-ULP-close to the pixel in every dimension)
    // but the lo*lo terms accumulated above carry a positive contribution,
    // sum_hi is zero while sum_lo holds the true squared distance. Without
    // this renormalize, rsqrt(sum_hi) returns +inf and the subsequent
    // fmul(sum_hi, xn) gives 0 * inf = NaN, poisoning the bin.
    //
    // fast_two_sum's |a| >= |b| precondition is satisfied for ordinary SAR
    // (sum_hi dominates by 6+ orders of magnitude); when sum_hi = 0 the
    // precondition is violated but the addition 0 + sum_lo is exact, so the
    // returned (s, err) still represent the value correctly.
    const fltflt sum_sq = fltflt_fast_two_sum(sum_hi, sum_lo);

    // Stage 3: Newton-Raphson sqrt step without the trailing
    // fast_two_sum(yn, correction) that fltflt_sqrt_fast applies. We drop
    // the (a.hi == 0) guard from sqrt_fast because the renormalize above
    // already canonicalized the pair; sum_sq.hi == 0 now only occurs when
    // the true squared distance is exactly zero (antenna coincident with
    // the pixel at fp64 precision), which is not a physical SAR geometry.
    const float xn = detail::fltflt_rsqrt(sum_sq.hi);
    const float yn = detail::fmul_rn(sum_sq.hi, xn);  // ~ sqrt(sum_sq.hi)
    const float residual = detail::fadd_rn(
        detail::fmaf_rn(-yn, yn, sum_sq.hi), sum_sq.lo);
    const float r_lo = detail::fmul_rn(detail::fmul_rn(xn, 0.5f), residual);

    // Stage 4: bin = R * dr_inv + mcp_partial, where R = {yn, r_lo} is the
    // non-canonical sqrt result. fma_approx tolerates this because its only
    // dropped term is a.lo*b.lo = r_lo * dr_inv.lo, which is O(ULP^2 * R)
    // regardless of whether (yn, r_lo) has been canonicalized.
    return fltflt_fma_approx(fltflt{yn, r_lo}, dr_inv, mcp_partial);
}

template <typename PlatPosAccessor, SarBpComputeType ComputeType, typename strict_compute_t, typename loose_compute_t>
__device__ inline strict_compute_t ComputeRangeToPixel(const PlatPosAccessor &ant_pos, const index_t pulse_idx, const loose_compute_t px, const loose_compute_t py, const loose_compute_t z0) {
    using plat_pos_t = typename PlatPosAccessor::value_type;
    constexpr int Rank = PlatPosAccessor::Rank;

    strict_compute_t dx, dy, dz;
    static_assert((Rank == 1 && (cuda::std::is_same_v<plat_pos_t, double3> || cuda::std::is_same_v<plat_pos_t, double4> ||
        cuda::std::is_same_v<plat_pos_t, float3> || cuda::std::is_same_v<plat_pos_t, float4>)) || Rank == 2,
        "ComputeRangeToPixel: plat_pos_t must be a 1D tensor of 3D or 4D vectorized type or a 2D tensor with size 3 (x,y,z) in the second dimension");

    if constexpr (Rank == 1) {
        const plat_pos_t ant_pos_p = ant_pos(pulse_idx);
        dx = static_cast<strict_compute_t>(px) - static_cast<strict_compute_t>(ant_pos_p.x);
        dy = static_cast<strict_compute_t>(py) - static_cast<strict_compute_t>(ant_pos_p.y);
        dz = static_cast<strict_compute_t>(z0) - static_cast<strict_compute_t>(ant_pos_p.z);
    } else {
        dx = static_cast<strict_compute_t>(px) - static_cast<strict_compute_t>(ant_pos(pulse_idx, 0));
        dy = static_cast<strict_compute_t>(py) - static_cast<strict_compute_t>(ant_pos(pulse_idx, 1));
        dz = static_cast<strict_compute_t>(z0) - static_cast<strict_compute_t>(ant_pos(pulse_idx, 2));
    }

    if constexpr (ComputeType == SarBpComputeType::Float) {
        return ::sqrtf(dx*dx + dy*dy + dz*dz);
    } else {
        if constexpr (ComputeType == SarBpComputeType::Mixed) {
#if __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
            return ::sqrt(dx*dx + dy*dy + dz*dz);
#else
            // Only use the Newton-Raphson approach on systems with reduced FP64 throughput
            return NewtonRaphsonSqrt(dx*dx + dy*dy + dz*dz);
#endif
        } else {
            return ::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
}

template <typename ComputeType, typename StorageType>
__global__ void SarBpFillPhaseLUT(cuda::std::complex<StorageType> *phase_lut, ComputeType ref_freq, ComputeType dr, index_t num_range_bins)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_range_bins) return;
    constexpr ComputeType four_pi_over_c = static_cast<ComputeType>(4.0 * M_PI / SPEED_OF_LIGHT);
    const ComputeType range_bin_start = static_cast<ComputeType>(tid - 0.5 * (num_range_bins-1)) * dr;
    const ComputeType phase = four_pi_over_c * ref_freq * range_bin_start;
    if constexpr (cuda::std::is_same_v<ComputeType, float>) {
        ComputeType sinx, cosx;
        ::sincosf(phase, &sinx, &cosx);
        phase_lut[tid] = cuda::std::complex<StorageType>(
            static_cast<StorageType>(cosx), static_cast<StorageType>(sinx));
    } else {
        ComputeType sinx, cosx;
        ::sincos(phase, &sinx, &cosx);
        phase_lut[tid] = cuda::std::complex<StorageType>(
            static_cast<StorageType>(cosx), static_cast<StorageType>(sinx));
    }
}

// Template alias for the strict compute parameter type used in SarBp kernel
template <SarBpComputeType ComputeType>
using strict_compute_param_t = typename std::conditional<ComputeType == SarBpComputeType::Double || ComputeType == SarBpComputeType::Mixed || ComputeType == SarBpComputeType::FloatFloat, double, float>::type;

template <SarBpComputeType ComputeType>
using strict_or_ff_compute_param_t = typename std::conditional<ComputeType == SarBpComputeType::FloatFloat, fltflt, strict_compute_param_t<ComputeType>>::type;

template <SarBpComputeType ComputeType>
using loose_compute_param_t = typename std::conditional<ComputeType == SarBpComputeType::Double, double, float>::type;

// Shared-memory layout is parameterized on whether the kernel runs the
// cooperative preamble (see UseSharedPreamble in SarBp). The preamble loads
// per-pulse antenna positions and pre-computes mcp_partial = bin_offset
// - rtm*dr_inv once per pulse block, amortizing the global FP64 loads and
// FP64->compute-type casts across all threads.
template <SarBpComputeType ComputeType, bool PreambleEnabled>
struct SarBpSharedMemory {};

template <bool PreambleEnabled>
struct SarBpSharedMemory<SarBpComputeType::FloatFloat, PreambleEnabled> {
    fltflt ant_pos[PULSE_BLOCK_SIZE][4];
};

template <>
struct SarBpSharedMemory<SarBpComputeType::Float, true> {
    // Slots: [0..2] = ant_pos.x/y/z as float, [3] = bin_offset - rtm*dr_inv
    float ant_pos[PULSE_BLOCK_SIZE][4];
};

template <>
struct SarBpSharedMemory<SarBpComputeType::Mixed, true> {
    // Same layout as Float, but with double elements so strict_compute_t (double)
    // arithmetic in the inner loop reads directly without up-casting.
    double ant_pos[PULSE_BLOCK_SIZE][4];
};

template <SarBpComputeType ComputeType, typename OutImageType, typename InitialImageType, typename RangeProfilesType, typename PlatPosType, typename VoxLocType, typename RangeToMcpType, bool PhaseLUT, bool IsUnitStride>
__launch_bounds__(16*16)
__global__ void SarBp(OutImageType output, const InitialImageType initial_image, const __grid_constant__ RangeProfilesType range_profiles, const __grid_constant__ PlatPosType platform_positions, const __grid_constant__ VoxLocType voxel_locations, const __grid_constant__ RangeToMcpType range_to_mcp,
                      strict_or_ff_compute_param_t<ComputeType> dr_inv,
                      strict_compute_param_t<ComputeType> phase_correction_partial,
                      cuda::std::complex<loose_compute_param_t<ComputeType>> *phase_lut)
{
    static_assert(OutImageType::Rank() == 2, "Output image must be a 2D tensor");
    static_assert(InitialImageType::Rank() == 2, "Initial image must be a 2D tensor");
    static_assert(RangeProfilesType::Rank() == 2, "Range profiles must be a 2D tensor");
    static_assert(PlatPosType::Rank() == 1 || PlatPosType::Rank() == 2, "Platform positions must be a 1D or 2D tensor");
    static_assert(VoxLocType::Rank() == 2, "Voxel locations must be a 2D tensor");
    static_assert(is_complex_v<typename OutImageType::value_type>, "Output image must be complex");
    static_assert(is_complex_v<typename InitialImageType::value_type>, "Initial image must be complex");
    static_assert(is_complex_v<typename RangeProfilesType::value_type>, "Range profiles must be complex");

    static_assert(is_matx_op<RangeToMcpType>(),
        "RangeToMcpType must be a MatX operator");
    static_assert(RangeToMcpType::Rank() == 0 || RangeToMcpType::Rank() == 1,
        "RangeToMcpType must be a rank-0 or rank-1 operator");
    static_assert(cuda::std::is_same_v<typename RangeToMcpType::value_type, float> ||
                  cuda::std::is_same_v<typename RangeToMcpType::value_type, double> ||
                  cuda::std::is_same_v<typename RangeToMcpType::value_type, fltflt>,
        "RangeToMcpType::value_type must be float, double, or fltflt");

    using initial_image_t = typename InitialImageType::value_type;
    using image_t = typename OutImageType::value_type;
    using range_profiles_t = typename RangeProfilesType::value_type;
    using plat_pos_t = typename PlatPosType::value_type;
    using voxel_loc_t = typename VoxLocType::value_type;
    using compute_t = typename std::conditional<ComputeType == SarBpComputeType::Double, double, float>::type;
    using strict_compute_t = typename std::conditional<ComputeType == SarBpComputeType::Double || ComputeType == SarBpComputeType::Mixed, double, float>::type;
    using strict_or_ff_compute_t = typename std::conditional<ComputeType == SarBpComputeType::FloatFloat, fltflt, strict_compute_t>::type;
    using strict_complex_compute_t = cuda::std::complex<strict_compute_t>;
    using loose_compute_t = typename std::conditional<ComputeType == SarBpComputeType::Double, double, float>::type;
    using loose_complex_compute_t = cuda::std::complex<loose_compute_t>;

    const index_t image_height = output.Size(0);
    const index_t image_width = output.Size(1);
    const index_t ix = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const index_t iy = static_cast<index_t>(blockIdx.y * blockDim.y + threadIdx.y);

    // IsUnitStride=true implies the hot inputs must be tensor views (they
    // use .Data()). The transform-level dispatch enforces this; the
    // static_asserts below make the contract explicit so a misuse fails at
    // compile time rather than via a missing-method error deep in the kernel.
    if constexpr (IsUnitStride) {
        static_assert(is_tensor_view_v<RangeProfilesType>,
                      "SarBp IsUnitStride path requires range_profiles to be a tensor view");
        static_assert(is_tensor_view_v<PlatPosType>,
                      "SarBp IsUnitStride path requires platform_positions to be a tensor view");
    }

    // The cooperative shared-memory preamble is required for FloatFloat (its
    // inner loop reads directly from shared memory) and enabled for Float and
    // Mixed when PhaseLUT is on.  With PhaseLUT=false the inner loop still
    // needs diffR = dist - rtm for sincos, which the preamble does not
    // preserve, so fall back to direct global reads in that case.
    constexpr bool UseSharedPreamble =
        (ComputeType == SarBpComputeType::FloatFloat) ||
        ((ComputeType == SarBpComputeType::Float ||
          ComputeType == SarBpComputeType::Mixed) && PhaseLUT);

    __shared__ SarBpSharedMemory<ComputeType, UseSharedPreamble> sh_mem;

    const bool is_valid = ix < image_width && iy < image_height;
    if constexpr (!UseSharedPreamble) {
        // When the preamble is enabled, keep all threads active so they can
        // participate in the CTA-wide cooperative loads of antenna positions.
        if (! is_valid) return;
    }

    const index_t num_pulses = range_profiles.Size(0);
    const int32_t num_range_bins = static_cast<int32_t>(range_profiles.Size(1));

    static_assert(cuda::std::is_same_v<voxel_loc_t, double3> || cuda::std::is_same_v<voxel_loc_t, double4> ||
        cuda::std::is_same_v<voxel_loc_t, float3> || cuda::std::is_same_v<voxel_loc_t, float4>, "SarBp: VoxLocType must represent a 2D operator of type double3, double4, float3, or float4");
    // voxel_locations is read once per thread (outside the pulse loop), so it
    // is deliberately not covered by the IsUnitStride fast-path logic. The
    // one-shot cost of its operator() is negligible and its layout is free to
    // be anything the caller wants without blocking fast-path eligibility.
    const voxel_loc_t voxel_loc = is_valid ? voxel_locations(iy, ix) : voxel_loc_t{};
    const loose_compute_t py = static_cast<loose_compute_t>(voxel_loc.y);
    const loose_compute_t px = static_cast<loose_compute_t>(voxel_loc.x);
    const loose_compute_t pz = static_cast<loose_compute_t>(voxel_loc.z);

    // TensorAccessor wraps each hot tensor and picks the fast pointer path
    // when IsUnitStride && is_tensor_view_v<Op>, otherwise forwards to
    // operator(). Binding the tensor's Data() and Stride(0) into the
    // accessor's members forces the compiler to materialize the grid-constant
    // LDCs once at kernel entry rather than reloading inside the pulse loop.
    detail::TensorAccessor<RangeProfilesType, IsUnitStride> rp(range_profiles);
    detail::TensorAccessor<PlatPosType, IsUnitStride> pp(platform_positions);

    // range_to_mcp is required to be a rank-0 or rank-1 MatX operator; the
    // TensorAccessor picks the fast pointer path on IsUnitStride for tensor
    // views or forwards to operator() for computed ops (e.g. ConstVal).
    const detail::TensorAccessor<RangeToMcpType, IsUnitStride> rtm_acc(range_to_mcp);

    const auto r_to_mcp = [rtm_acc](index_t p) -> auto {
        if constexpr (RangeToMcpType::Rank() == 0) {
            return rtm_acc();
        } else {
            return rtm_acc(p);
        }
    };

    [[maybe_unused]] const loose_compute_t phase_correction_partial_loose = static_cast<loose_compute_t>(phase_correction_partial);
    const auto get_reference_phase = [&phase_lut, &phase_correction_partial, &phase_correction_partial_loose](strict_or_ff_compute_t diffR, int32_t bin_floor_int, loose_compute_t w) -> loose_complex_compute_t {
        if constexpr (PhaseLUT) {
            const loose_complex_compute_t base_phase = phase_lut[bin_floor_int];
            loose_compute_t incr_sinx, incr_cosx;
            if constexpr (cuda::std::is_same_v<loose_compute_t, double>) {
                ::sincos(phase_correction_partial_loose * w, &incr_sinx, &incr_cosx);
            } else {
                __sincosf(phase_correction_partial_loose * w, &incr_sinx, &incr_cosx);
            }

            return loose_complex_compute_t{
                base_phase.real() * incr_cosx - base_phase.imag() * incr_sinx,
                base_phase.real() * incr_sinx + base_phase.imag() * incr_cosx
            };
        } else {
            // With PhaseLUT == false, strict_or_ff_compute_t is either float or double, so we can use sincos[f] directly.
            strict_or_ff_compute_t sinx, cosx;
            if constexpr (cuda::std::is_same_v<strict_compute_t, double>) {
                ::sincos(phase_correction_partial * diffR, &sinx, &cosx);
            } else {
                ::sincosf(phase_correction_partial * diffR, &sinx, &cosx);
            }
            return loose_complex_compute_t{
                static_cast<loose_compute_t>(cosx), static_cast<loose_compute_t>(sinx)
            };
        }
    };

    // Explicitly use FMA instructions for pixel accumulations
    const auto accumulate_contribution =
        [](loose_complex_compute_t accum_in, loose_complex_compute_t sample, loose_complex_compute_t ref_phase) -> loose_complex_compute_t {
            const loose_compute_t sr = sample.real();
            const loose_compute_t si = sample.imag();
            const loose_compute_t pr = ref_phase.real();
            const loose_compute_t pi = ref_phase.imag();
            if constexpr (cuda::std::is_same_v<loose_compute_t, float>) {
                return loose_complex_compute_t{
                    __fmaf_rn(sr, pr, __fmaf_rn(-si, pi, accum_in.real())),
                    __fmaf_rn(sr, pi, __fmaf_rn( si, pr, accum_in.imag()))
                };
            } else {
                return loose_complex_compute_t{
                    fma(sr, pr, fma(-si, pi, accum_in.real())),
                    fma(sr, pi, fma( si, pr, accum_in.imag()))
                };
            }
        };

    [[maybe_unused]] const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    loose_complex_compute_t accum{};
    const loose_compute_t bin_offset = static_cast<loose_compute_t>(0.5) * static_cast<loose_compute_t>(num_range_bins-1);

    const int num_pulse_blocks = static_cast<int>(
        (num_pulses + static_cast<index_t>(PULSE_BLOCK_SIZE) - 1) / static_cast<index_t>(PULSE_BLOCK_SIZE));
    for (int block = 0; block < num_pulse_blocks; ++block) {
        const index_t pulse_base = static_cast<index_t>(block) * static_cast<index_t>(PULSE_BLOCK_SIZE);
        const index_t pulses_remaining = num_pulses - pulse_base;
        const index_t num_pulses_in_block =
            (pulses_remaining < static_cast<index_t>(PULSE_BLOCK_SIZE)) ? pulses_remaining : static_cast<index_t>(PULSE_BLOCK_SIZE);
        if constexpr (UseSharedPreamble) {
            __syncthreads();
            for (index_t ip = tid; ip < num_pulses_in_block; ip += blockDim.x * blockDim.y) {
                const index_t p = pulse_base + ip;
                // Accessor does the IsUnitStride / rank dispatch internally,
                // so we just ask for (apx, apy, apz) uniformly.
                auto load_xyz = [&]() {
                    if constexpr (PlatPosType::Rank() == 1) {
                        const plat_pos_t ap = pp(p);
                        return cuda::std::make_tuple(ap.x, ap.y, ap.z);
                    } else {
                        return cuda::std::make_tuple(pp(p, 0), pp(p, 1), pp(p, 2));
                    }
                };
                const auto xyz = load_xyz();

                if constexpr (ComputeType == SarBpComputeType::FloatFloat) {
                    sh_mem.ant_pos[ip][0] = static_cast<fltflt>(cuda::std::get<0>(xyz));
                    sh_mem.ant_pos[ip][1] = static_cast<fltflt>(cuda::std::get<1>(xyz));
                    sh_mem.ant_pos[ip][2] = static_cast<fltflt>(cuda::std::get<2>(xyz));
                    const fltflt rtm = static_cast<fltflt>(r_to_mcp(p));
                    const fltflt neg_rtm = fltflt{-rtm.hi, -rtm.lo};
                    sh_mem.ant_pos[ip][3] = fltflt_fma_approx(neg_rtm, dr_inv, bin_offset);
                } else {
                    // Float / Mixed: cast inputs to strict_compute_t (float / double)
                    // once per pulse here, instead of once per pulse per pixel.
                    sh_mem.ant_pos[ip][0] = static_cast<strict_compute_t>(cuda::std::get<0>(xyz));
                    sh_mem.ant_pos[ip][1] = static_cast<strict_compute_t>(cuda::std::get<1>(xyz));
                    sh_mem.ant_pos[ip][2] = static_cast<strict_compute_t>(cuda::std::get<2>(xyz));
                    const strict_compute_t rtm = static_cast<strict_compute_t>(r_to_mcp(p));
                    if constexpr (cuda::std::is_same_v<strict_compute_t, double>) {
                        sh_mem.ant_pos[ip][3] = ::fma(-rtm, dr_inv, static_cast<double>(bin_offset));
                    } else {
                        sh_mem.ant_pos[ip][3] = ::fmaf(-rtm, dr_inv, bin_offset);
                    }
                }
            }
            __syncthreads();
            if (! is_valid) {
                continue;
            }
        }
        #pragma unroll 4
        for (index_t ip = 0; ip < num_pulses_in_block; ++ip) {
            const index_t p = pulse_base + ip;
            strict_or_ff_compute_t diffR;
            loose_compute_t w;
            int32_t bin_floor_int;
            if constexpr (ComputeType == SarBpComputeType::FloatFloat) {
                // ComputeBinToPixelFloatFloat fuses the coordinate-difference,
                // squared-norm, sqrt, and (R-mcp)*dr_inv + bin_offset chain into
                // one helper that never materializes a canonical R. The
                // shared-memory slot sh_mem.ant_pos[ip][3] holds
                // -mcp*dr_inv + bin_offset, precomputed in the pulse-block
                // preamble.
                const fltflt bin = ComputeBinToPixelFloatFloat(
                    sh_mem.ant_pos[ip][0], sh_mem.ant_pos[ip][1], sh_mem.ant_pos[ip][2],
                    px, py, pz, dr_inv, sh_mem.ant_pos[ip][3]);
                diffR = bin;  // unused below (FloatFloat requires PhaseLUT); assign to silence maybe-uninitialized warning
                float floor_hi = ::floorf(bin.hi);
                float frac = (bin.hi - floor_hi) + bin.lo;
                // bin.lo may push bin over a boundary, in which case floor and frac are incorrect.
                // Compute an adjustment based on whether or not the fractional part is outside (0.0, 1.0).
                const float adjust = ::floorf(frac);  // -1, 0, or 1
                bin_floor_int = static_cast<int32_t>(floor_hi + adjust);
                w = frac - adjust;
            } else if constexpr (UseSharedPreamble) {
                // Float / Mixed with shared-mem preamble: antenna position and
                // mcp_partial have been pre-loaded / pre-computed in shared
                // memory, so the inner loop is pure strict_compute_t arithmetic.
                const strict_compute_t apx = sh_mem.ant_pos[ip][0];
                const strict_compute_t apy = sh_mem.ant_pos[ip][1];
                const strict_compute_t apz = sh_mem.ant_pos[ip][2];
                const strict_compute_t mcp_partial = sh_mem.ant_pos[ip][3];
                const strict_compute_t dx = static_cast<strict_compute_t>(px) - apx;
                const strict_compute_t dy = static_cast<strict_compute_t>(py) - apy;
                const strict_compute_t dz = static_cast<strict_compute_t>(pz) - apz;
                strict_compute_t dist;
                if constexpr (ComputeType == SarBpComputeType::Float) {
                    dist = ::sqrtf(dx*dx + dy*dy + dz*dz);
                } else {
#if __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
                    dist = ::sqrt(dx*dx + dy*dy + dz*dz);
#else
                    dist = NewtonRaphsonSqrt(dx*dx + dy*dy + dz*dz);
#endif
                }
                // bin = (dist - rtm)*dr_inv + bin_offset = dist*dr_inv + mcp_partial
                const strict_compute_t bin = dist * dr_inv + mcp_partial;
                strict_compute_t bin_floor;
                if constexpr (cuda::std::is_same_v<strict_compute_t, double>) {
                    bin_floor = ::floor(bin);
                } else {
                    bin_floor = ::floorf(bin);
                }
                w = static_cast<loose_compute_t>(bin - bin_floor);
                bin_floor_int = static_cast<int32_t>(bin_floor);
                // diffR is unused when PhaseLUT=true (required for this branch);
                // assign to avoid any maybe-uninitialized warning downstream.
                diffR = dist;
            } else {
                diffR = ComputeRangeToPixel<decltype(pp), ComputeType, strict_compute_t, loose_compute_t>(
                    pp, p, px, py, pz) - static_cast<strict_compute_t>(r_to_mcp(p));
                const strict_compute_t bin = diffR * dr_inv + bin_offset;
                strict_compute_t bin_floor;
                if constexpr (cuda::std::is_same_v<strict_compute_t, double>) {
                    bin_floor = ::floor(bin);
                } else {
                    bin_floor = ::floorf(bin);
                }
                w = static_cast<loose_compute_t>(bin - bin_floor);
                bin_floor_int = static_cast<int32_t>(bin_floor);
            }
            if (bin_floor_int >= 0 && bin_floor_int < num_range_bins-1) {
                // rp accessor picks the fast pointer path on IsUnitStride or
                // falls through to operator().
                const range_profiles_t sample_lo = rp(p, bin_floor_int);
                const range_profiles_t sample_hi = rp(p, bin_floor_int + 1);

                const loose_complex_compute_t sample = [&sample_lo, &sample_hi, &w]() -> loose_complex_compute_t {
                    const loose_complex_compute_t loose_sample_lo = static_cast<loose_complex_compute_t>(sample_lo);
                    const loose_complex_compute_t loose_sample_hi = static_cast<loose_complex_compute_t>(sample_hi);
                    if constexpr (cuda::std::is_same_v<loose_compute_t, float>) {
                        return loose_complex_compute_t{
                            __fmaf_rn(w, loose_sample_hi.real(), __fmaf_rn(-w, loose_sample_lo.real(), loose_sample_lo.real())),
                            __fmaf_rn(w, loose_sample_hi.imag(), __fmaf_rn(-w, loose_sample_lo.imag(), loose_sample_lo.imag()))
                        };
                    } else {
                        return loose_complex_compute_t{
                            fma(w, loose_sample_hi.real(), fma(-w, loose_sample_lo.real(), loose_sample_lo.real())),
                            fma(w, loose_sample_hi.imag(), fma(-w, loose_sample_lo.imag(), loose_sample_lo.imag()))
                        };
                    }
                }();

                // For FloatFloat mode, diffR has been set to the distance to the pixel rather than the differential range to the MCP.
                // However, FloatFloat mode currently requires PhaseLUT optimization due to missing fltflt sin/cos implementations,
                // so diffR will not actually be used in get_reference_phase() below.
                static_assert(ComputeType != SarBpComputeType::FloatFloat || PhaseLUT == true, "SarBp: FloatFloat compute type requires PhaseLUT optimization");
                const loose_complex_compute_t ref_phase = get_reference_phase(diffR, bin_floor_int, w);

                accum = accumulate_contribution(accum, sample, ref_phase);
            }
        } // pulse
    } // pulse block

    if (is_valid) {
        // initial_image and output are each touched exactly once per thread,
        // so they are intentionally not included in the IsUnitStride fast-path
        // check in the transform layer. Direct operator() calls keep the
        // transform's fast-path eligibility as permissive as possible:
        // non-unit-stride or computed-op initial_image (e.g. ConstVal zeros)
        // and non-unit-stride output layouts do not disqualify the fast path
        // for the hot tensors (range_profiles, platform_positions, rtm).
        const initial_image_t initial_image_voxel = initial_image.operator()(iy, ix);
        using out_value_t = typename OutImageType::value_type;
        using out_scalar_t = typename out_value_t::value_type;

        output(iy, ix) = out_value_t{
            static_cast<out_scalar_t>(initial_image_voxel.real() + accum.real()),
            static_cast<out_scalar_t>(initial_image_voxel.imag() + accum.imag())
        };
    }
}

#endif // __CUDACC__

}; // namespace matx
