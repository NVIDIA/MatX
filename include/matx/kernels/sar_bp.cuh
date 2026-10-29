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
#include <cuda/std/numbers>
#include <iomanip>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "matx/core/constants.h"
#include "matx/core/utils.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"
#include "matx/kernels/fltflt.h"
#include "matx/kernels/tensor_accessor.h"

#define PULSE_BLOCK_SIZE 256

namespace matx {

// Backward-compatibility alias; remove when sar_bp is promoted from experimental.
[[deprecated("Use matx::constants::speed_of_light instead")]]
inline constexpr double SPEED_OF_LIGHT = matx::constants::speed_of_light;

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

// The base range bin index and interpolation weight. These are bundled together
// so that they can be returned from a single function call.
template <typename weight_t>
struct SarBpBinAndWeight {
    weight_t w;
    int32_t bin_floor_int;
};

template <typename PlatPosAccessor, SarBpComputeType ComputeType, typename strict_compute_t, typename loose_compute_t, typename IdxT>
__device__ inline strict_compute_t ComputeRangeToPixel(const PlatPosAccessor &ant_pos, const IdxT pulse_idx, const loose_compute_t px, const loose_compute_t py, const loose_compute_t pz) {
    using plat_pos_t = typename PlatPosAccessor::value_type;
    constexpr int Rank = PlatPosAccessor::Rank;
    constexpr bool IsFloat = ComputeType == SarBpComputeType::Float;
    constexpr bool IsMixed = ComputeType == SarBpComputeType::Mixed;

    strict_compute_t dx, dy, dz;
    static_assert((Rank == 1 && (cuda::std::is_same_v<plat_pos_t, double3> || cuda::std::is_same_v<plat_pos_t, double4> ||
        cuda::std::is_same_v<plat_pos_t, float3> || cuda::std::is_same_v<plat_pos_t, float4>)) || Rank == 2,
        "ComputeRangeToPixel: plat_pos_t must be a 1D tensor of 3D or 4D vectorized type or a 2D tensor with size 3 (x,y,z) in the second dimension");

    // In the PixelZIsZero modes the caller passes pz as a compile-time constant
    // 0, so dz = pz - ant_pos.z folds to -ant_pos.z.
    if constexpr (Rank == 1) {
        const plat_pos_t ant_pos_p = ant_pos(pulse_idx);
        dx = static_cast<strict_compute_t>(px) - static_cast<strict_compute_t>(ant_pos_p.x);
        dy = static_cast<strict_compute_t>(py) - static_cast<strict_compute_t>(ant_pos_p.y);
        dz = static_cast<strict_compute_t>(pz) - static_cast<strict_compute_t>(ant_pos_p.z);
    } else {
        dx = static_cast<strict_compute_t>(px) - static_cast<strict_compute_t>(ant_pos(pulse_idx, 0));
        dy = static_cast<strict_compute_t>(py) - static_cast<strict_compute_t>(ant_pos(pulse_idx, 1));
        dz = static_cast<strict_compute_t>(pz) - static_cast<strict_compute_t>(ant_pos(pulse_idx, 2));
    }

    if constexpr (IsFloat) {
        return ::sqrtf(dx*dx + dy*dy + dz*dz);
    } else {
        if constexpr (IsMixed) {
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
    constexpr ComputeType four_pi_over_c =
        static_cast<ComputeType>(4.0 * cuda::std::numbers::pi /
                                 matx::constants::speed_of_light);
    // fftshift places zero differential range at floor(N/2). For even N,
    // this is index N/2, the higher-indexed of the two bins adjacent to
    // the array's geometric midpoint (N-1)/2.
    const index_t bin_offset = num_range_bins / 2;
    const ComputeType range_bin_start =
        static_cast<ComputeType>(static_cast<index_t>(tid) - bin_offset) * dr;
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
using strict_compute_param_t = typename std::conditional<
    ComputeType == SarBpComputeType::Double ||
    ComputeType == SarBpComputeType::Mixed ||
    ComputeType == SarBpComputeType::FloatFloat ||
    ComputeType == SarBpComputeType::TaylorFast,
    double, float>::type;

template <SarBpComputeType ComputeType>
using strict_or_ff_compute_param_t = typename std::conditional<
    ComputeType == SarBpComputeType::FloatFloat,
    fltflt, strict_compute_param_t<ComputeType>>::type;

// TaylorFast uses dr_inv in double precision for the per-pulse-block reference bin
// (FillPulseBlockCache) and in single precision in the inner pulse loop. Both are passed
// from the host so that the kernel performs no double->float conversions that may not
// hoist out of the inner loop in all cases.
struct SarBpTaylorFastDrInv {
    double f64;
    float f32;
};

template <SarBpComputeType ComputeType>
using dr_inv_param_t = typename std::conditional<
    ComputeType == SarBpComputeType::TaylorFast,
    SarBpTaylorFastDrInv, strict_or_ff_compute_param_t<ComputeType>>::type;

template <SarBpComputeType ComputeType>
using sarbp_strict_compute_t = typename std::conditional<
    ComputeType == SarBpComputeType::Double ||
    ComputeType == SarBpComputeType::Mixed ||
    ComputeType == SarBpComputeType::TaylorFast,
    double, float>::type;

template <SarBpComputeType ComputeType>
using loose_compute_param_t = typename std::conditional<
    ComputeType == SarBpComputeType::Double,
    double, float>::type;

// TaylorFast synthesizes its phase (see below) from the phase-correction factor
// alpha (radians per range bin). It uses alpha in single precision for the
// fractional-bin term in the inner pulse loop, and the fractional revolutions per
// bin, frac(alpha / 2*pi), in fixed point: Q0.64 to form the per-pulse reference
// ticks (Q0.32) and Q0.32 for the per-bin increment Q. As with dr_inv, all three
// are formed on the host, here by Make, so the kernel performs no conversions or
// FP64 rounding to obtain them.
struct SarBpTaylorFastPhaseCorrection {
    uint64_t Q64;     // fractional revolutions per bin, Q0.64
    float    alpha32; // radians per bin
    uint32_t Q;       // fractional revolutions per bin, Q0.32

    static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ SarBpTaylorFastPhaseCorrection Make(double alpha)
    {
        constexpr double inv_two_pi = 0.5 * cuda::std::numbers::inv_pi_v<double>;
        const double rho = alpha * inv_two_pi;
        // frac(rho) reaches 1 only by rounding when rho is a tiny negative value,
        // and is NaN for a non-finite alpha. Map both to 0 (a whole revolution, or
        // no increment) so that the conversion stays in range. Scaling by 2^64 is
        // exact, and the conversion truncates only when frac(rho) < 2^-12, by less
        // than 2^-64 revolutions.
        const double frac = rho - ::floor(rho);
        const uint64_t q64 = (frac < 1.0) ? static_cast<uint64_t>(frac * 0x1p64) : 0;
        // Round Q0.64 to Q0.32. A sum that carries out of 64 bits is a rounded
        // 2^32, which wraps to 0, the phase-equivalent increment. We add half
        // of one Q0.32 step (0x80000000ull) to properly round the last bit of q32.
        const uint32_t q32 = static_cast<uint32_t>((q64 + 0x80000000ull) >> 32);
        return SarBpTaylorFastPhaseCorrection{q64, static_cast<float>(alpha), q32};
    }
};

// The phase-correction factor is consumed in loose precision on the PhaseLUT path
// (multiplied by the fractional bin weight) and in strict precision on the direct
// path (multiplied by the full range delta). Passing the PhaseLUT value already in
// loose precision avoids in-kernel conversions that may not hoist in all cases.
template <SarBpComputeType ComputeType, bool PhaseLUT>
using phase_correction_param_t = typename std::conditional<
    ComputeType == SarBpComputeType::TaylorFast, SarBpTaylorFastPhaseCorrection,
    typename std::conditional<PhaseLUT, loose_compute_param_t<ComputeType>,
                              strict_compute_param_t<ComputeType>>::type>::type;

// ---------------------------------------------------------------------------
// TaylorFast reference-relative phase synthesis.
//
// The TaylorFast reference phase is phi = alpha*(b - bin_offset), alpha being
// the phase advance per range bin. TaylorFast already splits the bin as
// b = J + n + w, where J is the per-pulse integer reference bin shared by the
// CTA, n the integer local offset and w the interpolation weight. Then
//
//     phi = alpha*(J - bin_offset) + alpha*n + alpha*w
//
// and each component may be reduced modulo 2*pi independently. The first two
// are combined in Q0.32 binary angular measure (BAM), where one revolution is
// 2^32 ticks and uint32 wraparound performs the reduction modulo one
// revolution exactly:
//
//     ticks = T_ref + Q*n  (mod 2^32),
//
// with Q the fractional revolutions per bin in Q0.32 and T_ref the reference
// angle alpha*(J - bin_offset) in ticks, formed once per pulse per CTA from the
// same fraction in Q0.64 by the same wraparound argument. The fractional-bin
// term alpha*w is added in float. This is an alternative to the phase lookup
// table, which gathered e^{i*alpha*(J+n-bin_offset)} from a pre-computed table.
// Memory access is a bottleneck for TaylorFast with a phase LUT and phase
// synthesis partially mitigates that memory bottleneck by removing the LUT loads.
// ---------------------------------------------------------------------------

// The per-pulse reference angle alpha*(J - bin_offset) in Q0.32 ticks, formed
// once per pulse per CTA and amortized over the CTA's pixels. k = J - bin_offset
// is an integer, so only the fractional revolutions per bin matter: the angle is
// the upper word of the low 64 bits of Q64*k, which wrap modulo one revolution
// exactly for either sign. The upper word is truncated rather than rounded,
// costing at most one tick (1.5e-9 rad).
__MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ uint32_t SarBpRefPhaseTicks(
    const SarBpTaylorFastPhaseCorrection &c, int32_t ref_bin_int, int32_t bin_offset)
{
    // k modulo 2^32. It is exact as a signed value whenever the CTA has any
    // in-range sample, as |k| is then far below 2^31.
    const uint32_t k = static_cast<uint32_t>(ref_bin_int) - static_cast<uint32_t>(bin_offset);
    const uint32_t q_lo = static_cast<uint32_t>(c.Q64);
    const uint32_t q_hi = static_cast<uint32_t>(c.Q64 >> 32);
    // We want bits 63..32 of the product Q64 * k. Higher bits are whole revolutions
    // and can be dropped. Bits 31..0 are fractions of a tick. We decompose the product
    // into 32-bit components due to register spills in some cases when using 64-bit
    // integer arithmetic.
    //
    // With Q64 = q_hi*2^32 + q_lo, k_u the bits of k read as unsigned and
    // s = 1 for negative k (else 0), two's complement gives k = k_u - s*2^32, so
    //
    //   Q64*k = q_lo*k_u + 2^32*(q_hi*k_u) - 2^32*(s*q_lo) - 2^64*(s*q_hi).
    //
    // The last term is whole revolutions and vanishes modulo 2^64. Adding
    // 2^32*X to a 64-bit value adds X to its upper word modulo 2^32, so
    //
    //   upper word = hi32(q_lo*k_u) + q_hi*k_u - s*q_lo  (mod 2^32).
    //
    // Only the low 32 bits of q_hi*k_u matter, as the rest are also whole
    // revolutions. Dropping the lower word of q_lo*k_u is truncation.
    const uint32_t t = static_cast<uint32_t>((static_cast<uint64_t>(q_lo) * k) >> 32) + q_hi * k;
    return t - ((static_cast<int32_t>(k) < 0) ? q_lo : 0u);
}

// Form the phase factor for local integer offset n and weight w from the
// per-pulse reference ticks produced by SarBpRefPhaseTicks.
__device__ __forceinline__ cuda::std::complex<float> SarBpSynthesizePhase(
    const SarBpTaylorFastPhaseCorrection &c, uint32_t ref_ticks, int32_t n, float w)
{
    // 2*pi/2^32 is a power-of-two rescaling of float(pi), so the float constant
    // carries only the rounding of pi itself.
    constexpr float rad_per_tick =
        static_cast<float>(cuda::std::numbers::pi_v<double>) * 0x1p-31f;
    // The uint32 product wraps modulo one revolution for any n, including
    // negative n via the int32 -> uint32 conversion. Reading the tick count as
    // int32 centres it to [-1/2, 1/2) revolutions.
    const int32_t z = static_cast<int32_t>(ref_ticks + c.Q * static_cast<uint32_t>(n));
    float sn, cs;
    __sincosf(__fmaf_rn(static_cast<float>(z), rad_per_tick, c.alpha32 * w), &sn, &cs);
    return cuda::std::complex<float>{cs, sn};
}

// Shared-memory layout for per-pulse-block cached state. Some compute modes
// cache per-pulse antenna data or Taylor reference values once per pulse block,
// amortizing global FP64 loads and FP64->compute-type casts across all threads.
template <SarBpComputeType ComputeType, bool CacheEnabled, bool PixelZIsUniform = false,
          bool TaylorFastAddThirdOrder = false>
struct SarBpPulseBlockCache {};

template <bool CacheEnabled, bool PixelZIsUniform, bool TaylorFastAddThirdOrder>
struct SarBpPulseBlockCache<SarBpComputeType::FloatFloat, CacheEnabled, PixelZIsUniform,
                            TaylorFastAddThirdOrder> {
    fltflt ant_pos[PULSE_BLOCK_SIZE][4];
};

template <bool PixelZIsUniform, bool TaylorFastAddThirdOrder>
struct SarBpPulseBlockCache<SarBpComputeType::Float, true, PixelZIsUniform,
                            TaylorFastAddThirdOrder> {
    // Slots: [0..2] = ant_pos.x/y/z as float, [3] = bin_offset - rtm*dr_inv
    float ant_pos[PULSE_BLOCK_SIZE][4];
};

template <bool PixelZIsUniform, bool TaylorFastAddThirdOrder>
struct SarBpPulseBlockCache<SarBpComputeType::Mixed, true, PixelZIsUniform,
                            TaylorFastAddThirdOrder> {
    // Same layout as Float, but with double elements so strict_compute_t (double)
    // arithmetic in the inner loop reads directly without up-casting.
    double ant_pos[PULSE_BLOCK_SIZE][4];
};

template <bool PixelZIsUniform, bool TaylorFastAddThirdOrder>
struct SarBpTaylorFastSharedMemory;

template <bool TaylorFastAddThirdOrder>
struct SarBpTaylorFastSharedMemory<false, TaylorFastAddThirdOrder> {
    // Keep Variable-Z values consumed by each pulse together. This layout lets
    // ptxas schedule the six vector loads for a four-pulse unrolled group close
    // to their uses while meeting the six-CTA register constraint.
    float4 geometry[PULSE_BLOCK_SIZE]; // ux, uy, uz, half_inv_R
    // Integer bin, fractional-bin bits, and reference phase ticks (w unused).
    // The int4 keeps the reference phase on the geometry's 16-byte stride, so
    // one base address serves both arrays; a separate 4-byte-stride array had
    // its address rebuilt for every sample.
    int4 ref_bin[PULSE_BLOCK_SIZE];
    double ref_x;
    double ref_y;
    double ref_z;
};

template <>
struct SarBpTaylorFastSharedMemory<true, false> {
    // Uniform pixel z does not consume uz, so do not cache it. Store the five
    // consumed values in separate arrays; across the four-way unrolled pulse
    // loop, ptxas combines four adjacent entries from each array into five
    // LDS.128 loads. The reference phase ticks are paired with the integer
    // bin, which costs one more LDS.128 per four pulses; a separate array was
    // loaded with four scalar LDS instead.
    int2 ref_bin_int[PULSE_BLOCK_SIZE]; // integer bin, reference phase ticks
    float  ref_bin_frac[PULSE_BLOCK_SIZE];
    float  ux          [PULSE_BLOCK_SIZE];
    float  uy          [PULSE_BLOCK_SIZE];
    float  half_inv_R  [PULSE_BLOCK_SIZE];
    double ref_x;
    double ref_y;
    double ref_z;
};

template <>
struct SarBpTaylorFastSharedMemory<true, true> {
    // The third-order Uniform-Z path benefits from keeping each pulse's values
    // together so its additional arithmetic does not extend vector live ranges.
    float4 geometry[PULSE_BLOCK_SIZE]; // ux, uy, half_inv_R, ref_bin_frac
    int2 ref_bin[PULSE_BLOCK_SIZE]; // integer bin, reference phase ticks
    double ref_x;
    double ref_y;
    double ref_z;
};

template <bool PixelZIsUniform, bool TaylorFastAddThirdOrder>
struct SarBpPulseBlockCache<SarBpComputeType::TaylorFast, true, PixelZIsUniform,
                            TaylorFastAddThirdOrder> :
    SarBpTaylorFastSharedMemory<PixelZIsUniform, TaylorFastAddThirdOrder> {};

template <SarBpComputeType ComputeType, bool PhaseLUT>
struct SarBpThreadState {};

template <SarBpComputeType ComputeType>
struct SarBpThreadState<ComputeType, false> {
    sarbp_strict_compute_t<ComputeType> range_delta;
};

template <>
struct SarBpThreadState<SarBpComputeType::TaylorFast, true> {
    float ref_dx = 0.0f;
    float ref_dy = 0.0f;
    float ref_dz = 0.0f;
    float ref_dsq = 0.0f;
    float dr_inv_f32 = 0.0f;
};

// ComputeBinWeightToPixelFloatFloat() fuses the FloatFloat path's
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
template <bool UseCachedPz2 = false, bool PixelZIsUniform = false, typename IdxT = index_t>
__device__ inline SarBpBinAndWeight<float> ComputeBinWeightToPixelFloatFloat(
    const SarBpPulseBlockCache<SarBpComputeType::FloatFloat, true, PixelZIsUniform> &sh_mem,
    IdxT ip,
    float px, float py, [[maybe_unused]] float pz,
    fltflt dr_inv)
{
    const fltflt apx = sh_mem.ant_pos[ip][0];
    const fltflt apy = sh_mem.ant_pos[ip][1];
    const fltflt mcp_partial = sh_mem.ant_pos[ip][3];

    // Stage 1: loose dx/dy. dz is only formed here when the per-pulse dz^2 is
    // not cached. We use the general fltflt_two_sum (6 fp32 ops) for each
    // dimension.
    fltflt dx = fltflt_two_sum(px, -apx.hi);
    fltflt dy = fltflt_two_sum(py, -apy.hi);
    dx.lo = detail::fadd_rn(dx.lo, -apx.lo);
    dy.lo = detail::fadd_rn(dy.lo, -apy.lo);

    // dz is formed here (unless the per-pulse dz^2 is cached) so that the
    // non-cached path keeps the original dx/dy/dz -> px2/py2/pz2 ordering and
    // codegen. The uniform-z modes always cache (UseCachedPz2), so this branch
    // is the general per-pixel path.
    [[maybe_unused]] fltflt dz{};
    if constexpr (!UseCachedPz2) {
        const fltflt apz = sh_mem.ant_pos[ip][2];
        dz = fltflt_two_sum(pz, -apz.hi);
        dz.lo = detail::fadd_rn(dz.lo, -apz.lo);
    }

    // Stage 2: squared-norm accumulation. Same body as the canonical
    // fltflt_norm3d() but with explicit dx.lo*dx.lo / dy.lo*dy.lo /
    // dz.lo*dz.lo contributions retained. The canonical helper drops those
    // as O(eps^2 * sum_hi); for our non-canonical inputs (post-cancellation
    // |lo| can approach |hi|), they carry the cancelled dimension's full
    // contribution and must be kept.
    //
    // For the uniform-z modes the entire z contribution -- the dz two-sum, the
    // z two-prod, and the 2*dz.hi*dz.lo / dz.lo^2 low-order terms -- is hoisted
    // into a single fully-compensated fltflt dz^2 (pz2) precomputed once per
    // pulse in FillPulseBlockCache and read from the antenna-z slot here.
    const fltflt px2 = fltflt_two_prod_fma(dx.hi, dx.hi);
    const fltflt py2 = fltflt_two_prod_fma(dy.hi, dy.hi);
    fltflt pz2;
    if constexpr (UseCachedPz2) {
        pz2 = sh_mem.ant_pos[ip][2];
    } else {
        pz2 = fltflt_two_prod_fma(dz.hi, dz.hi);
    }
    const fltflt s = fltflt_two_sum(px2.hi, py2.hi);
    const fltflt t = fltflt_two_sum(s.hi, pz2.hi);

    float sum_lo = detail::fadd_rn(t.lo, s.lo);
    sum_lo = detail::fadd_rn(sum_lo, px2.lo);
    sum_lo = detail::fadd_rn(sum_lo, py2.lo);
    sum_lo = detail::fadd_rn(sum_lo, pz2.lo);
    sum_lo = detail::fmaf_rn(detail::fadd_rn(dx.hi, dx.hi), dx.lo, sum_lo);
    sum_lo = detail::fmaf_rn(detail::fadd_rn(dy.hi, dy.hi), dy.lo, sum_lo);
    sum_lo = detail::fmaf_rn(dx.lo, dx.lo, sum_lo);
    sum_lo = detail::fmaf_rn(dy.lo, dy.lo, sum_lo);
    if constexpr (!UseCachedPz2) {
        // The z low-order terms are already folded into the cached pz2.
        sum_lo = detail::fmaf_rn(detail::fadd_rn(dz.hi, dz.hi), dz.lo, sum_lo);
        sum_lo = detail::fmaf_rn(dz.lo, dz.lo, sum_lo);
    }
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
    const fltflt bin = fltflt_fma_approx(fltflt{yn, r_lo}, dr_inv, mcp_partial);
    float floor_hi = ::floorf(bin.hi);
    float frac = (bin.hi - floor_hi) + bin.lo;
    // bin.lo may push bin over a boundary, in which case floor and frac are incorrect.
    // Compute an adjustment based on whether or not the fractional part is outside (0.0, 1.0).
    const float adjust = ::floorf(frac);  // -1, 0, or 1
    return SarBpBinAndWeight<float>{
        frac - adjust,
        static_cast<int32_t>(floor_hi + adjust)
    };
}

template <bool TaylorFastAddThirdOrder = false, bool PixelZIsUniform = false,
          typename IdxT = index_t>
__device__ inline SarBpBinAndWeight<float> ComputeBinWeightToPixelTaylorFast(
    const SarBpPulseBlockCache<SarBpComputeType::TaylorFast, true, PixelZIsUniform,
                               TaylorFastAddThirdOrder> &sh_mem,
    const SarBpThreadState<SarBpComputeType::TaylorFast, true> &thread_state,
    IdxT ip,
    // Reference-relative integer bin offset n, so that the lower sample is
    // ref_bin_int + n, and the pulse's reference phase ticks.
    int32_t &bin_floor_local,
    uint32_t &ref_phase_ticks)
{
    // In the mathematical derivation in the documentation, we defined s = dot(u, d) as
    // the dot product of the along-range unit vector u=(ux, uy, uz) with the local offset d,
    // and q as the squared perpendicular distance from the local pixel
    // to the pulse-to-reference-pixel line. These calculations then map directly to the
    // derivation. Each pulse's integer entry (ref_bin, or ref_bin_int for the
    // second-order uniform-z layout) holds the integer component of the
    // reference bin b_0 in .x and the pulse's reference phase ticks in .z for
    // variable z or .y for uniform z.
    // Its fractional component remains separate to preserve the interpolation
    // weight: otherwise, b_0 can be large enough that storing it as a float
    // loses low-order bin precision. Variable z stores the fractional
    // component's bit pattern in ref_bin.y. Uniform z stores the float either
    // in geometry.w or in its own array, depending on the selected layout. The
    // inner loop adds delta_bin to that fraction in fp32, floors the sum, and
    // then adds the reference integer component back. This relies on the exact
    // identity
    // floor(n + x) = n + floor(x) for integer n while avoiding the lossy fp32
    // computation of n + x. Both packed components are per-pulse values
    // populated in FillPulseBlockCache.
    // With PixelZIsUniform, every pixel shares the per-block reference's z
    // (z == 0 for Zero, or a common constant for Fixed), so ref_dz == 0 and the
    // uz term drops out of the dot product.
    float s;
    float half_inv_R;
    float ref_bin_frac;
    int32_t ref_bin_int;
    if constexpr (PixelZIsUniform && !TaylorFastAddThirdOrder) {
        s = sh_mem.ux[ip] * thread_state.ref_dx +
            sh_mem.uy[ip] * thread_state.ref_dy;
        half_inv_R = sh_mem.half_inv_R[ip];
        ref_bin_frac = sh_mem.ref_bin_frac[ip];
        const int2 ref_bin = sh_mem.ref_bin_int[ip];
        ref_bin_int = ref_bin.x;
        ref_phase_ticks = static_cast<uint32_t>(ref_bin.y);
    } else if constexpr (PixelZIsUniform) {
        const float4 geometry = sh_mem.geometry[ip];
        s = geometry.x * thread_state.ref_dx +
            geometry.y * thread_state.ref_dy;
        half_inv_R = geometry.z;
        ref_bin_frac = geometry.w;
        const int2 ref_bin = sh_mem.ref_bin[ip];
        ref_bin_int = ref_bin.x;
        ref_phase_ticks = static_cast<uint32_t>(ref_bin.y);
    } else {
        const float4 geometry = sh_mem.geometry[ip];
        const int4 ref_bin = sh_mem.ref_bin[ip];
        ref_phase_ticks = static_cast<uint32_t>(ref_bin.z);
        s = geometry.x * thread_state.ref_dx +
            geometry.y * thread_state.ref_dy +
            geometry.z * thread_state.ref_dz;
        half_inv_R = geometry.w;
        ref_bin_frac = __int_as_float(ref_bin.y);
        ref_bin_int = ref_bin.x;
    }
    const float q = fmaf(-s, s, thread_state.ref_dsq);
    const float dR_2nd = fmaf(q, half_inv_R, s);
    float dR = dR_2nd;
    if constexpr (TaylorFastAddThirdOrder) {
        dR = fmaf(-2.0f * half_inv_R * half_inv_R * s, q, dR_2nd);
    }
    const float bin_loc = fmaf(dR, thread_state.dr_inv_f32, ref_bin_frac);
    // Convert the floored integer back to float rather than calling floorf:
    // the int-to-float conversion issues on the FMA pipe, while FRND would
    // share the conversion/MUFU pipe with the phase's __sincosf.
    bin_floor_local = __float2int_rd(bin_loc);
    const float bin_floor = static_cast<float>(bin_floor_local);
    // Re-form the weight against the integer bin rather than taking
    // bin_loc - bin_floor, so that both roundings happen at O(1) instead of
    // inheriting ulp(bin_loc) from a value spanning the CTA's bins. The
    // synthesized phase scales w by alpha, so its rounding lands in the phase
    // unattenuated. The pair is not exactly consistent: w can land outside
    // [0, 1) by up to ulp(bin_loc)/2, which the linear interpolation and the
    // phase both tolerate.
    return SarBpBinAndWeight<float>{
        fmaf(dR, thread_state.dr_inv_f32, -bin_floor) + ref_bin_frac,
        ref_bin_int + bin_floor_local
    };
}

template <SarBpComputeType ComputeType, bool PixelZIsZero, bool UseCachedDz2, bool PixelZIsUniform, typename strict_compute_t, typename loose_compute_t, typename IdxT>
__device__ inline SarBpBinAndWeight<loose_compute_t> ComputeBinWeightToPixelPulseBlockCache(
    const SarBpPulseBlockCache<ComputeType, true, PixelZIsUniform> &sh_mem,
    IdxT ip,
    loose_compute_t px,
    loose_compute_t py,
    [[maybe_unused]] loose_compute_t pz,
    strict_compute_t dr_inv)
{
    constexpr bool IsFloat = ComputeType == SarBpComputeType::Float;
    constexpr bool IsMixed = ComputeType == SarBpComputeType::Mixed;
    static_assert(IsFloat || IsMixed, "SarBp: pulse-block cache bin helper only supports Float and Mixed");

    const strict_compute_t apx = sh_mem.ant_pos[ip][0];
    const strict_compute_t apy = sh_mem.ant_pos[ip][1];
    const strict_compute_t mcp_partial = sh_mem.ant_pos[ip][3];
    const strict_compute_t dx = static_cast<strict_compute_t>(px) - apx;
    const strict_compute_t dy = static_cast<strict_compute_t>(py) - apy;
    // The z contribution to the squared distance. In Fixed mode with caching,
    // slot [2] already holds the per-pulse dz^2 (uniform across pixels), so the
    // per-pixel subtract+square is skipped. Otherwise slot [2] holds the antenna
    // z and dz = pz - apz (or -apz when PixelZIsZero -- the explicit elision is
    // kept here because, unlike the other helpers, the FP64 Mixed path does not
    // fold the pz == 0 constant as tightly, costing a few registers).
    strict_compute_t dz2;
    if constexpr (UseCachedDz2) {
        dz2 = sh_mem.ant_pos[ip][2];
    } else {
        const strict_compute_t apz = sh_mem.ant_pos[ip][2];
        const strict_compute_t dz = PixelZIsZero ? -apz : (static_cast<strict_compute_t>(pz) - apz);
        dz2 = dz * dz;
    }
    strict_compute_t dist;
    if constexpr (IsFloat) {
        dist = ::sqrtf(dx*dx + dy*dy + dz2);
    } else {
#if __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
        dist = ::sqrt(dx*dx + dy*dy + dz2);
#else
        dist = NewtonRaphsonSqrt(dx*dx + dy*dy + dz2);
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
    return SarBpBinAndWeight<loose_compute_t>{
        static_cast<loose_compute_t>(bin - bin_floor),
        static_cast<int32_t>(bin_floor)
    };
}

template <SarBpComputeType ComputeType, SarBpPixelZMode PixelZMode, bool TaylorFastAddThirdOrder>
struct SarBpMinCtaCount { static constexpr int value = 5; };

template <> struct SarBpMinCtaCount<SarBpComputeType::TaylorFast, SarBpPixelZMode::Variable, false> { static constexpr int value = 6; };
template <> struct SarBpMinCtaCount<SarBpComputeType::TaylorFast, SarBpPixelZMode::Zero, false> { static constexpr int value = 6; };
template <> struct SarBpMinCtaCount<SarBpComputeType::TaylorFast, SarBpPixelZMode::Fixed, false> { static constexpr int value = 6; };
template <> struct SarBpMinCtaCount<SarBpComputeType::Float, SarBpPixelZMode::Zero, false> { static constexpr int value = 6; };
template <> struct SarBpMinCtaCount<SarBpComputeType::Float, SarBpPixelZMode::Fixed, false> { static constexpr int value = 6; };
template <> struct SarBpMinCtaCount<SarBpComputeType::Float, SarBpPixelZMode::Variable, false> { static constexpr int value = 6; };

// Cap the maximum CTA count to the architecture's limit. sm_75 caps at 1024 threads/SM
// and all other supported architectures allow at least 1536 (6 CTAs). Pushing to 8
// CTAS_PER_SM would greatly limit register allocation, so treat 6 as the common maximum.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
#define MATX_SARBP_MAX_CTAS_PER_SM 4
#else
#define MATX_SARBP_MAX_CTAS_PER_SM 6
#endif

// The requested CTA count, clamped to what this architecture can honor.
template <SarBpComputeType ComputeType, SarBpPixelZMode PixelZMode, bool TaylorFastAddThirdOrder>
inline constexpr int SarBpMinCtas =
    (SarBpMinCtaCount<ComputeType, PixelZMode, TaylorFastAddThirdOrder>::value < MATX_SARBP_MAX_CTAS_PER_SM)
        ? SarBpMinCtaCount<ComputeType, PixelZMode, TaylorFastAddThirdOrder>::value
        : MATX_SARBP_MAX_CTAS_PER_SM;

// SarBp backprojection kernel. PixelZMode is a compile-time assumption about
// the pixel z (height) coordinates, selected by the caller via the
// PropSarBpPixelZIsZero / PropSarBpPixelZIsFixed properties (see
// operators/sar_bp.h):
//   - Variable (default): each pixel may have a distinct z; produces the same
//     code (and identical mangled name / register usage) as before this
//     parameter existed, so default callers are unaffected.
//   - Zero: every pixel z == 0, so the per-pulse range computation drops the
//     pixel-z term (dz collapses to -antenna_z).
//   - Fixed: every pixel shares one (runtime) z value; on the shared-cache
//     compute paths the per-pulse z contribution dz^2 is uniform across pixels
//     and is precomputed once per pulse (see FillPulseBlockCache), removing the
//     per-pixel-per-pulse subtract+square from the inner loop.
template <SarBpComputeType ComputeType, typename OutImageType, typename InitialImageType, typename RangeProfilesType, typename PlatPosType, typename VoxLocType, typename RangeToMcpType, bool PhaseLUT, bool IsUnitStride, bool TaylorFastAddThirdOrder = false, SarBpPixelZMode PixelZMode = SarBpPixelZMode::Variable, typename IdxT = index_t>
// Set the minimum CTA count as a function of the compute type and pixel z mode. This
// is most important for Double/Mixed where increasing occupancy even at the cost of
// register spills is beneficial. For modes with reduced register pressure, we set
// higher minimum CTA counts to avoid the occupancy loss associated with a fixed value
// for all instantiations.
__launch_bounds__(16*16, (SarBpMinCtas<ComputeType, PixelZMode, TaylorFastAddThirdOrder>))
__global__ void SarBp(OutImageType output, const InitialImageType initial_image, const __grid_constant__ RangeProfilesType range_profiles, const __grid_constant__ PlatPosType platform_positions, const __grid_constant__ VoxLocType voxel_locations, const __grid_constant__ RangeToMcpType range_to_mcp,
                      dr_inv_param_t<ComputeType> dr_inv,
                      phase_correction_param_t<ComputeType, PhaseLUT> phase_correction_partial,
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

    constexpr bool IsDouble = ComputeType == SarBpComputeType::Double;
    constexpr bool IsMixed = ComputeType == SarBpComputeType::Mixed;
    constexpr bool IsFloatFloat = ComputeType == SarBpComputeType::FloatFloat;
    constexpr bool IsFloat = ComputeType == SarBpComputeType::Float;
    constexpr bool IsTaylorFast = ComputeType == SarBpComputeType::TaylorFast;

    constexpr bool PixelZIsZero  = PixelZMode == SarBpPixelZMode::Zero;
    constexpr bool PixelZIsFixed = PixelZMode == SarBpPixelZMode::Fixed;
    // For the per-pulse-cache compute paths, PixelZIsZero treats z as the
    // compile-time constant 0 (no pz value needed). Both Zero and Fixed share
    // the property that the pixel z is uniform across the CTA.
    constexpr bool PixelZIsUniform = PixelZIsZero || PixelZIsFixed;
    // Per-pulse z^2 caching. When the pixel z is uniform across the CTA the z
    // contribution to the squared range is the same for every pixel, so it is
    // computed once per pulse in FillPulseBlockCache and stored in the
    // (otherwise unused) antenna-z cache slot -- no extra shared memory.
    //   - UseCachedDz2: Float/Mixed pulse-block-cache path. Scalar dz^2. Fixed
    //     only; Zero there keeps the bit-exact inline elision (dz = -apz).
    //   - UseCachedPz2FF: FloatFloat path. Fully-compensated fltflt dz^2,
    //     applied to both Zero and Fixed since the compensated-arithmetic z
    //     terms (dz two-sum, z two-prod, 2*dz.hi*dz.lo, dz.lo^2) are the most
    //     expensive part of the inner loop and collapse to a single fltflt add.
    // TaylorFast handles uniform z via its reference-relative offsets
    // (ref_dz == 0, see PixelZIsUniform); the non-cache paths fall back to the
    // per-pixel dz computation.
    constexpr bool UseCachedDz2 =
        PixelZIsFixed && (IsFloat || IsMixed) && PhaseLUT;
    constexpr bool UseCachedPz2FF =
        PixelZIsUniform && IsFloatFloat;
    // The cooperative cache fill needs the (non-zero) uniform pixel z in every
    // thread, including out-of-image threads whose own voxel_loc is zeroed.
    // Those threads patch voxel_loc.z from the CTA-origin pixel (see below) so
    // pz is correct for all. Zero needs nothing here (its uniform z is 0).
    constexpr bool NeedFixedPz =
        PixelZIsFixed &&
        (((IsFloat || IsMixed) && PhaseLUT) || IsFloatFloat);

    using initial_image_t = typename InitialImageType::value_type;
    using image_t = typename OutImageType::value_type;
    using range_profiles_t = typename RangeProfilesType::value_type;
    using plat_pos_t = typename PlatPosType::value_type;
    using voxel_loc_t = typename VoxLocType::value_type;
    using compute_t = typename std::conditional<IsDouble, double, float>::type;
    using strict_compute_t = sarbp_strict_compute_t<ComputeType>;
    using strict_or_ff_compute_t = typename std::conditional<IsFloatFloat, fltflt, strict_compute_t>::type;
    using strict_complex_compute_t = cuda::std::complex<strict_compute_t>;
    using loose_compute_t = typename std::conditional<IsDouble, double, float>::type;
    using loose_complex_compute_t = cuda::std::complex<loose_compute_t>;

    const index_t image_height = output.Size(0);
    const index_t image_width = output.Size(1);
    const index_t ix = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const index_t iy = static_cast<index_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const bool is_valid = ix < image_width && iy < image_height;

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

    // Some compute modes cache per-pulse data in shared memory once per pulse
    // block. With PhaseLUT=false, Float/Mixed still need the range-to-MCP
    // delta for sincos, which the cache does not preserve, so fall back to
    // direct global reads in that case.
    constexpr bool UsePulseBlockCache =
        IsFloatFloat ||
        IsTaylorFast ||
        ((IsFloat || IsMixed) && PhaseLUT);
    constexpr bool TaylorFastThirdOrderCache =
        IsTaylorFast && TaylorFastAddThirdOrder;

    __shared__ SarBpPulseBlockCache<ComputeType, UsePulseBlockCache, PixelZIsUniform,
                                    TaylorFastThirdOrderCache> sh_mem;

    if constexpr (!UsePulseBlockCache) {
        // When the cache is enabled, keep all threads active so they can
        // participate in the CTA-wide cooperative loads of antenna positions.
        if (! is_valid) return;
    }

    const IdxT num_pulses = static_cast<IdxT>(range_profiles.Size(0));
    const int32_t num_range_bins = static_cast<int32_t>(range_profiles.Size(1));

    static_assert(cuda::std::is_same_v<voxel_loc_t, double3> || cuda::std::is_same_v<voxel_loc_t, double4> ||
        cuda::std::is_same_v<voxel_loc_t, float3> || cuda::std::is_same_v<voxel_loc_t, float4>, "SarBp: VoxLocType must represent a 2D operator of type double3, double4, float3, or float4");
    // voxel_locations is read once per thread (outside the pulse loop), so it
    // is deliberately not covered by the IsUnitStride fast-path logic. The
    // one-shot cost of its operator() is negligible and its layout is free to
    // be anything the caller wants without blocking fast-path eligibility.
    voxel_loc_t voxel_loc = is_valid ? voxel_locations(iy, ix) : voxel_loc_t{};
    // Fixed mode needs the uniform pixel z in every thread that fills the cache,
    // but out-of-image threads have a zeroed voxel_loc. They take z from the
    // CTA-origin pixel (thread 0's, always in-image since the CTA origin lies
    // within the image); in-image threads already have the correct z. Taken only
    // by out-of-image threads on edge CTAs.
    if constexpr (NeedFixedPz) {
        if (!is_valid) {
            const index_t origin_x = static_cast<index_t>(blockIdx.x * blockDim.x);
            const index_t origin_y = static_cast<index_t>(blockIdx.y * blockDim.y);
            voxel_loc.z = voxel_locations(origin_y, origin_x).z;
        }
    }
    const loose_compute_t py = static_cast<loose_compute_t>(voxel_loc.y);
    const loose_compute_t px = static_cast<loose_compute_t>(voxel_loc.x);
    // When PixelZIsZero is known, pz folds to a compile-time 0 so the z terms in
    // the range/bin helpers are dropped.
    const loose_compute_t pz = PixelZIsZero ? static_cast<loose_compute_t>(0) : static_cast<loose_compute_t>(voxel_loc.z);

    // TensorAccessor wraps each hot tensor and picks the fast pointer path
    // when IsUnitStride && is_tensor_view_v<Op>, otherwise forwards to
    // operator(). Binding the tensor's Data() and Stride(0) into the
    // accessor's members forces the compiler to materialize the grid-constant
    // LDCs once at kernel entry rather than reloading inside the pulse loop.
    detail::TensorAccessor<RangeProfilesType, IsUnitStride, IdxT> rp(range_profiles);
    detail::TensorAccessor<PlatPosType, IsUnitStride> pp(platform_positions);

    // range_to_mcp is required to be a rank-0 or rank-1 MatX operator; the
    // TensorAccessor picks the fast pointer path on IsUnitStride for tensor
    // views or forwards to operator() for computed ops (e.g. ConstVal).
    const detail::TensorAccessor<RangeToMcpType, IsUnitStride> rtm_acc(range_to_mcp);

    const auto r_to_mcp = [rtm_acc](IdxT p) -> auto {
        if constexpr (RangeToMcpType::Rank() == 0) {
            return rtm_acc();
        } else {
            return rtm_acc(p);
        }
    };

    // TaylorFast synthesizes its reference phase from a per-pulse reference
    // angle and the local bin offset instead of gathering it from the LUT.
    constexpr bool SynthesizePhase = detail::SarBpSynthesizesPhase(ComputeType);
    // The LUT and direct helpers below take the scalar phase-correction factor.
    // TaylorFast receives it in both precisions and calls neither helper, so they
    // see its float value. Either way this is a copy, not a conversion.
    const auto phase_correction_scalar = [&]() {
        if constexpr (SynthesizePhase) {
            return phase_correction_partial.alpha32;
        } else {
            return phase_correction_partial;
        }
    }();

    // On the PhaseLUT path, phase_correction_partial is already loose_compute_t (see
    // phase_correction_param_t), so it is used directly without a conversion. A mismatch
    // would silently reintroduce a per-pulse conversion in the inner loop.
    static_assert(!PhaseLUT ||
                  cuda::std::is_same_v<cuda::std::remove_cv_t<decltype(phase_correction_scalar)>, loose_compute_t>,
                  "SarBp: PhaseLUT path expects phase_correction_partial pre-converted to loose_compute_t");
    const auto get_reference_phase_lut = [&phase_lut, &phase_correction_scalar](int32_t bin_floor_int, loose_compute_t w) -> loose_complex_compute_t {
        const loose_complex_compute_t base_phase = phase_lut[bin_floor_int];
        loose_compute_t incr_sinx, incr_cosx;
        if constexpr (cuda::std::is_same_v<loose_compute_t, double>) {
            ::sincos(phase_correction_scalar * w, &incr_sinx, &incr_cosx);
        } else {
            // The lambda is also checked when PhaseLUT is false, where the phase
            // factor can be double. On the LUT path this cast is a no-op because
            // the host already supplies loose_compute_t.
            __sincosf(static_cast<loose_compute_t>(phase_correction_scalar) * w, &incr_sinx, &incr_cosx);
        }

        return loose_complex_compute_t{
            base_phase.real() * incr_cosx - base_phase.imag() * incr_sinx,
            base_phase.real() * incr_sinx + base_phase.imag() * incr_cosx
        };
    };

    const auto get_reference_phase_direct = [&phase_correction_scalar](strict_compute_t range_delta) -> loose_complex_compute_t {
        const strict_compute_t phase = static_cast<strict_compute_t>(phase_correction_scalar * range_delta);
        strict_compute_t sinx, cosx;
        if constexpr (cuda::std::is_same_v<strict_compute_t, double>) {
            ::sincos(phase, &sinx, &cosx);
        } else {
            ::sincosf(phase, &sinx, &cosx);
        }
        return loose_complex_compute_t{
            static_cast<loose_compute_t>(cosx), static_cast<loose_compute_t>(sinx)
        };
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

    SarBpThreadState<ComputeType, PhaseLUT> thread_state{};
    if constexpr (IsTaylorFast) {
        static_assert(PhaseLUT == true, "SarBp: TaylorFast is only instantiated with PhaseLUT=true");
        if constexpr (PhaseLUT) {
            thread_state.dr_inv_f32 = dr_inv.f32;
            if (tid == 0) {
                // Use the midpoint of the CTA's valid image footprint. For
                // partial right/bottom edge CTAs, clamping the nominal CTA
                // center to image_width-1/image_height-1 would choose an edge
                // or corner reference and unnecessarily increase local offsets.
                const index_t block_x0 = static_cast<index_t>(blockIdx.x * blockDim.x);
                const index_t block_y0 = static_cast<index_t>(blockIdx.y * blockDim.y);
                const index_t valid_width = ((image_width - block_x0) < static_cast<index_t>(blockDim.x)) ?
                    (image_width - block_x0) : static_cast<index_t>(blockDim.x);
                const index_t valid_height = ((image_height - block_y0) < static_cast<index_t>(blockDim.y)) ?
                    (image_height - block_y0) : static_cast<index_t>(blockDim.y);
                const index_t ax_c = block_x0 + valid_width / 2;
                const index_t ay_c = block_y0 + valid_height / 2;
                const voxel_loc_t ref_loc = voxel_locations(ay_c, ax_c);
                sh_mem.ref_x = static_cast<strict_compute_t>(ref_loc.x);
                sh_mem.ref_y = static_cast<strict_compute_t>(ref_loc.y);
                sh_mem.ref_z = static_cast<strict_compute_t>(ref_loc.z);
            }
            __syncthreads();

            thread_state.ref_dx = is_valid ? static_cast<float>(static_cast<strict_compute_t>(voxel_loc.x) - sh_mem.ref_x) : 0.0f;
            thread_state.ref_dy = is_valid ? static_cast<float>(static_cast<strict_compute_t>(voxel_loc.y) - sh_mem.ref_y) : 0.0f;
            // When the pixel z is uniform across the CTA (Zero: z == 0, or
            // Fixed: z == const), the per-block reference pixel shares that same
            // z, so ref_dz is identically 0 and drops out of ref_dsq and the dot
            // product in ComputeBinWeightToPixelTaylorFast.
            if constexpr (PixelZIsUniform) {
                thread_state.ref_dz = 0.0f;
                thread_state.ref_dsq = fmaf(thread_state.ref_dx, thread_state.ref_dx,
                                            thread_state.ref_dy * thread_state.ref_dy);
            } else {
                thread_state.ref_dz = is_valid ? static_cast<float>(static_cast<strict_compute_t>(voxel_loc.z) - sh_mem.ref_z) : 0.0f;
                thread_state.ref_dsq = fmaf(thread_state.ref_dx, thread_state.ref_dx,
                                            fmaf(thread_state.ref_dy, thread_state.ref_dy,
                                                 thread_state.ref_dz * thread_state.ref_dz));
            }
        }
    }

    loose_complex_compute_t accum{};
    // fftshift places zero differential range at floor(N/2). For even N,
    // this is index N/2, the higher-indexed of the two bins adjacent to
    // the array's geometric midpoint (N-1)/2.
    const loose_compute_t bin_offset =
        static_cast<loose_compute_t>(num_range_bins / 2);

    // Most ComputeTypes amortize some redundant computations or data conversions by leveraging
    // per-pulse-block shared memory.
    const auto FillPulseBlockCache = [&](IdxT pulse_base, IdxT num_pulses_in_block) {
        if constexpr (UsePulseBlockCache) {
            __syncthreads();
            for (IdxT ip = static_cast<IdxT>(tid); ip < num_pulses_in_block;
                 ip += static_cast<IdxT>(blockDim.x * blockDim.y)) {
                const IdxT p = pulse_base + ip;
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

                if constexpr (IsFloatFloat) {
                    sh_mem.ant_pos[ip][0] = static_cast<fltflt>(cuda::std::get<0>(xyz));
                    sh_mem.ant_pos[ip][1] = static_cast<fltflt>(cuda::std::get<1>(xyz));
                    const fltflt apz = static_cast<fltflt>(cuda::std::get<2>(xyz));
                    if constexpr (UseCachedPz2FF) {
                        // Cache the fully-compensated per-pulse dz^2 (uniform
                        // pixel z) in the antenna-z slot. dz is formed with the
                        // same arithmetic the inner loop would have used, so the
                        // only difference from Variable is that the z low-order
                        // terms are summed here once per pulse rather than per
                        // pixel. In the Zero mode pz is the compile-time 0.
                        fltflt dz = fltflt_two_sum(pz, -apz.hi);
                        dz.lo = detail::fadd_rn(dz.lo, -apz.lo);
                        const fltflt dz2hi = fltflt_two_prod_fma(dz.hi, dz.hi);
                        float pz2_lo = dz2hi.lo;
                        pz2_lo = detail::fmaf_rn(detail::fadd_rn(dz.hi, dz.hi), dz.lo, pz2_lo);
                        pz2_lo = detail::fmaf_rn(dz.lo, dz.lo, pz2_lo);
                        // Renormalize dz^2 into a canonical fltflt. fast_two_sum's
                        // |a| >= |b| precondition (a = dz.hi^2, b = the
                        // O(ulp(dz.hi^2)) remainder) holds whenever |dz.hi| is not
                        // tiny. It can fail only if the platform sits within
                        // ~ulp(antenna_z) of the pixel-z plane (sub-mm at km
                        // altitudes, not physical for SAR), where cancellation makes
                        // dz non-canonical. Even then it is typically benign; the hi word
                        // is still correctly rounded and dz^2 is negligible versus
                        // dx^2 + dy^2 for physically meaningful SAR geometries.
                        sh_mem.ant_pos[ip][2] = fltflt_fast_two_sum(dz2hi.hi, pz2_lo);
                    } else {
                        sh_mem.ant_pos[ip][2] = apz;
                    }
                    const fltflt rtm = static_cast<fltflt>(r_to_mcp(p));
                    const fltflt neg_rtm = fltflt{-rtm.hi, -rtm.lo};
                    sh_mem.ant_pos[ip][3] = fltflt_fma_approx(neg_rtm, dr_inv, bin_offset);
                } else if constexpr (IsTaylorFast) {
                    const strict_compute_t pcx = static_cast<strict_compute_t>(cuda::std::get<0>(xyz));
                    const strict_compute_t pcy = static_cast<strict_compute_t>(cuda::std::get<1>(xyz));
                    const strict_compute_t pcz = static_cast<strict_compute_t>(cuda::std::get<2>(xyz));
                    const strict_compute_t rtm = static_cast<strict_compute_t>(r_to_mcp(p));

                    const strict_compute_t dxa = sh_mem.ref_x - pcx;
                    const strict_compute_t dya = sh_mem.ref_y - pcy;
                    const strict_compute_t dza = sh_mem.ref_z - pcz;
                    // We assume double for TaylorFast strict_compute_t and thus use ::sqrt()/::floor() here
                    static_assert(cuda::std::is_same_v<strict_compute_t, double>, "SarBp: TaylorFast compute type requires double precision");
                    const strict_compute_t ref_range_pure = ::sqrt(dxa*dxa + dya*dya + dza*dza);
                    const strict_compute_t ref_range = ref_range_pure - rtm;
                    const strict_compute_t ref_bin = ref_range * dr_inv.f64 + static_cast<strict_compute_t>(bin_offset);
                    const strict_compute_t bin_floor = ::floor(ref_bin);
                    const strict_compute_t inv_R_d = static_cast<strict_compute_t>(1.0) / ref_range_pure;

                    const int32_t ref_bin_int = static_cast<int32_t>(bin_floor);
                    const float ref_bin_frac = static_cast<float>(ref_bin - bin_floor);
                    const float ux = static_cast<float>(dxa * inv_R_d);
                    const float uy = static_cast<float>(dya * inv_R_d);
                    const float half_inv_R = static_cast<float>(static_cast<strict_compute_t>(0.5) * inv_R_d);
                    // Once per pulse per CTA, amortized over the CTA's pixels.
                    const int32_t ref_phase_ticks = static_cast<int32_t>(SarBpRefPhaseTicks(
                        phase_correction_partial, ref_bin_int, num_range_bins / 2));
                    if constexpr (PixelZIsUniform && !TaylorFastAddThirdOrder) {
                        sh_mem.ref_bin_int[ip] = int2{ref_bin_int, ref_phase_ticks};
                        sh_mem.ref_bin_frac[ip] = ref_bin_frac;
                        sh_mem.ux[ip] = ux;
                        sh_mem.uy[ip] = uy;
                        sh_mem.half_inv_R[ip] = half_inv_R;
                    } else if constexpr (PixelZIsUniform) {
                        sh_mem.ref_bin[ip] = int2{ref_bin_int, ref_phase_ticks};
                        sh_mem.geometry[ip] = float4{ux, uy, half_inv_R, ref_bin_frac};
                    } else {
                        sh_mem.ref_bin[ip] = int4{ref_bin_int, __float_as_int(ref_bin_frac),
                                                  ref_phase_ticks, 0};
                        sh_mem.geometry[ip] = float4{
                            ux,
                            uy,
                            static_cast<float>(dza * inv_R_d),
                            half_inv_R
                        };
                    }
                } else {
                    // Float / Mixed: cast inputs to strict_compute_t (float / double)
                    // once per pulse here, instead of once per pulse per pixel.
                    sh_mem.ant_pos[ip][0] = static_cast<strict_compute_t>(cuda::std::get<0>(xyz));
                    sh_mem.ant_pos[ip][1] = static_cast<strict_compute_t>(cuda::std::get<1>(xyz));
                    const strict_compute_t apz = static_cast<strict_compute_t>(cuda::std::get<2>(xyz));
                    if constexpr (UseCachedDz2) {
                        // Fixed mode: the pixel z is uniform, so dz = pz - apz and
                        // its square are per-pulse constants. Cache dz^2 in the
                        // (otherwise unused after this) antenna-z slot so the inner
                        // loop loads it directly instead of recomputing the
                        // subtract+square for every pixel.
                        const strict_compute_t dz = static_cast<strict_compute_t>(pz) - apz;
                        sh_mem.ant_pos[ip][2] = dz * dz;
                    } else {
                        sh_mem.ant_pos[ip][2] = apz;
                    }
                    const strict_compute_t rtm = static_cast<strict_compute_t>(r_to_mcp(p));
                    if constexpr (cuda::std::is_same_v<strict_compute_t, double>) {
                        sh_mem.ant_pos[ip][3] = ::fma(-rtm, dr_inv, static_cast<double>(bin_offset));
                    } else {
                        sh_mem.ant_pos[ip][3] = ::fmaf(-rtm, dr_inv, bin_offset);
                    }
                }
            }
            __syncthreads();
        }
    };

    const int num_pulse_blocks = static_cast<int>(
        (num_pulses + static_cast<IdxT>(PULSE_BLOCK_SIZE) - static_cast<IdxT>(1)) /
        static_cast<IdxT>(PULSE_BLOCK_SIZE));
    for (int block = 0; block < num_pulse_blocks; ++block) {
        const IdxT pulse_base = static_cast<IdxT>(block) * static_cast<IdxT>(PULSE_BLOCK_SIZE);
        const IdxT pulses_remaining = num_pulses - pulse_base;
        const IdxT num_pulses_in_block =
            (pulses_remaining < static_cast<IdxT>(PULSE_BLOCK_SIZE)) ?
                pulses_remaining : static_cast<IdxT>(PULSE_BLOCK_SIZE);

        FillPulseBlockCache(pulse_base, num_pulses_in_block);
        if (! is_valid) {
            continue;
        }

        // On the unit-stride fast path, carry the current pulse's row offset
        // across iterations, so a sample's element index is row_offset + bin:
        // one add instead of recomputing (pulse_base + ip)*stride + bin.
        // Exclude direct-sincos paths as this added register spills in that case.
        constexpr bool UseRowOffset = IsUnitStride && PhaseLUT;
        [[maybe_unused]] IdxT row_offset = 0;
        [[maybe_unused]] IdxT row_step = 0;
        if constexpr (UseRowOffset) {
            row_step = rp.outer_stride(0);
            row_offset = pulse_base * row_step;
        }

        // The row offset advances in the loop header so that samples skipped
        // by the bounds check below still step it.
        #pragma unroll 4
        for (IdxT ip = 0; ip < num_pulses_in_block; ++ip, row_offset += row_step) {
            [[maybe_unused]] const IdxT p = pulse_base + ip;

            // The following branches all compute at least the base range bin index and interpolation weight
            // for the current pulse. Paths not using the PhaseLUT optimization will also populate the
            // thread_state.range_delta member for use in get_reference_phase_direct().
            SarBpBinAndWeight<loose_compute_t> bin_weight;
            [[maybe_unused]] int32_t bin_floor_local = 0;
            [[maybe_unused]] uint32_t ref_phase_ticks = 0;
            if constexpr (IsFloatFloat) {
                bin_weight = ComputeBinWeightToPixelFloatFloat<UseCachedPz2FF, PixelZIsUniform>(sh_mem, ip, px, py, pz, dr_inv);
            } else if constexpr (IsTaylorFast) {
                static_assert(PhaseLUT == true, "SarBp: TaylorFast is only instantiated with PhaseLUT=true");
                // Zero and Fixed both make ref_dz == 0, so PixelZIsUniform drives the elision.
                bin_weight = ComputeBinWeightToPixelTaylorFast<
                    TaylorFastAddThirdOrder, PixelZIsUniform>(
                        sh_mem, thread_state, ip, bin_floor_local, ref_phase_ticks);
            } else if constexpr (UsePulseBlockCache) {
                bin_weight = ComputeBinWeightToPixelPulseBlockCache<ComputeType, PixelZIsZero, UseCachedDz2, PixelZIsUniform>(sh_mem, ip, px, py, pz, dr_inv);
            } else {
                // Below is the most direct / standard path for backprojection. We compute the distance
                // from the antenna phase center to the pixel for this pulse and from that the differential range
                // to the MCP. From that differential range, we compute the base range bin index and interpolation weight
                // for range profile sampling. Finally, we use the differential range to compute the reference phase
                // and accumulate the contribution to the pixel. All other paths implement similar logic using various
                // optimizations.
                const strict_compute_t range_delta =
                    ComputeRangeToPixel<decltype(pp), ComputeType, strict_compute_t, loose_compute_t>(
                        pp, p, px, py, pz) - static_cast<strict_compute_t>(r_to_mcp(p));
                if constexpr (!PhaseLUT) {
                    thread_state.range_delta = range_delta;
                }
                const strict_compute_t bin = range_delta * dr_inv + bin_offset;
                strict_compute_t bin_floor;
                if constexpr (cuda::std::is_same_v<strict_compute_t, double>) {
                    bin_floor = ::floor(bin);
                } else {
                    bin_floor = ::floorf(bin);
                }
                bin_weight.w = static_cast<loose_compute_t>(bin - bin_floor);
                bin_weight.bin_floor_int = static_cast<int32_t>(bin_floor);
            }

            // One unsigned compare covers both bounds: a negative bin wraps to
            // a value above any valid upper bound. sar_bp_impl verifies
            // 2 <= num_range_bins <= INT32_MAX on the host, so num_range_bins - 1
            // lies in [1, 2^31 - 2], below every negative bin read as uint32_t
            // (at least 2^31).
            if (static_cast<uint32_t>(bin_weight.bin_floor_int) >=
                static_cast<uint32_t>(num_range_bins - 1)) {
                continue;
            }

            // rp accessor picks the fast pointer path on IsUnitStride or
            // falls through to operator().
            range_profiles_t sample_lo, sample_hi;
            if constexpr (UseRowOffset) {
                const IdxT offset = row_offset + bin_weight.bin_floor_int;
                sample_lo = rp.at_offset(offset);
                sample_hi = rp.at_offset(offset + 1);
            } else {
                sample_lo = rp(p, bin_weight.bin_floor_int);
                sample_hi = rp(p, bin_weight.bin_floor_int + 1);
            }

            const loose_complex_compute_t sample = [&sample_lo, &sample_hi, &bin_weight]() -> loose_complex_compute_t {
                const loose_complex_compute_t loose_sample_lo = static_cast<loose_complex_compute_t>(sample_lo);
                const loose_complex_compute_t loose_sample_hi = static_cast<loose_complex_compute_t>(sample_hi);
                const loose_compute_t w = bin_weight.w;
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

            static_assert((!IsFloatFloat &&
                            !IsTaylorFast) || PhaseLUT == true,
                            "SarBp: FloatFloat and TaylorFast are only instantiated with PhaseLUT=true");
            const loose_complex_compute_t ref_phase = [&]() {
                if constexpr (SynthesizePhase) {
                    return SarBpSynthesizePhase(phase_correction_partial, ref_phase_ticks, bin_floor_local, bin_weight.w);
                } else if constexpr (PhaseLUT) {
                    return get_reference_phase_lut(bin_weight.bin_floor_int, bin_weight.w);
                } else {
                    return get_reference_phase_direct(thread_state.range_delta);
                }
            }();

            accum = accumulate_contribution(accum, sample, ref_phase);
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
