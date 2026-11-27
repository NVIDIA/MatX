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

// a * b + c as a single fused multiply-add.
template <typename T>
__device__ __forceinline__ T SarBpFma(T a, T b, T c)
{
    if constexpr (cuda::std::is_same_v<T, float>) {
        return __fmaf_rn(a, b, c);
    } else {
        return __fma_rn(a, b, c);
    }
}

// Squared distance dx^2 + dy^2 + dz^2 as one multiply and two fused
// multiply-adds, the minimum for three per-pixel offsets. The operation order
// is explicit so that the rounding does not depend on how the compiler
// contracts the expression. In particular, with two pixels per thread and a
// uniform pixel z, the compiler would otherwise compute the shared dz * dz once
// and add it to each pixel unfused, which costs an extra instruction per
// pixel-pulse and rounds differently from one pixel per thread.
template <typename T>
__device__ __forceinline__ T SarBpRangeSquared(T dx, T dy, T dz)
{
    return SarBpFma(dz, dz, SarBpFma(dx, dx, dy * dy));
}

// Squared distance from a precomputed dz^2 as two fused multiply-adds. With a
// zero or fixed pixel z, dz^2 is the same for every pixel of a pulse, so it is
// computed once per pulse (or once per thread for all of its pixels) and each
// pixel needs only these two instructions. The rounding differs from
// SarBpRangeSquared.
template <typename T>
__device__ __forceinline__ T SarBpRangeSquaredFromDz2(T dx, T dy, T dz2)
{
    return SarBpFma(dx, dx, SarBpFma(dy, dy, dz2));
}

// pixel_coord_t is the kernel's pixel coordinate type (see SarBpPixelCoord), so
// Mixed receives fp64 coordinates when the voxel locations are fp64. With
// SharedDz2, the caller guarantees that pz is the same for all of a thread's
// pixels, so the compiler computes dz * dz once per pulse for all of them.
template <typename PlatPosAccessor, SarBpComputeType ComputeType, typename strict_compute_t, typename pixel_coord_t, bool SharedDz2, typename IdxT>
__device__ inline strict_compute_t ComputeRangeToPixel(const PlatPosAccessor &ant_pos, const IdxT pulse_idx, const pixel_coord_t px, const pixel_coord_t py, const pixel_coord_t pz) {
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

    strict_compute_t r2;
    if constexpr (SharedDz2) {
        r2 = SarBpRangeSquaredFromDz2(dx, dy, dz * dz);
    } else {
        r2 = SarBpRangeSquared(dx, dy, dz);
    }
    if constexpr (IsFloat) {
        return ::sqrtf(r2);
    } else {
        if constexpr (IsMixed) {
#if __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
            return ::sqrt(r2);
#else
            // Only use the Newton-Raphson approach on systems with reduced FP64 throughput
            return NewtonRaphsonSqrt(r2);
#endif
        } else {
            return ::sqrt(r2);
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

// Type in which the kernel holds a pixel's coordinates for the pulse-to-pixel
// range computation. FP64 voxel locations are held in FP64 for mixed and as
// fltflt for FloatFloat. FP32 voxel locations stay FP32. TaylorFast forms its
// offsets from a CTA reference pixel in FP64.
template <SarBpComputeType ComputeType, typename voxel_loc_t, typename loose_compute_t>
using SarBpPixelCoord = typename std::conditional<
    // voxel_loc_t is double3/double4 or float3/float4 (asserted).
    cuda::std::is_same_v<cuda::std::remove_cvref_t<decltype(cuda::std::declval<voxel_loc_t>().x)>, double>,
    typename std::conditional<
        ComputeType == SarBpComputeType::Mixed, double,
        typename std::conditional<ComputeType == SarBpComputeType::FloatFloat, fltflt,
                                  loose_compute_t>::type>::type,
    loose_compute_t>::type;

// The single-precision phase paths (the PhaseLUT fractional-bin increment for
// Float, Mixed and FloatFloat, and TaylorFast phase synthesis) evaluate sine and
// cosine with SarBpFastSinCos, whose argument is in "phase units":
// revolutions with __sincos2pif when MATX_HAS_SINCOS2PIF, otherwise radians with
// __sincosf. The host (sar_bp_impl) supplies the phase advances in the same
// units, selected by the same macro. SarBpPhaseUnitsPerRev is the number of
// phase units in one revolution.
inline constexpr double SarBpPhaseUnitsPerRev =
    MATX_HAS_SINCOS2PIF ? 1.0 : 2.0 * cuda::std::numbers::pi_v<double>;

__device__ __forceinline__ void SarBpFastSinCos(float phase, float *sinx, float *cosx)
{
#if MATX_HAS_SINCOS2PIF
    __sincos2pif(phase, sinx, cosx);
#else
    __sincosf(phase, sinx, cosx);
#endif
}

// TaylorFast synthesizes its phase (see below) from the phase advance per range
// bin, rho, in revolutions. It uses the fractional revolutions per bin, frac(rho),
// in fixed point: Q0.64 to form the per-pulse reference ticks (Q0.32) and Q0.32
// for the per-bin increment Q. It also uses the phase advance per bin in single
// precision and phase units (alpha32, see SarBpPhaseUnitsPerRev) for the
// fractional-bin term in the inner pulse loop. As with dr_inv, all three are
// formed on the host, here by MakeFromRevolutions, so the kernel performs no
// conversions or FP64 rounding to obtain them.
struct SarBpTaylorFastPhaseCorrection {
    uint64_t Q64;     // fractional revolutions per bin, Q0.64
    float    alpha32; // phase advance per bin in phase units (SarBpPhaseUnitsPerRev)
    uint32_t Q;       // fractional revolutions per bin, Q0.32

    // rho is the phase advance per range bin in revolutions, and phase_units the
    // same advance in phase units (see SarBpPhaseUnitsPerRev). Both are taken so
    // that the caller controls how each is rounded.
    static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ SarBpTaylorFastPhaseCorrection MakeFromRevolutions(
        double rho, double phase_units)
    {
        // frac(rho) reaches 1 only by rounding when rho is a tiny negative value,
        // and is NaN for a non-finite rho. Map both to 0 (a whole revolution, or
        // no increment) so that the conversion stays in range. Scaling by 2^64 is
        // exact, and the conversion truncates only when frac(rho) < 2^-12, by less
        // than 2^-64 revolutions.
        const double frac = rho - ::floor(rho);
        const uint64_t q64 = (frac < 1.0) ? static_cast<uint64_t>(frac * 0x1p64) : 0;
        // Round Q0.64 to Q0.32. A sum that carries out of 64 bits is a rounded
        // 2^32, which wraps to 0, the phase-equivalent increment. We add half
        // of one Q0.32 step (0x80000000ull) to properly round the last bit of q32.
        const uint32_t q32 = static_cast<uint32_t>((q64 + 0x80000000ull) >> 32);
        return SarBpTaylorFastPhaseCorrection{q64, static_cast<float>(phase_units), q32};
    }

    // As above, with the phase-unit advance derived from rho.
    static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ SarBpTaylorFastPhaseCorrection MakeFromRevolutions(double rho)
    {
        return MakeFromRevolutions(rho, SarBpPhaseUnitsPerRev * rho);
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
    // The uint32 product wraps modulo one revolution for any n, including
    // negative n via the int32 -> uint32 conversion. Reading the tick count as
    // int32 centres it to [-1/2, 1/2) revolutions.
    const int32_t z = static_cast<int32_t>(ref_ticks + c.Q * static_cast<uint32_t>(n));
    // One tick is 2^-32 revolutions. In revolutions the scale is exact; in
    // radians it is a power-of-two rescaling of float(2*pi), so the constant
    // carries only the rounding of pi itself.
    constexpr float units_per_tick = static_cast<float>(SarBpPhaseUnitsPerRev) * 0x1p-32f;
    float sn, cs;
    SarBpFastSinCos(__fmaf_rn(static_cast<float>(z), units_per_tick, c.alpha32 * w), &sn, &cs);
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

// Loose fltflt difference p - a of a pixel coordinate p and an antenna
// coordinate a on the FloatFloat path: TwoSum on the hi words (exact for any
// magnitudes) with the lo words added to its error term and no trailing
// renormalization. The result may be non-canonical after cancellation, which
// the callers' squared-norm accumulation tolerates.
template <typename pixel_coord_t>
__device__ inline fltflt SarBpLooseDiffFF(pixel_coord_t p, fltflt a)
{
    if constexpr (cuda::std::is_same_v<pixel_coord_t, fltflt>) {
        fltflt d = fltflt_two_sum(p.hi, -a.hi);
        d.lo = detail::fadd_rn(d.lo, detail::fadd_rn(p.lo, -a.lo));
        return d;
    } else {
        static_assert(cuda::std::is_same_v<pixel_coord_t, float>,
                      "SarBpLooseDiffFF: pixel coordinates must be float or fltflt");
        fltflt d = fltflt_two_sum(p, -a.hi);
        d.lo = detail::fadd_rn(d.lo, -a.lo);
        return d;
    }
}

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
//
// px/py/pz are fp32 for fp32 voxel locations and fltflt for fp64 voxel
// locations (see SarBpPixelCoord and SarBpLooseDiffFF).
template <bool UseCachedPz2 = false, bool PixelZIsUniform = false, typename pixel_coord_t = float,
          typename IdxT = index_t>
__device__ inline SarBpBinAndWeight<float> ComputeBinWeightToPixelFloatFloat(
    const SarBpPulseBlockCache<SarBpComputeType::FloatFloat, true, PixelZIsUniform> &sh_mem,
    IdxT ip,
    pixel_coord_t px, pixel_coord_t py, [[maybe_unused]] pixel_coord_t pz,
    fltflt dr_inv)
{
    const fltflt apx = sh_mem.ant_pos[ip][0];
    const fltflt apy = sh_mem.ant_pos[ip][1];
    const fltflt mcp_partial = sh_mem.ant_pos[ip][3];

    // Stage 1: loose dx/dy. dz is only formed here when the per-pulse dz^2 is
    // not cached.
    fltflt dx = SarBpLooseDiffFF(px, apx);
    fltflt dy = SarBpLooseDiffFF(py, apy);

    // dz is formed here (unless the per-pulse dz^2 is cached) so that the
    // non-cached path keeps the original dx/dy/dz -> px2/py2/pz2 ordering and
    // codegen. The uniform-z modes always cache (UseCachedPz2), so this branch
    // is the general per-pixel path.
    [[maybe_unused]] fltflt dz{};
    if constexpr (!UseCachedPz2) {
        const fltflt apz = sh_mem.ant_pos[ip][2];
        dz = SarBpLooseDiffFF(pz, apz);
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
    // share the conversion/MUFU pipe with the phase's sine/cosine
    // (SarBpFastSinCos).
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

// pixel_coord_t is the kernel's pixel coordinate type (see SarBpPixelCoord): fp64
// for Mixed with fp64 voxel locations, otherwise loose_compute_t.
template <SarBpComputeType ComputeType, bool UseCachedDz2, bool PixelZIsUniform,
          typename strict_compute_t, typename loose_compute_t, typename pixel_coord_t, typename IdxT>
__device__ inline SarBpBinAndWeight<loose_compute_t> ComputeBinWeightToPixelPulseBlockCache(
    const SarBpPulseBlockCache<ComputeType, true, PixelZIsUniform> &sh_mem,
    IdxT ip,
    pixel_coord_t px,
    pixel_coord_t py,
    [[maybe_unused]] pixel_coord_t pz,
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
    // The z contribution to the squared distance. With a zero or fixed pixel z,
    // slot [2] holds the per-pulse dz^2 (uniform across pixels), so each pixel
    // needs only two fused multiply-adds. With variable z, slot [2] holds the
    // antenna z.
    strict_compute_t r2;
    if constexpr (UseCachedDz2) {
        r2 = SarBpRangeSquaredFromDz2(dx, dy, sh_mem.ant_pos[ip][2]);
    } else {
        const strict_compute_t dz = static_cast<strict_compute_t>(pz) - sh_mem.ant_pos[ip][2];
        r2 = SarBpRangeSquared(dx, dy, dz);
    }
    strict_compute_t dist;
    if constexpr (IsFloat) {
        dist = ::sqrtf(r2);
    } else {
#if __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000
        dist = ::sqrt(r2);
#else
        dist = NewtonRaphsonSqrt(r2);
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

// Number of image pixels each thread accumulates. Each CTA covers a 16 x 16
// pixel tile, 16 x (16 / pixels per thread) threads. Thus, pixels per thread
// must evenly divide the 16 rows of the tile. Thread (x, y) owns the tile's rows y,
// y + blockDim.y, and so on. A thread reads each pulse's values from shared
// memory once for all of its pixels, which halves the shared-memory traffic
// and loop overhead per pixel-pulse.
template <SarBpComputeType ComputeType>
inline constexpr int SarBpPixelsPerThread =
    (ComputeType == SarBpComputeType::TaylorFast || ComputeType == SarBpComputeType::FloatFloat ||
     ComputeType == SarBpComputeType::Mixed) ? 2 : 1;

// Threads per CTA, 16 x 16 / SarBpPixelsPerThread. The tile's 16 rows must
// split evenly among each thread's pixels.
template <SarBpComputeType ComputeType>
constexpr int SarBpThreadsPerCtaFor()
{
    static_assert(16 % SarBpPixelsPerThread<ComputeType> == 0,
                  "SarBp: pixels per thread must evenly divide the 16 rows of a tile");
    return 16 * 16 / SarBpPixelsPerThread<ComputeType>;
}
template <SarBpComputeType ComputeType>
inline constexpr int SarBpThreadsPerCta = SarBpThreadsPerCtaFor<ComputeType>();

// Kernel families with separate launch tuning: one per compute type, plus
// TaylorFast with the third-order term and Mixed with direct sincos instead of
// the phase LUT.
enum class SarBpTunedKernel {
    Double,
    Mixed,                 // phase LUT
    MixedDirect,           // direct sincos
    FloatFloat,
    TaylorFast,            // second order
    TaylorFastThirdOrder,
    Float
};

template <SarBpComputeType ComputeType, bool PhaseLUT, bool TaylorFastAddThirdOrder>
inline constexpr SarBpTunedKernel SarBpTunedKernelFor =
    ComputeType == SarBpComputeType::Mixed      ? (PhaseLUT ? SarBpTunedKernel::Mixed : SarBpTunedKernel::MixedDirect) :
    ComputeType == SarBpComputeType::FloatFloat ? SarBpTunedKernel::FloatFloat :
    ComputeType == SarBpComputeType::TaylorFast ? (TaylorFastAddThirdOrder ? SarBpTunedKernel::TaylorFastThirdOrder
                                                                           : SarBpTunedKernel::TaylorFast) :
    ComputeType == SarBpComputeType::Float      ? SarBpTunedKernel::Float :
                                                  SarBpTunedKernel::Double;

// Launch settings for one kernel family. CTA counts are the minimum resident
// CTAs per SM requested through __launch_bounds__, in CTAs of
// SarBpThreadsPerCta threads, so TaylorFast, FloatFloat and Mixed count
// 128-thread CTAs and Double and Float 256-thread CTAs. Unroll factors apply to the pulse loop. The first CTA count
// and unroll factor apply with zero or fixed pixel z, the VariableZ ones with
// variable pixel z. Every entry states all four values, so an entry never
// inherits one value while overriding another.
template <int CtasV, int VariableZCtasV, int UnrollV, int VariableZUnrollV>
struct SarBpLaunchTuning {
    static constexpr int Ctas = CtasV;
    static constexpr int VariableZCtas = VariableZCtasV;
    static constexpr int Unroll = UnrollV;
    static constexpr int VariableZUnroll = VariableZUnrollV;
};

// All settings below were tuned empirically by timing the sarbp example on
// each GPU; to retune, re-run those benchmarks. SarBpTuningDefaults holds the
// settings for architectures without their own entries (tuned on H100 and
// RTX PRO 6000), and SarBpTuning<Arch, Kernel> overrides them when
// __CUDA_ARCH__ == Arch.
// SarBpLaunchTuning arguments: Ctas, VariableZCtas, Unroll, VariableZUnroll.
template <SarBpTunedKernel Kernel>
struct SarBpTuningDefaults;
template <>
struct SarBpTuningDefaults<SarBpTunedKernel::Double>               : SarBpLaunchTuning<5, 5, 4, 4> {};
template <>
struct SarBpTuningDefaults<SarBpTunedKernel::Mixed>                : SarBpLaunchTuning<9, 9, 2, 2> {};
template <>
struct SarBpTuningDefaults<SarBpTunedKernel::MixedDirect>          : SarBpLaunchTuning<9, 9, 4, 4> {};
template <>
struct SarBpTuningDefaults<SarBpTunedKernel::FloatFloat>           : SarBpLaunchTuning<8, 8, 2, 2> {};
template <>
struct SarBpTuningDefaults<SarBpTunedKernel::TaylorFast>           : SarBpLaunchTuning<8, 8, 4, 4> {};
template <>
struct SarBpTuningDefaults<SarBpTunedKernel::TaylorFastThirdOrder> : SarBpLaunchTuning<8, 8, 2, 2> {};
template <>
struct SarBpTuningDefaults<SarBpTunedKernel::Float>                : SarBpLaunchTuning<6, 6, 4, 4> {};

template <int Arch, SarBpTunedKernel Kernel>
struct SarBpTuning : SarBpTuningDefaults<Kernel> {};

// L40S (sm_89)
template <>
struct SarBpTuning<890, SarBpTunedKernel::TaylorFast>            : SarBpLaunchTuning<8, 7, 4, 4> {};
template <>
struct SarBpTuning<890, SarBpTunedKernel::FloatFloat>            : SarBpLaunchTuning<7, 7, 2, 2> {};

// B200 (sm_100)
template <>
struct SarBpTuning<1000, SarBpTunedKernel::TaylorFast>           : SarBpLaunchTuning<9, 8, 4, 4> {};
template <>
struct SarBpTuning<1000, SarBpTunedKernel::TaylorFastThirdOrder> : SarBpLaunchTuning<9, 8, 2, 4> {};
template <>
struct SarBpTuning<1000, SarBpTunedKernel::FloatFloat>           : SarBpLaunchTuning<7, 7, 4, 4> {};
template <>
struct SarBpTuning<1000, SarBpTunedKernel::Mixed>                : SarBpLaunchTuning<9, 9, 4, 2> {};
template <>
struct SarBpTuning<1000, SarBpTunedKernel::Double>               : SarBpLaunchTuning<5, 5, 16, 16> {};

// B300 (sm_103)
template <>
struct SarBpTuning<1030, SarBpTunedKernel::TaylorFast>           : SarBpLaunchTuning<9, 9, 4, 4> {};
template <>
struct SarBpTuning<1030, SarBpTunedKernel::TaylorFastThirdOrder> : SarBpLaunchTuning<10, 9, 2, 2> {};
template <>
struct SarBpTuning<1030, SarBpTunedKernel::FloatFloat>           : SarBpLaunchTuning<6, 6, 4, 4> {};

// Jetson Thor (sm_110) uses the B300 settings.
template <SarBpTunedKernel Kernel>
struct SarBpTuning<1100, Kernel> : SarBpTuning<1030, Kernel> {};

// Resident threads per SM that the requested CTA count may cover; ptxas ignores
// a __launch_bounds__ minimum CTA count above the architecture's limit. sm_75
// and sm_107 allow 1024 resident threads per SM. The other architectures allow
// 1536 or 2048, and no entry requests more than 1536.
template <int Arch>
inline constexpr int SarBpMaxThreadsPerSm = (Arch == 750 || Arch == 1070) ? 1024 : 1536;

#if defined(__CUDA_ARCH__)
inline constexpr int SarBpTargetArch = __CUDA_ARCH__;
#else
inline constexpr int SarBpTargetArch = 0;
#endif

// Launch settings for one kernel instantiation on the architecture being
// compiled: the minimum CTA count for __launch_bounds__, clamped to what the
// architecture can honor, and the pulse-loop unroll factor.
template <SarBpComputeType ComputeType, SarBpPixelZMode PixelZMode, bool PhaseLUT, bool TaylorFastAddThirdOrder>
struct SarBpLaunchSettings {
    using Tuning = SarBpTuning<SarBpTargetArch, SarBpTunedKernelFor<ComputeType, PhaseLUT, TaylorFastAddThirdOrder>>;
    static constexpr bool VariableZ = PixelZMode == SarBpPixelZMode::Variable;
    static constexpr int RequestedCtas = VariableZ ? Tuning::VariableZCtas : Tuning::Ctas;
    static constexpr int MaxCtas = SarBpMaxThreadsPerSm<SarBpTargetArch> / SarBpThreadsPerCta<ComputeType>;
    static constexpr int MinCtas = RequestedCtas < MaxCtas ? RequestedCtas : MaxCtas;
    static constexpr int PulseUnroll = VariableZ ? Tuning::VariableZUnroll : Tuning::Unroll;
};

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
__launch_bounds__(SarBpThreadsPerCta<ComputeType>, (SarBpLaunchSettings<ComputeType, PixelZMode, PhaseLUT, TaylorFastAddThirdOrder>::MinCtas))
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
    //   - UseCachedDz2: Float/Mixed pulse-block-cache path. Scalar dz^2, so
    //     each pixel forms its squared range with two fused multiply-adds
    //     (SarBpRangeSquaredFromDz2).
    //   - UseCachedPz2FF: FloatFloat path. Fully-compensated fltflt dz^2,
    //     applied to both Zero and Fixed since the compensated-arithmetic z
    //     terms (dz two-sum, z two-prod, 2*dz.hi*dz.lo, dz.lo^2) are the most
    //     expensive part of the inner loop and collapse to a single fltflt add.
    // TaylorFast handles uniform z via its reference-relative offsets
    // (ref_dz == 0, see PixelZIsUniform). The non-cache paths compute dz per
    // pulse in each thread; with more than one pixel per thread, the thread
    // shares that dz^2 among its pixels (SharedDz2).
    constexpr bool UseCachedDz2 =
        PixelZIsUniform && (IsFloat || IsMixed) && PhaseLUT;
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
    // This thread owns pixel rows iy + k * blockDim.y of column ix, for
    // k < PPT. Its first row is the topmost, so if it lies outside the image
    // so do the others: is_valid means the thread has at least one pixel.
    constexpr int PPT = SarBpPixelsPerThread<ComputeType>;
    const index_t ix = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const index_t iy = static_cast<index_t>(blockIdx.y * (blockDim.y * PPT) + threadIdx.y);
    const bool is_valid = ix < image_width && iy < image_height;
    [[maybe_unused]] const auto pixel_valid = [&](int k) {
        return ix < image_width && iy + k * static_cast<index_t>(blockDim.y) < image_height;
    };

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
    // pixel at the CTA's tile origin (thread 0's first pixel, always in-image
    // since the tile origin lies within the image); in-image threads already
    // have the correct z. Taken only by out-of-image threads on edge CTAs.
    if constexpr (NeedFixedPz) {
        if (!is_valid) {
            const index_t origin_x = static_cast<index_t>(blockIdx.x * blockDim.x);
            const index_t origin_y = static_cast<index_t>(blockIdx.y * (blockDim.y * PPT));
            voxel_loc.z = voxel_locations(origin_y, origin_x).z;
        }
    }
    // Pixel coordinates for the range helpers. Mixed and FloatFloat keep fp64
    // voxel locations at fp64 / fltflt precision (see SarBpPixelCoord).
    // TaylorFast does not use these; it forms fp64 offsets from voxel_loc below.
    using pixel_coord_t = SarBpPixelCoord<ComputeType, voxel_loc_t, loose_compute_t>;
    const pixel_coord_t py = static_cast<pixel_coord_t>(voxel_loc.y);
    const pixel_coord_t px = static_cast<pixel_coord_t>(voxel_loc.x);
    // When PixelZIsZero is known, pz folds to a compile-time 0 so the z terms in
    // the range/bin helpers are dropped.
    const pixel_coord_t pz = PixelZIsZero ? static_cast<pixel_coord_t>(0.0f) : static_cast<pixel_coord_t>(voxel_loc.z);
    // FloatFloat and Mixed pass each pixel's own coordinates to their range
    // helpers, so with more than one pixel per thread they are kept per pixel k;
    // pixel 0's are (px, py, pz). A uniform z (Zero or Fixed) is shared by every pixel.
    pixel_coord_t px_k[PPT], py_k[PPT], pz_k[PPT];
    px_k[0] = px;
    py_k[0] = py;
    pz_k[0] = pz;
    #pragma unroll
    for (int k = 1; k < PPT; ++k) {
        if constexpr (!IsTaylorFast) {
            const voxel_loc_t loc = pixel_valid(k) ?
                voxel_locations(iy + k * static_cast<index_t>(blockDim.y), ix) : voxel_loc_t{};
            px_k[k] = static_cast<pixel_coord_t>(loc.x);
            py_k[k] = static_cast<pixel_coord_t>(loc.y);
            pz_k[k] = PixelZIsUniform ? pz : static_cast<pixel_coord_t>(loc.z);
        }
    }

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
            // the host already supplies loose_compute_t, in phase units (see
            // SarBpPhaseUnitsPerRev).
            SarBpFastSinCos(static_cast<loose_compute_t>(phase_correction_scalar) * w, &incr_sinx, &incr_cosx);
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

    SarBpThreadState<ComputeType, PhaseLUT> thread_state[PPT]{};
    if constexpr (IsTaylorFast) {
        static_assert(PhaseLUT == true, "SarBp: TaylorFast is only instantiated with PhaseLUT=true");
        if constexpr (PhaseLUT) {
            if (tid == 0) {
                // Use the midpoint of the CTA's valid image footprint. For
                // partial right/bottom edge CTAs, clamping the nominal CTA
                // center to image_width-1/image_height-1 would choose an edge
                // or corner reference and unnecessarily increase local offsets.
                const index_t block_x0 = static_cast<index_t>(blockIdx.x * blockDim.x);
                const index_t tile_height = static_cast<index_t>(blockDim.y * PPT);
                const index_t block_y0 = static_cast<index_t>(blockIdx.y * (blockDim.y * PPT));
                const index_t valid_width = ((image_width - block_x0) < static_cast<index_t>(blockDim.x)) ?
                    (image_width - block_x0) : static_cast<index_t>(blockDim.x);
                const index_t valid_height = ((image_height - block_y0) < tile_height) ?
                    (image_height - block_y0) : tile_height;
                const index_t ax_c = block_x0 + valid_width / 2;
                const index_t ay_c = block_y0 + valid_height / 2;
                const voxel_loc_t ref_loc = voxel_locations(ay_c, ax_c);
                sh_mem.ref_x = static_cast<strict_compute_t>(ref_loc.x);
                sh_mem.ref_y = static_cast<strict_compute_t>(ref_loc.y);
                sh_mem.ref_z = static_cast<strict_compute_t>(ref_loc.z);
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < PPT; ++k) {
                const bool valid_k = pixel_valid(k);
                const voxel_loc_t loc = (k == 0) ? voxel_loc :
                    (valid_k ? voxel_locations(iy + k * static_cast<index_t>(blockDim.y), ix) : voxel_loc_t{});
                SarBpThreadState<ComputeType, PhaseLUT> &ts = thread_state[k];
                ts.dr_inv_f32 = dr_inv.f32;
                ts.ref_dx = valid_k ? static_cast<float>(static_cast<strict_compute_t>(loc.x) - sh_mem.ref_x) : 0.0f;
                ts.ref_dy = valid_k ? static_cast<float>(static_cast<strict_compute_t>(loc.y) - sh_mem.ref_y) : 0.0f;
                // When the pixel z is uniform across the CTA (Zero: z == 0, or
                // Fixed: z == const), the per-block reference pixel shares that same
                // z, so ref_dz is identically 0 and drops out of ref_dsq and the dot
                // product in ComputeBinWeightToPixelTaylorFast.
                if constexpr (PixelZIsUniform) {
                    ts.ref_dz = 0.0f;
                    ts.ref_dsq = fmaf(ts.ref_dx, ts.ref_dx,
                                                ts.ref_dy * ts.ref_dy);
                } else {
                    ts.ref_dz = valid_k ? static_cast<float>(static_cast<strict_compute_t>(loc.z) - sh_mem.ref_z) : 0.0f;
                    ts.ref_dsq = fmaf(ts.ref_dx, ts.ref_dx,
                                                fmaf(ts.ref_dy, ts.ref_dy,
                                                     ts.ref_dz * ts.ref_dz));
                }
            }
        }
    }

    loose_complex_compute_t accum[PPT]{};
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
                        const fltflt dz = SarBpLooseDiffFF(pz, apz);
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
                        // Zero or fixed pixel z: dz = pz - apz and its square are
                        // per-pulse constants (pz is a compile-time 0 with zero z).
                        // Cache dz^2 in the (otherwise unused after this) antenna-z
                        // slot so the inner loop loads it directly instead of
                        // recomputing the subtract+square for every pixel.
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

        // The row offset advances in the loop header, once per pulse. The unroll
        // factor comes from the per-architecture tuning (SarBpTuning).
        constexpr int PulseUnrollFactor = SarBpLaunchSettings<ComputeType, PixelZMode, PhaseLUT, TaylorFastAddThirdOrder>::PulseUnroll;
        #pragma unroll PulseUnrollFactor
        for (IdxT ip = 0; ip < num_pulses_in_block; ++ip, row_offset += row_step) {
            [[maybe_unused]] const IdxT p = pulse_base + ip;

            // The pulse's values in shared memory are the same for each of this
            // thread's pixels, so they are read once for all of them.
            #pragma unroll
            for (int k = 0; k < PPT; ++k) {
                // The following branches all compute at least the base range bin index and interpolation weight
                // for the current pulse. Paths not using the PhaseLUT optimization will also populate the
                // thread_state[k].range_delta member for use in get_reference_phase_direct().
                SarBpBinAndWeight<loose_compute_t> bin_weight;
                [[maybe_unused]] int32_t bin_floor_local = 0;
                [[maybe_unused]] uint32_t ref_phase_ticks = 0;
                if constexpr (IsFloatFloat) {
                    bin_weight = ComputeBinWeightToPixelFloatFloat<UseCachedPz2FF, PixelZIsUniform>(sh_mem, ip, px_k[k], py_k[k], pz_k[k], dr_inv);
                } else if constexpr (IsTaylorFast) {
                    static_assert(PhaseLUT == true, "SarBp: TaylorFast is only instantiated with PhaseLUT=true");
                    // Zero and Fixed both make ref_dz == 0, so PixelZIsUniform drives the elision.
                    bin_weight = ComputeBinWeightToPixelTaylorFast<
                        TaylorFastAddThirdOrder, PixelZIsUniform>(
                            sh_mem, thread_state[k], ip, bin_floor_local, ref_phase_ticks);
                } else if constexpr (UsePulseBlockCache) {
                    bin_weight = ComputeBinWeightToPixelPulseBlockCache<ComputeType, UseCachedDz2, PixelZIsUniform,
                                                                        strict_compute_t, loose_compute_t>(
                        sh_mem, ip, px_k[k], py_k[k], pz_k[k], dr_inv);
                } else {
                    // Below is the most direct / standard path for backprojection. We compute the distance
                    // from the antenna phase center to the pixel for this pulse and from that the differential range
                    // to the MCP. From that differential range, we compute the base range bin index and interpolation weight
                    // for range profile sampling. Finally, we use the differential range to compute the reference phase
                    // and accumulate the contribution to the pixel. All other paths implement similar logic using various
                    // optimizations.
                    const strict_compute_t range_delta =
                        ComputeRangeToPixel<decltype(pp), ComputeType, strict_compute_t, pixel_coord_t, PixelZIsUniform && (PPT > 1)>(
                            pp, p, px_k[k], py_k[k], pz_k[k]) - static_cast<strict_compute_t>(r_to_mcp(p));
                    if constexpr (!PhaseLUT) {
                        thread_state[k].range_delta = range_delta;
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
                //
                // Out-of-range samples are read from bin 0 and their contribution
                // is discarded below, rather than skipped with a branch. Without
                // the branch, the loads of the unrolled pulses can be issued
                // together and overlap their latency.
                const bool in_range = static_cast<uint32_t>(bin_weight.bin_floor_int) <
                    static_cast<uint32_t>(num_range_bins - 1);
                const int32_t read_bin = in_range ? bin_weight.bin_floor_int : 0;

                // rp accessor picks the fast pointer path on IsUnitStride or
                // falls through to operator().
                range_profiles_t sample_lo, sample_hi;
                if constexpr (UseRowOffset) {
                    const IdxT offset = row_offset + read_bin;
                    sample_lo = rp.at_offset(offset);
                    sample_hi = rp.at_offset(offset + 1);
                } else {
                    sample_lo = rp(p, read_bin);
                    sample_hi = rp(p, read_bin + 1);
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
                        // read_bin keeps the LUT read in bounds; the result is
                        // discarded when the sample is out of range.
                        return get_reference_phase_lut(read_bin, bin_weight.w);
                    } else {
                        return get_reference_phase_direct(thread_state[k].range_delta);
                    }
                }();

                // Select rather than multiply by zero, so that non-finite values in
                // a discarded sample or phase cannot reach the accumulator.
                const loose_complex_compute_t updated = accumulate_contribution(accum[k], sample, ref_phase);
                accum[k] = in_range ? updated : accum[k];
            } // pixel
        } // pulse
    } // pulse block

    #pragma unroll
    for (int k = 0; k < PPT; ++k) {
        if (!pixel_valid(k)) {
            continue;
        }
        const index_t iy_k = iy + k * static_cast<index_t>(blockDim.y);
        // initial_image and output are each touched exactly once per pixel,
        // so they are intentionally not included in the IsUnitStride fast-path
        // check in the transform layer. Direct operator() calls keep the
        // transform's fast-path eligibility as permissive as possible:
        // non-unit-stride or computed-op initial_image (e.g. ConstVal zeros)
        // and non-unit-stride output layouts do not disqualify the fast path
        // for the hot tensors (range_profiles, platform_positions, rtm).
        const initial_image_t initial_image_voxel = initial_image.operator()(iy_k, ix);
        using out_value_t = typename OutImageType::value_type;
        using out_scalar_t = typename out_value_t::value_type;

        output(iy_k, ix) = out_value_t{
            static_cast<out_scalar_t>(initial_image_voxel.real() + accum[k].real()),
            static_cast<out_scalar_t>(initial_image_voxel.imag() + accum[k].imag())
        };
    }
}

#endif // __CUDACC__

}; // namespace matx
