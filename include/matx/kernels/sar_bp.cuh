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

#define PULSE_BLOCK_SIZE 512

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

__device__ inline fltflt ComputeRangeToPixelFloatFloat(fltflt apx, fltflt apy, fltflt apz, float px, float py, float pz) {
    const fltflt dx = px - apx;
    const fltflt dy = py - apy;
    const fltflt dz = pz - apz;
    return fltflt_norm3d(dx, dy, dz);
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

    static_assert(
        (is_matx_op<RangeToMcpType>() && (RangeToMcpType::Rank() == 0 || RangeToMcpType::Rank() == 1) &&
        (cuda::std::is_same_v<typename RangeToMcpType::value_type, float> || cuda::std::is_same_v<typename RangeToMcpType::value_type, double> || cuda::std::is_same_v<typename RangeToMcpType::value_type, fltflt>)) ||
        (cuda::std::is_same_v<RangeToMcpType, float> || cuda::std::is_same_v<RangeToMcpType, double> || cuda::std::is_same_v<RangeToMcpType, fltflt>),
        "RangeToMcpType must currently be a 0D tensor or scalar of type float, double, or fltflt");

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
    const index_t num_range_bins = range_profiles.Size(1);

    static_assert(cuda::std::is_same_v<voxel_loc_t, double3> || cuda::std::is_same_v<voxel_loc_t, double4> ||
        cuda::std::is_same_v<voxel_loc_t, float3> || cuda::std::is_same_v<voxel_loc_t, float4>, "SarBp: VoxLocType must represent a 2D operator of type double3, double4, float3, or float4");
    // voxel_locations is read once per thread (outside the pulse loop), so it
    // is deliberately not covered by the IsUnitStride fast-path logic. The
    // one-shot cost of its operator() is negligible and its layout is free to
    // be anything the caller wants without blocking fast-path eligibility.
    const voxel_loc_t voxel_loc = is_valid ? voxel_locations(iy, ix) : voxel_loc_t{};
    const loose_compute_t py = voxel_loc.y;
    const loose_compute_t px = voxel_loc.x;
    const loose_compute_t pz = voxel_loc.z;

    // TensorAccessor wraps each hot tensor and picks the fast pointer path
    // when IsUnitStride && is_tensor_view_v<Op>, otherwise forwards to
    // operator(). Binding the tensor's Data() and Stride(0) into the
    // accessor's members forces the compiler to materialize the grid-constant
    // LDCs once at kernel entry rather than reloading inside the pulse loop.
    detail::TensorAccessor<RangeProfilesType, IsUnitStride> rp(range_profiles);
    detail::TensorAccessor<PlatPosType, IsUnitStride> pp(platform_positions);

    // range_to_mcp can be a rank-0 or rank-1 matx op, or a plain scalar.
    // The TensorAccessor's operator() overloads cover the matx-op cases;
    // the scalar case is returned directly by the lambda.
    [[maybe_unused]] const auto rtm_acc = [&]() {
        if constexpr (is_matx_op<RangeToMcpType>()) {
            return detail::TensorAccessor<RangeToMcpType, IsUnitStride>(range_to_mcp);
        } else {
            return int{0};  // placeholder; never dereferenced for scalars
        }
    }();

    const auto r_to_mcp = [&range_to_mcp, rtm_acc](index_t p) -> auto {
        if constexpr (is_matx_op<RangeToMcpType>()) {
            if constexpr (RangeToMcpType::Rank() == 0) {
                return rtm_acc();
            } else {
                return rtm_acc(p);
            }
        } else {
            return range_to_mcp;
        }
    };

    [[maybe_unused]] const loose_compute_t phase_correction_partial_loose = static_cast<loose_compute_t>(phase_correction_partial);
    const auto get_reference_phase = [&phase_lut, &phase_correction_partial, &phase_correction_partial_loose](strict_or_ff_compute_t diffR, index_t bin_floor_int, loose_compute_t w) -> loose_complex_compute_t {
        if constexpr (PhaseLUT) {
            const loose_complex_compute_t base_phase = phase_lut[bin_floor_int];
            float incr_sinx, incr_cosx;
            __sincosf(phase_correction_partial_loose * w, &incr_sinx, &incr_cosx);
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

    [[maybe_unused]] const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    loose_complex_compute_t accum{};
    const loose_compute_t bin_offset = static_cast<loose_compute_t>(0.5) * static_cast<loose_compute_t>(num_range_bins-1);

    const int num_pulse_blocks = (num_pulses + PULSE_BLOCK_SIZE - 1) / PULSE_BLOCK_SIZE;
    for (int block = 0; block < num_pulse_blocks; ++block) {
        const int num_pulses_in_block = num_pulses - block * PULSE_BLOCK_SIZE < PULSE_BLOCK_SIZE ?
            num_pulses - block * PULSE_BLOCK_SIZE : PULSE_BLOCK_SIZE;
        if constexpr (UseSharedPreamble) {
            __syncthreads();
            for (index_t ip = tid; ip < num_pulses_in_block; ip += blockDim.x * blockDim.y) {
                const int p = block * PULSE_BLOCK_SIZE + ip;
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
                    sh_mem.ant_pos[ip][3] = fltflt_fma(neg_rtm, dr_inv, bin_offset);
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
            const int p = block * PULSE_BLOCK_SIZE + ip;
            strict_or_ff_compute_t diffR;
            loose_compute_t w;
            index_t bin_floor_int;
            if constexpr (ComputeType == SarBpComputeType::FloatFloat) {
                // This is just the distance to the pixel rather than the differential range to the MCP.
                // We use diffR because otherwise we would need to initialize diffR to avoid a
                // compiler warning about uninitialized use of diffR.
                diffR = ComputeRangeToPixelFloatFloat(
                    sh_mem.ant_pos[ip][0], sh_mem.ant_pos[ip][1], sh_mem.ant_pos[ip][2], px, py, pz);
                // sh_mem.ant_pos[ip][3] is -mcp * dr_inv + bin_offset, so here we compute
                // dist * dr_inv + (-mcp * dr_inv + bin_offset) = (dist - mcp) * dr_inv + bin_offset
                const fltflt bin = fltflt_fma(diffR, dr_inv, sh_mem.ant_pos[ip][3]);
                float floor_hi = ::floorf(bin.hi);
                float frac = (bin.hi - floor_hi) + bin.lo;
                // bin.lo may push bin over a boundary, in which case floor and frac are incorrect.
                // Compute an adjustment based on whether or not the fractional part is outside (0.0, 1.0).
                const float adjust = ::floorf(frac);  // -1, 0, or 1
                bin_floor_int = static_cast<index_t>(floor_hi + adjust);
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
                bin_floor_int = static_cast<index_t>(bin_floor);
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
                bin_floor_int = static_cast<index_t>(bin_floor);
            }
            if (bin_floor_int >= 0 && bin_floor_int < static_cast<index_t>(num_range_bins-1)) {
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

                accum += sample * ref_phase;
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
        const image_t voxel_contribution {
            initial_image_voxel.real() + accum.real(), initial_image_voxel.imag() + accum.imag() };
        output.operator()(iy, ix) = voxel_contribution;
    }
}

#endif // __CUDACC__

}; // namespace matx