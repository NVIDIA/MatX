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

#define PULSE_BLOCK_SIZE 1024

namespace matx {

static constexpr double SPEED_OF_LIGHT = 2.997291625155841e+08;

#ifdef __CUDACC__

// Use two iterations of Newton-Raphson to compute the square root of a double. The
// initial estimate is computed using a single-precision square root.
static __device__ __forceinline__ double NewtonRaphsonSqrt(double x) {
    const float est = sqrtf(static_cast<float>(x));
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
    const fltflt dx = fltflt_sub(fltflt_make_from_float(px), apx);
    const fltflt dy = fltflt_sub(fltflt_make_from_float(py), apy);
    const fltflt dz = fltflt_sub(fltflt_make_from_float(pz), apz);
    const fltflt dist = fltflt_add(fltflt_add(fltflt_mul(dx, dx), fltflt_mul(dy, dy)), fltflt_mul(dz, dz));
    return fltflt_sqrt(dist);
}

template <typename PlatPosType, SarBpComputeType ComputeType, typename strict_compute_t, typename loose_compute_t>
__device__ inline strict_compute_t ComputeRangeToPixel(const PlatPosType &ant_pos, const index_t pulse_idx, const loose_compute_t px, const loose_compute_t py, const loose_compute_t z0) {
    using plat_pos_t = typename PlatPosType::value_type;

    static_assert(std::is_same_v<plat_pos_t, double3> || std::is_same_v<plat_pos_t, double4> ||
        std::is_same_v<plat_pos_t, float3> || std::is_same_v<plat_pos_t, float4>, "ComputeRangeToPixel: plat_pos_t must be a 3D or 4D vector");
    const plat_pos_t ant_pos_p = ant_pos.operator()(pulse_idx);
    const strict_compute_t dx = static_cast<strict_compute_t>(px) - static_cast<strict_compute_t>(ant_pos_p.x);
    const strict_compute_t dy = static_cast<strict_compute_t>(py) - static_cast<strict_compute_t>(ant_pos_p.y);
    const strict_compute_t dz = static_cast<strict_compute_t>(z0) - static_cast<strict_compute_t>(ant_pos_p.z);

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
    if constexpr (std::is_same_v<ComputeType, float>) {
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
using loose_compute_param_t = typename std::conditional<ComputeType == SarBpComputeType::Double, double, float>::type;

template <SarBpComputeType ComputeType, typename OutImageType, typename InitialImageType, typename RangeProfilesType, typename PlatPosType, typename VoxLocType, typename RangeToMcpType, bool PhaseLUT>
__launch_bounds__(16*16)
__global__ void SarBp(OutImageType output, const InitialImageType initial_image, const __grid_constant__ RangeProfilesType range_profiles, const __grid_constant__ PlatPosType platform_positions, const __grid_constant__ VoxLocType voxel_locations, const __grid_constant__ RangeToMcpType range_to_mcp,
                      strict_compute_param_t<ComputeType> dr_inv,
                      strict_compute_param_t<ComputeType> phase_correction_partial,
                      cuda::std::complex<loose_compute_param_t<ComputeType>> *phase_lut)
{
    static_assert(OutImageType::Rank() == 2, "Output image must be a 2D tensor");
    static_assert(InitialImageType::Rank() == 2, "Initial image must be a 2D tensor");
    static_assert(RangeProfilesType::Rank() == 2, "Range profiles must be a 2D tensor");
    static_assert(PlatPosType::Rank() == 1, "Platform positions must be a 1D tensor");
    static_assert(VoxLocType::Rank() == 2, "Voxel locations must be a 2D tensor");
    static_assert(is_complex_v<typename OutImageType::value_type>, "Output image must be complex");
    static_assert(is_complex_v<typename InitialImageType::value_type>, "Initial image must be complex");
    static_assert(is_complex_v<typename RangeProfilesType::value_type>, "Range profiles must be complex");

    static_assert(
        (is_matx_op<RangeToMcpType>() && (RangeToMcpType::Rank() == 0 || RangeToMcpType::Rank() == 1) && (std::is_same_v<typename RangeToMcpType::value_type, float> || std::is_same_v<typename RangeToMcpType::value_type, double>)) ||
        (std::is_same_v<RangeToMcpType, float> || std::is_same_v<RangeToMcpType, double>),
        "RangeToMcpType must currently be a 0D tensor or scalar of type float or double");

    using initial_image_t = typename InitialImageType::value_type;
    using image_t = typename OutImageType::value_type;
    using range_profiles_t = typename RangeProfilesType::value_type;
    using plat_pos_t = typename PlatPosType::value_type;
    using voxel_loc_t = typename VoxLocType::value_type;
    using compute_t = typename std::conditional<ComputeType == SarBpComputeType::Double, double, float>::type;
    using strict_compute_t = typename std::conditional<ComputeType == SarBpComputeType::Double || ComputeType == SarBpComputeType::Mixed, double, float>::type;
    using strict_complex_compute_t = cuda::std::complex<strict_compute_t>;
    using loose_compute_t = typename std::conditional<ComputeType == SarBpComputeType::Double, double, float>::type;
    using loose_complex_compute_t = cuda::std::complex<loose_compute_t>;

    const index_t image_height = output.Size(0);
    const index_t image_width = output.Size(1);
    const index_t ix = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const index_t iy = static_cast<index_t>(blockIdx.y * blockDim.y + threadIdx.y);

    // Currently only used for FloatFloat compute type
    __shared__ fltflt sh_ant_pos[PULSE_BLOCK_SIZE][4];

    const bool is_valid = ix < image_width && iy < image_height;
    if constexpr (ComputeType != SarBpComputeType::FloatFloat) {
        // For the FloatFloat ComputeType, keep all threads active to participate in CTA-wide
        // antenna position loads
        if (! is_valid) return;
    }

    const index_t num_pulses = range_profiles.Size(0);
    const index_t num_range_bins = range_profiles.Size(1);

    constexpr loose_compute_t half = static_cast<loose_compute_t>(0.5);
    static_assert(std::is_same_v<voxel_loc_t, double3> || std::is_same_v<voxel_loc_t, double4> ||
        std::is_same_v<voxel_loc_t, float3> || std::is_same_v<voxel_loc_t, float4>, "SarBp: VoxLocType must represent a 2D operator of type double3, double4, float3, or float4");
    const voxel_loc_t voxel_loc = voxel_locations(iy, ix);
    const loose_compute_t py = voxel_loc.y;
    const loose_compute_t px = voxel_loc.x;
    const loose_compute_t pz = voxel_loc.z;

    const auto r_to_mcp = [&range_to_mcp](index_t p) -> auto {
        if constexpr (is_matx_op<RangeToMcpType>()) {
            if constexpr (RangeToMcpType::Rank() == 0) {
                return range_to_mcp();
            } else {
                return range_to_mcp(p);
            }
        } else {
            return range_to_mcp;
        }
    };

    const loose_compute_t phase_correction_partial_loose = static_cast<loose_compute_t>(phase_correction_partial);
    const auto get_reference_phase = [&phase_lut, &phase_correction_partial, &phase_correction_partial_loose](strict_compute_t diffR, index_t bin_floor_int, loose_compute_t w) -> loose_complex_compute_t {
        if constexpr (PhaseLUT) {
            const loose_complex_compute_t base_phase = phase_lut[bin_floor_int];
            float incr_sinx, incr_cosx;
            __sincosf(phase_correction_partial_loose * w, &incr_sinx, &incr_cosx);
            return loose_complex_compute_t{
                base_phase.real() * incr_cosx - base_phase.imag() * incr_sinx,
                base_phase.real() * incr_sinx + base_phase.imag() * incr_cosx
            };
        } else {
            strict_compute_t sinx, cosx;
            if constexpr (std::is_same_v<strict_compute_t, double>) {
                ::sincos(phase_correction_partial * diffR, &sinx, &cosx);
            } else {
                ::sincosf(phase_correction_partial * diffR, &sinx, &cosx);
            }
            return loose_complex_compute_t{
                static_cast<loose_compute_t>(cosx), static_cast<loose_compute_t>(sinx)
            };
        }
    };

    [[maybe_unused]] fltflt dr_inv_fltflt{};
    if constexpr (ComputeType == SarBpComputeType::FloatFloat) {
        dr_inv_fltflt = fltflt_make_from_double(dr_inv);
    }
    [[maybe_unused]] const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    loose_complex_compute_t accum{};
    const loose_compute_t bin_offset = static_cast<loose_compute_t>(0.5) * static_cast<loose_compute_t>(num_range_bins-1);
    const loose_compute_t max_bin_f = static_cast<loose_compute_t>(num_range_bins) - static_cast<loose_compute_t>(2.0);
    const int num_pulse_blocks = (num_pulses + PULSE_BLOCK_SIZE - 1) / PULSE_BLOCK_SIZE;
    for (int block = 0; block < num_pulse_blocks; ++block) {
    const int num_pulses_in_block = num_pulses - block * PULSE_BLOCK_SIZE < PULSE_BLOCK_SIZE ?
        num_pulses - block * PULSE_BLOCK_SIZE : PULSE_BLOCK_SIZE;
    if constexpr (ComputeType == SarBpComputeType::FloatFloat) {
        __syncthreads();
        for (index_t ip = tid; ip < num_pulses_in_block; ip += blockDim.x * blockDim.y) {
            const int p = block * PULSE_BLOCK_SIZE + ip;
            const plat_pos_t ant_pos_p = platform_positions.operator()(p);
            sh_ant_pos[ip][0] = fltflt_make_from_double(ant_pos_p.x);
            sh_ant_pos[ip][1] = fltflt_make_from_double(ant_pos_p.y);
            sh_ant_pos[ip][2] = fltflt_make_from_double(ant_pos_p.z);
            sh_ant_pos[ip][3] = fltflt_make_from_double(r_to_mcp(p));
        }
        __syncthreads();
        if (! is_valid) {
            continue;
        }
    }
    #pragma unroll 4
    for (index_t ip = 0; ip < num_pulses_in_block; ++ip) {
        const int p = block * PULSE_BLOCK_SIZE + ip;
        [[maybe_unused]] strict_compute_t diffR{};
        loose_compute_t bin;
        if constexpr (ComputeType == SarBpComputeType::FloatFloat) {
            const fltflt diffR_ff = fltflt_sub(ComputeRangeToPixelFloatFloat(
                sh_ant_pos[ip][0], sh_ant_pos[ip][1], sh_ant_pos[ip][2], px, py, pz), sh_ant_pos[ip][3]);
            bin = static_cast<loose_compute_t>(
                fltflt_to_float(fltflt_mul(diffR_ff, dr_inv_fltflt)) + bin_offset);
            // diffR is otherwise unused for FloatFloat and thus not set
        } else {
            diffR = ComputeRangeToPixel<PlatPosType, ComputeType, strict_compute_t, loose_compute_t>(
                platform_positions, p, px, py, pz) - r_to_mcp(p);
            bin = static_cast<loose_compute_t>(diffR * dr_inv) + bin_offset;
        }
        if (bin >= 0.0f && bin < max_bin_f) {
            loose_compute_t bin_floor, w;
            if constexpr (std::is_same_v<loose_compute_t, float>) {
                bin_floor = ::floorf(bin);
                w = (bin - bin_floor);
            } else {
                bin_floor = ::floor(bin);
                w = (bin - bin_floor);
            }
            const index_t bin_floor_int = static_cast<index_t>(bin_floor);

            range_profiles_t sample_lo, sample_hi;

            cuda::std::apply([&sample_lo, &range_profiles](auto &&...args) {
                sample_lo = range_profiles.operator()(args...);
            }, cuda::std::make_tuple(p, bin_floor_int));

            cuda::std::apply([&sample_hi, &range_profiles](auto &&...args) {
                sample_hi = range_profiles.operator()(args...);
            }, cuda::std::make_tuple(p, bin_floor_int + 1));

            const loose_complex_compute_t sample =
                (static_cast<loose_compute_t>(1.0) - w) * static_cast<loose_complex_compute_t>(sample_lo) +
                w * static_cast<loose_complex_compute_t>(sample_hi);

            const loose_complex_compute_t ref_phase = get_reference_phase(diffR, bin_floor_int, w);

            accum += sample * ref_phase;
        }
    }
}

    if (is_valid) {
        initial_image_t initial_image_voxel = initial_image.operator()(iy, ix);
        const image_t voxel_contribution {
            initial_image_voxel.real() + accum.real(), initial_image_voxel.imag() + accum.imag() };
        cuda::std::apply([voxel_contribution, &output](auto &&...args) {
            output.operator()(args...) = voxel_contribution;
        }, cuda::std::make_tuple(iy, ix));
    }
}

#endif // __CUDACC__

}; // namespace matx