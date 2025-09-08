////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
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

namespace matx {

#ifdef __CUDACC__

template <typename PlatPosType, typename strict_compute_t, typename loose_compute_t>
__device__ inline strict_compute_t ComputeRangeToPixel(const PlatPosType &ant_pos, const index_t pulse_idx, const loose_compute_t px, const loose_compute_t py, const loose_compute_t z0) {
    using plat_pos_t = typename PlatPosType::value_type;

    static_assert(std::is_same_v<plat_pos_t, double3> || std::is_same_v<plat_pos_t, double4> ||
        std::is_same_v<plat_pos_t, float3> || std::is_same_v<plat_pos_t, float4>, "ComputeRangeToPixel: plat_pos_t must be a 3D or 4D vector");

    const plat_pos_t ant_pos_p = ant_pos.operator()(pulse_idx);
    const strict_compute_t dx = static_cast<strict_compute_t>(px) - static_cast<strict_compute_t>(ant_pos_p.x);
    const strict_compute_t dy = static_cast<strict_compute_t>(py) - static_cast<strict_compute_t>(ant_pos_p.y);
    const strict_compute_t dz = static_cast<strict_compute_t>(z0) - static_cast<strict_compute_t>(ant_pos_p.z);

    if constexpr (std::is_same_v<strict_compute_t, float>) {
        return ::sqrtf(dx*dx + dy*dy + dz*dz);
    } else {
        return ::sqrt(dx*dx + dy*dy + dz*dz);
    }
}

template <SarBpComputeType ComputeType, typename OutImageType, typename InitialImageType, typename RangeProfilesType, typename PlatPosType, typename VoxLocType, typename RangeToMcpType>
__launch_bounds__(16*16)
__global__ void SarBp(OutImageType output, const InitialImageType initial_image, const __grid_constant__ RangeProfilesType range_profiles, const __grid_constant__ PlatPosType platform_positions, const __grid_constant__ VoxLocType voxel_locations, const __grid_constant__ RangeToMcpType range_to_mcp,
                      typename std::conditional<ComputeType == SarBpComputeType::Double || ComputeType == SarBpComputeType::Mixed, double, float>::type dr_inv,
                      typename std::conditional<ComputeType == SarBpComputeType::Double || ComputeType == SarBpComputeType::Mixed, double, float>::type phase_correction_partial)
{
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
    using loose_compute_t = typename std::conditional<ComputeType == SarBpComputeType::Mixed || ComputeType == SarBpComputeType::Float, float, double>::type;
    using loose_complex_compute_t = cuda::std::complex<loose_compute_t>;

    const index_t num_pulses = range_profiles.Size(0);
    const index_t num_range_bins = range_profiles.Size(1);
    const index_t image_height = output.Size(0);
    const index_t image_width = output.Size(1);
    const index_t ix = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const index_t iy = static_cast<index_t>(blockIdx.y * blockDim.y + threadIdx.y);

    if (ix >= image_width || iy >= image_height) return;

    constexpr loose_compute_t half = static_cast<loose_compute_t>(0.5);
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

    loose_complex_compute_t accum{};
    const loose_compute_t bin_offset = static_cast<loose_compute_t>(0.5) * static_cast<loose_compute_t>(num_range_bins-1);
    const loose_compute_t max_bin_f = static_cast<loose_compute_t>(num_range_bins) - static_cast<loose_compute_t>(2.0);
    #pragma unroll 4
    for (index_t p = 0; p < num_pulses; ++p) {
        const strict_compute_t diffR =
            ComputeRangeToPixel<PlatPosType, strict_compute_t, loose_compute_t>(platform_positions, p, px, py, pz) - r_to_mcp(p);
        const loose_compute_t bin = static_cast<loose_compute_t>(diffR * dr_inv) + bin_offset;
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
                (static_cast<loose_compute_t>(1.0) - w) * sample_lo + w * sample_hi;

            strict_compute_t sinx, cosx;
            if constexpr (std::is_same_v<strict_compute_t, double>) {
                ::sincos(phase_correction_partial * diffR, &sinx, &cosx);
            } else {
                ::sincosf(phase_correction_partial * diffR, &sinx, &cosx);
                //__sincosf(phase_correction_partial * diffR, &sinx, &cosx);
            }
            const loose_complex_compute_t matched_filter{
                static_cast<loose_compute_t>(cosx), static_cast<loose_compute_t>(sinx)};

            accum += sample * matched_filter;
        }
    }

    initial_image_t initial_image_voxel = initial_image.operator()(iy, ix);
    const image_t voxel_contribution {
        initial_image_voxel.real() + accum.real(), initial_image_voxel.imag() + accum.imag() };
    cuda::std::apply([voxel_contribution, &output](auto &&...args) {
        output.operator()(args...) = voxel_contribution;
    }, cuda::std::make_tuple(iy, ix));
}

#endif // __CUDACC__

}; // namespace matx