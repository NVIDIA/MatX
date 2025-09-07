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

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/kernels/sar_bp.cuh"

namespace matx {

  /**
 * @brief SAR Backprojection.
 * 
 * @tparam ImageType Type of input and output image
 * @tparam RangeProfilesType Type of range profiles
 * @tparam PlatPosType Type of platform positions
 * @param initial_image Initial image
 * @param range_profiles Range profiles
 * @param platform_positions Platform positions
 * @param params SAR BP parameters
 * @param stream CUDA stream on which to run the kernel(s)
 */
template <typename OutImageType, typename InitialImageType, typename RangeProfilesType, typename PlatPosType, typename VoxLocType, typename RangeToMcpType>
inline void sar_bp_impl(OutImageType &out, const InitialImageType &initial_image, const RangeProfilesType &range_profiles, const PlatPosType &platform_positions,
  const VoxLocType &voxel_locations, const RangeToMcpType &range_to_mcp, const SarBpParams &params, cudaStream_t stream = 0) {
#ifdef __CUDACC__
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  using image_t = typename OutImageType::value_type;
  using range_profiles_t = typename RangeProfilesType::value_type;
  using plat_pos_t = typename PlatPosType::value_type;

  MATX_STATIC_ASSERT_STR(OutImageType::Rank() == 2, matxInvalidDim, "sar_bp: output image must be a 2D tensor");
  MATX_STATIC_ASSERT_STR(InitialImageType::Rank() == 2, matxInvalidDim, "sar_bp: initial image must be a 2D tensor");
  MATX_STATIC_ASSERT_STR(RangeProfilesType::Rank() == 2, matxInvalidDim, "sar_bp: range profiles must be a 2D tensor");

  constexpr double C = 2.997291625155841e+08; // FIXME: Add to matx/core/constants.h or similar
  const double phase_correction_partial = 4.0 * M_PI * (params.center_frequency / C);
  const double dr_inv = 1.0 / params.del_r;
  printf("DEBUG Op: dr_inv: %e, phase_correction_partial: %e\n", dr_inv, phase_correction_partial);

  const dim3 block(16, 16);
  const dim3 grid(
    static_cast<uint32_t>((out.Size(1) + block.x - 1) / block.x),
    static_cast<uint32_t>((out.Size(0) + block.y - 1) / block.y));

  if (params.compute_type == SarBpComputeType::Double) {
    SarBp<double, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType><<<grid, block, 0, stream>>>(
      out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, dr_inv, phase_correction_partial);
  } else {
    SarBp<float, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType><<<grid, block, 0, stream>>>(
      out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp,
      static_cast<float>(dr_inv), static_cast<float>(phase_correction_partial));
  }
#endif // __CUDACC__
}
} // end namespace matx
