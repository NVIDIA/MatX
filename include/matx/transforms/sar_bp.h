///////////////////////////////////////////////////////////////////////////////
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

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "matx/core/cache.h"
#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/kernels/sar_bp.cuh"

namespace matx {

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

  const bool phase_lut_optimization = has_feature(params.features, SarBpFeature::PhaseLUTOptimization);
  if (params.compute_type == SarBpComputeType::FloatFloat && ! phase_lut_optimization) {
    // We currently require that phase LUT optimization be enabled for the FloatFloat compute type
    // because we do not yet have float-float based sin/cos implementations. Thus, we would fall back
    // to double-precision sincos() computations, which defeats the purpose of the FloatFloat compute type.
    MATX_THROW(matxInvalidParameter, "sar_bp: FloatFloat compute type requires phase LUT optimization");
  }

  const index_t num_pulses = range_profiles.Size(0);
  if (platform_positions.Size(0) != num_pulses) {
    MATX_THROW(matxInvalidParameter, "sar_bp: number of pulses in range profiles and platform positions must match");
  }
  if constexpr (PlatPosType::Rank() == 2) {
    if (platform_positions.Size(1) != 3) {
      MATX_THROW(matxInvalidParameter, "sar_bp: platform positions must be either a 1D tensor or a 2D tensor with size 3 (x,y,z) in the second dimension");
    }
  }

  const double dr_inv = 1.0 / params.del_r;

  const dim3 block(16, 16);
  const dim3 grid(
    static_cast<uint32_t>((out.Size(1) + block.x - 1) / block.x),
    static_cast<uint32_t>((out.Size(0) + block.y - 1) / block.y));

  // Unit-stride fast path: when every tensor-backed input accessed in the
  // inner loop has stride 1 in its last dimension, the kernel can skip the
  // IMAD-by-1 and the LDC-stride reload the compiler would otherwise emit.
  //
  // The fast path uses .Data(), which only exists on tensor views (types
  // that declare the `tensor_view` marker: tensor_t, tensor_impl_t, and
  // their slices).  Computed operators (broadcasts, clones, zip ops) don't
  // have Data(), so we compile-time gate the IsUnitStride=true kernel
  // instantiation behind a full is_tensor_view_v check.
  //
  // Only tensors accessed inside the hot pulse loop participate in this
  // check: range_profiles, platform_positions, and range_to_mcp. The output
  // image, initial_image, and voxel_locations are each touched once per
  // thread, so their layouts do not block fast-path eligibility and the
  // kernel reads/writes them via direct operator() calls regardless of
  // IsUnitStride. This maximizes the set of input types that benefit from
  // the fast path.
  // range_to_mcp is always a MatX operator (scalar wrapping is the caller's
  // responsibility). Rank-0 ops have no stride; rank-1 ops need a tensor-view
  // with unit stride for the fast path.
  constexpr bool rtm_fast_path_ok =
      RangeToMcpType::Rank() == 0 ||
      is_tensor_view_v<RangeToMcpType>;
  constexpr bool fast_path_eligible =
      is_tensor_view_v<RangeProfilesType> &&
      is_tensor_view_v<PlatPosType> &&
      rtm_fast_path_ok;

  bool is_unit_stride = false;
  if constexpr (fast_path_eligible) {
    is_unit_stride =
        range_profiles.Stride(RangeProfilesType::Rank() - 1) == 1 &&
        platform_positions.Stride(PlatPosType::Rank() - 1) == 1;
    if constexpr (RangeToMcpType::Rank() == 1) {
      is_unit_stride = is_unit_stride && range_to_mcp.Stride(0) == 1;
    }
  }

  auto dispatch = [&](auto is_unit_c) {
    constexpr bool IsUnitStride = decltype(is_unit_c)::value;
    if (phase_lut_optimization) {
      constexpr bool PhaseLUT = true;
      const double phase_correction_partial = 4.0 * M_PI * params.del_r * (params.center_frequency / SPEED_OF_LIGHT);

      const size_t workspace_elem_size = (params.compute_type == SarBpComputeType::Double) ?
        sizeof(cuda::std::complex<double>) : sizeof(cuda::std::complex<float>);
      void *workspace = detail::GetCache().GetStreamAlloc(stream, workspace_elem_size * range_profiles.Size(1));
      const dim3 lut_block(128);
      const dim3 lut_grid(static_cast<uint32_t>((range_profiles.Size(1) + lut_block.x - 1) / lut_block.x));

      if (params.compute_type == SarBpComputeType::Double) {
        cuda::std::complex<double> *phase_lut = static_cast<cuda::std::complex<double> *>(workspace);
        SarBpFillPhaseLUT<double, double><<<lut_grid, lut_block, 0, stream>>>(phase_lut, params.center_frequency, params.del_r, range_profiles.Size(1));
        SarBp<SarBpComputeType::Double, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType, PhaseLUT, IsUnitStride><<<grid, block, 0, stream>>>(
          out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, dr_inv, phase_correction_partial, phase_lut);
      } else if (params.compute_type == SarBpComputeType::Mixed) {
        cuda::std::complex<float> *phase_lut = static_cast<cuda::std::complex<float> *>(workspace);
        SarBpFillPhaseLUT<double, float><<<lut_grid, lut_block, 0, stream>>>(phase_lut, params.center_frequency, params.del_r, range_profiles.Size(1));
        SarBp<SarBpComputeType::Mixed, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType, PhaseLUT, IsUnitStride><<<grid, block, 0, stream>>>(
          out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, dr_inv, phase_correction_partial, phase_lut);
      } else if (params.compute_type == SarBpComputeType::FloatFloat) {
        cuda::std::complex<float> *phase_lut = static_cast<cuda::std::complex<float> *>(workspace);
        SarBpFillPhaseLUT<double, float><<<lut_grid, lut_block, 0, stream>>>(phase_lut, params.center_frequency, params.del_r, range_profiles.Size(1));
        SarBp<SarBpComputeType::FloatFloat, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType, PhaseLUT, IsUnitStride><<<grid, block, 0, stream>>>(
          out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, static_cast<fltflt>(dr_inv), phase_correction_partial, phase_lut);
      } else {
        cuda::std::complex<float> *phase_lut = static_cast<cuda::std::complex<float> *>(workspace);
        SarBpFillPhaseLUT<float, float><<<lut_grid, lut_block, 0, stream>>>(phase_lut, static_cast<float>(params.center_frequency), static_cast<float>(params.del_r), range_profiles.Size(1));
        SarBp<SarBpComputeType::Float, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType, PhaseLUT, IsUnitStride><<<grid, block, 0, stream>>>(
          out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp,
          static_cast<float>(dr_inv), static_cast<float>(phase_correction_partial), phase_lut);
      }
    } else {
      constexpr bool PhaseLUT = false;
      const double phase_correction_partial = 4.0 * M_PI * (params.center_frequency / SPEED_OF_LIGHT);
      if (params.compute_type == SarBpComputeType::Double) {
        SarBp<SarBpComputeType::Double, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType, PhaseLUT, IsUnitStride><<<grid, block, 0, stream>>>(
          out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, dr_inv, phase_correction_partial, nullptr);
      } else if (params.compute_type == SarBpComputeType::Mixed) {
        SarBp<SarBpComputeType::Mixed, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType, PhaseLUT, IsUnitStride><<<grid, block, 0, stream>>>(
          out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, dr_inv, phase_correction_partial, nullptr);
      } else if (params.compute_type == SarBpComputeType::FloatFloat) {
        // We currently require that phase LUT optimization be enabled for the FloatFloat compute type. See comment
        // in run-time check higher in this function.
        MATX_THROW(matxInvalidParameter, "sar_bp: FloatFloat compute type requires phase LUT optimization");
      } else {
        SarBp<SarBpComputeType::Float, OutImageType, InitialImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType, PhaseLUT, IsUnitStride><<<grid, block, 0, stream>>>(
          out, initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp,
          static_cast<float>(dr_inv), static_cast<float>(phase_correction_partial), nullptr);
      }
    }
  };

  // Only instantiate the IsUnitStride=true kernel when the types support it;
  // otherwise forcing that template instantiation would reach .Data() on a
  // non-tensor-view and fail at compile time.
  if constexpr (fast_path_eligible) {
    if (is_unit_stride) {
      dispatch(cuda::std::bool_constant<true>{});
    } else {
      dispatch(cuda::std::bool_constant<false>{});
    }
  } else {
    dispatch(cuda::std::bool_constant<false>{});
  }
#endif // __CUDACC__
}
} // end namespace matx
