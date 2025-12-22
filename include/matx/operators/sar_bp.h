////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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
// THIS SOFTWARE IS PROVIDED BY THE COpBRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx {

enum class SarBpComputeType {
  Double,
  Mixed,
  // The FloatFloat compute type combines mixed precision (i.e., fp32 when possible) with a
  // float-float handling of the values for which fp64 would otherwise be needed. The float-float
  // representation offers increased precision relative to fp32, but not full fp64 precision,
  // through the use of increased fp32 computation and representing each value as an unevaluated
  // sum of two fp32 components.
  FloatFloat,
  Float
};

// Features that can be enabled or disabled for the SAR BP kernel. These features are enabled by setting
// the corresponding bit in the SarBpParams::features field.
enum class SarBpFeature : uint32_t {
  None = 0x0,
  PhaseLUTOptimization = 0x1,
  // Add more features here as needed
};

// Enable bitmask operations for SarBpFeature
constexpr SarBpFeature operator|(SarBpFeature lhs, SarBpFeature rhs) noexcept {
  return static_cast<SarBpFeature>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr SarBpFeature operator&(SarBpFeature lhs, SarBpFeature rhs) noexcept {
  return static_cast<SarBpFeature>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

constexpr SarBpFeature& operator|=(SarBpFeature& lhs, SarBpFeature rhs) noexcept {
  return lhs = lhs | rhs;
}

// Helper function to test if a feature is enabled
constexpr bool has_feature(SarBpFeature features, SarBpFeature feature) noexcept {
  return (features & feature) != SarBpFeature::None;
}

struct SarBpParams {
  // compute_type controls the floating point precision of intermediate calculations in the SAR BP kernel.
  // While the inputs (range profiles, antenna positions, etc.) and output (image) have their own data
  // types, we may wish to use a different precision for the internal calculations. For example, the
  // output may be cuda::std::complex<float> while the intermediate calculations are done in double.
  SarBpComputeType compute_type{SarBpComputeType::Double};
  SarBpFeature features{SarBpFeature::None};
  double center_frequency{0.0};
  double del_r{0.0};
};

}

#include "matx/transforms/sar_bp.h"

namespace matx {

namespace detail {
  template<typename ImageType, typename RangeProfilesType, typename PlatPosType, typename VoxLocType, typename RangeToMcpType>
  class SarBpOp : public BaseOp<SarBpOp<ImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType>>
  {
    static_assert(is_complex_v<typename RangeProfilesType::value_type>, "Phase history must be complex");
    static_assert(is_complex_v<typename ImageType::value_type>, "Image must be complex");
    static_assert(RangeProfilesType::Rank() == 2, "Phase history must be a 2D tensor");
    static_assert(ImageType::Rank() == 2, "Image must be a 2D tensor");
    private:
      using out_t = typename detail::base_type_t<ImageType>;
      SarBpParams params_;
      ImageType initial_image_;
      RangeProfilesType range_profiles_;
      PlatPosType platform_positions_;
      VoxLocType voxel_locations_;
      RangeToMcpType range_to_mcp_;
      cuda::std::array<index_t, RangeProfilesType::Rank()> out_dims_;
      mutable detail::tensor_impl_t<out_t, RangeProfilesType::Rank()> tmp_out_;
      mutable out_t *ptr = nullptr;

    public:
      using matxop = bool;
      using matx_transform_op = bool;
      using sar_bp_xform_op = bool;
      using value_type = out_t;

      __MATX_INLINE__ std::string str() const { return "sar_bp(" + get_type_str(range_profiles_) + ")";}
      __MATX_INLINE__ SarBpOp(const ImageType &initial_image, const RangeProfilesType &range_profiles, const PlatPosType &platform_positions, const VoxLocType &voxel_locations, const RangeToMcpType &range_to_mcp, const SarBpParams &params) :
          initial_image_(initial_image), range_profiles_(range_profiles), platform_positions_(platform_positions), voxel_locations_(voxel_locations), range_to_mcp_(range_to_mcp), params_(params)
      { 
        for (int r = 0; r < ImageType::Rank(); r++) {
          out_dims_[r] = initial_image.Size(r);
        }
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      // Const versions
      template <ElementsPerThread EPT, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return tmp_out_.template operator()<EPT>(indices...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return this->operator()<detail::ElementsPerThread::ONE>(indices...);
      }


      template <OperatorCapability Cap>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, 
                                           detail::get_operator_capability<Cap>(initial_image_),
                                           detail::get_operator_capability<Cap>(range_profiles_),
                                           detail::get_operator_capability<Cap>(platform_positions_),
                                           detail::get_operator_capability<Cap>(range_to_mcp_));
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "sarbp() only supports the CUDA executor currently");

        sar_bp_impl(cuda::std::get<0>(out), initial_image_, range_profiles_, platform_positions_, voxel_locations_, range_to_mcp_, params_, ex.getStream());
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return ImageType::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<ImageType>()) {
          initial_image_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }     

        if constexpr (is_matx_op<RangeProfilesType>()) {
          range_profiles_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<PlatPosType>()) {
          platform_positions_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }      

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));           

        detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

        Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<ImageType>()) {
          initial_image_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }     

        if constexpr (is_matx_op<RangeProfilesType>()) {
          range_profiles_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        } 

        if constexpr (is_matx_op<PlatPosType>()) {
          platform_positions_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        matxFree(ptr);
      }        

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return out_dims_[dim];
      }

  };
}

namespace experimental {
template <typename ImageType, typename RangeProfilesType, typename PlatPosType, typename VoxLocType, typename RangeToMcpType>
inline auto sar_bp(const ImageType &initial_image, const RangeProfilesType &range_profiles,
    const PlatPosType &platform_positions, const VoxLocType &voxel_locations, const RangeToMcpType &range_to_mcp, const SarBpParams &params) {
  return matx::detail::SarBpOp<ImageType, RangeProfilesType, PlatPosType, VoxLocType, RangeToMcpType>(initial_image, range_profiles, platform_positions, voxel_locations, range_to_mcp, params);
}

} // end namespace experimental
} // end namespace matx
