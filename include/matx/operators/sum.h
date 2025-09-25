////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// sum rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must resumuce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote sumucts derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COpBRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHsum THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <unordered_map>
#include <string>
#include "matx/core/type_utils.h"
#include "matx/operators/permute.h"
#include "matx/transforms/reduce.h"

namespace matx {



namespace detail {
  template<typename OpA, int ORank>
  class SumOp : public BaseOp<SumOp<OpA, ORank>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      cuda::std::array<index_t, ORank> out_dims_;
      mutable ::matx::detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, ORank> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;

    public:
      using matxop = bool;
      using value_type = typename remove_cvref_t<OpA>::value_type;
      using matx_transform_op = bool;
      using sum_xform_op = bool;
      static constexpr int InRank = std::remove_reference_t<OpA>::Rank();
      static_assert(ORank < InRank, "SumOp output rank must be less than input rank");

      __MATX_INLINE__ std::string str() const { return "sum(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ SumOp(const OpA &a) : a_(a) { 
        for (int r = 0; r < ORank; r++) {
          out_dims_[r] = a_.Size(r);
        }
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      __MATX_INLINE__ std::string get_capability_str(int EPT) const {
        return std::string("template <> struct jit_reduce_params_t<0>  {\n") + 
               "  constexpr static int ttl_items = " + std::to_string(a_.Size(InRank - 1) / EPT) + ";\n"
               "};\n";         
      }      

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
#ifdef __CUDA_ARCH__
        if constexpr (CapType::jit) {
          if ((threadIdx.x * CapType::ept) >= Size(Rank() - 1)) {
            return detail::GetJitSentinelValue<CapType, value_type>();
          }
        }
#endif

#if defined(__CUDA_ARCH__) && defined(__CUDACC_RTC__)
        return BlockReduce<CapType, BlockReduceType::SUM>::Run(a_, indices...);
#else
        return tmp_out_.template operator()<CapType>(indices...);
#endif
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return this->operator()<DefaultCapabilities>(indices...);
      }

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {      
        if constexpr (Cap == OperatorCapability::BLOCK_DIM) {
          static_assert(std::is_same_v<InType, BlockSizeQueryInput>, "BLOCK_DIM capability requires BlockSizeQueryInput as input type");
#if defined(MATX_EN_JIT) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)
          return combine_capabilities<Cap>(static_cast<int>(a_.Size(InRank - 1) / static_cast<int>(in.ept)), detail::get_operator_capability<Cap>(a_, in));
#else
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
        }
        else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
          // For JIT_CLASS_QUERY, enforce that InType is std::unordered_map<string, string>
          static_assert(std::is_same_v<InType, std::unordered_map<std::string, std::string>>, 
                        "JIT_CLASS_QUERY capability requires std::unordered_map<std::string, std::string> as input type");
          
          // Get the capability string and add to map
          auto self_cap = get_capability_str(static_cast<int>(ElementsPerThread::ONE));
          // For sum, we need to handle the capability string differently
          // since it's not a tuple. Add it with a generated key.
          std::string key = "JITSumOp_" + std::to_string(a_.Rank());
          if (in.find(key) == in.end()) {
            in[key] = self_cap;
          }
          
          // Also handle child operators
          detail::get_operator_capability<Cap>(a_, in);
          
          // Always return true for now
          return true;
        }
        else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          static_assert(std::is_same_v<InType, EPTQueryInput>, "ELEMENTS_PER_THREAD capability requires EPTQueryInput as input type");
#if defined(MATX_EN_JIT) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)
          if (in.jit) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::THIRTY_TWO};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(a_, in));                
          }
          else {
            return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
          }
#else
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
        }            
        else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
          bool supported = true;
#if defined(MATX_EN_JIT) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)
          const auto scan_size = a_.Size(InRank - 1);     
          if (InRank == 0 || 
              scan_size > 4096 || 
             (scan_size & (scan_size - 1)) != 0) {
            supported = false;
          } 
#else
          supported = false;
#endif
          return combine_capabilities<Cap>(supported, detail::get_operator_capability<Cap>(a_, in));      
        }        
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
        }
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        sum_impl(cuda::std::get<0>(out), a_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return ORank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        matxFree(ptr);
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return out_dims_[dim];
      }

  };
}

/**
 * Compute sum of input along axes
 *
 * Returns a tensor representing the sum of all items in the reduction
 *
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Num of dimensions to reduce over
 *
 * @param in
 *   Input data to reduce
 * @param dims
 *   Array containing dimensions to reduce over
 * @returns Operator with reduced values of sum-reduce computed
 */
template <typename InType, int D>
__MATX_INLINE__ auto sum(const InType &in, const int (&dims)[D])
{
  static_assert(D <= InType::Rank(), "reduction dimensions must be <= Rank of input");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::SumOp<decltype(permop), InType::Rank() - D>(permop);
}

/**
 * Compute sum of input
 *
 * Returns a tensor representing the sum of all items in the reduction
 *
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Number of right-most dimensions to reduce over
 *
 * @param in
 *   Input data to reduce
 * @returns Operator with reduced values of sum-reduce computed
 */
#ifdef DOXYGEN_ONLY
template <typename InType>
#else
template <typename InType, int D = InType::Rank()>
#endif
__MATX_INLINE__ auto sum(const InType &in)
{
  return detail::SumOp<decltype(in), InType::Rank() - D>(in);
}

}
