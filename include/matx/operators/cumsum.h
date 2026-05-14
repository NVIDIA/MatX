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

#include <unordered_map>
#include <string>
#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"
#include "matx/transforms/cub.h"
#ifdef MATX_EN_JIT
  #include "matx/transforms/cub_device.h"
#endif

namespace matx {



namespace detail {
  template<typename OpA>
  class CumSumOp : public BaseOp<CumSumOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      cuda::std::array<index_t, OpA::Rank()> out_dims_;
      mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, OpA::Rank()> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;
      mutable bool prerun_done_ = false;  
      mutable ElementsPerThread current_ept_ = ElementsPerThread::ONE;
      mutable int current_groups_per_block_ = 1;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using cumsum_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "cumsum()"; }
      __MATX_INLINE__ CumSumOp(const OpA &a) : a_(a) { 
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
        for (int r = 0; r < Rank(); r++) {
          out_dims_[r] = a_.Size(r);
        }
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

#ifdef MATX_EN_JIT
      struct JIT_Storage {
        typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;
      };

      JIT_Storage ToJITStorage() const {
        return JIT_Storage{detail::to_jit_storage(a_)};
      }

      __MATX_INLINE__ static int LargestValidCubEPTAtMost(index_t value, int limit = 32) {
        if (value <= 0 || limit <= 0 || !cuda::is_power_of_two(value)) {
          return 0;
        }

        const auto capped = cuda::std::min(value, static_cast<index_t>(limit));
        return static_cast<int>(cuda::prev_power_of_two(capped));
      }

      __MATX_INLINE__ index_t CriticalDimSize() const {
        if constexpr (OpA::Rank() == 0) {
          return 0;
        }
        else {
          return a_.Size(OpA::Rank() - 1);
        }
      }

      __MATX_INLINE__ int MaxJitElementsPerThread() const {
        return LargestValidCubEPTAtMost(CriticalDimSize(), MaxCubJitElementsPerThreadByBytes<value_type>());
      }

      __MATX_INLINE__ int BlockThreadsForEPT(int ept) const {
        const auto critical_dim_size = CriticalDimSize();
        if (critical_dim_size <= 0 || ept <= 0 || (critical_dim_size % ept) != 0) {
          return 0;
        }

        const auto block_threads = critical_dim_size / ept;
        return cuda::is_power_of_two(block_threads) ? static_cast<int>(block_threads) : 0;
      }

      __MATX_INLINE__ bool BlockSizeFitsAtMaxEPT() const {
        const auto critical_dim_size = CriticalDimSize();
        if (critical_dim_size <= 0) {
          return false;
        }
        const int max_ept = MaxJitElementsPerThread();
        const int block_threads = BlockThreadsForEPT(max_ept);
        return block_threads > 0 && block_threads <= 1024;
      }

      __MATX_INLINE__ int CurrentBlockThreads() const {
        const int ept = static_cast<int>(current_ept_);
        return BlockThreadsForEPT(ept);
      }

      __MATX_INLINE__ int BatchGroupSize() const {
        if constexpr (OpA::Rank() <= 1) {
          return 1;
        }
        else {
          return static_cast<int>(a_.Size(OpA::Rank() - 2));
        }
      }

      __MATX_INLINE__ int MaxGroupsPerBlock() const {
        return 1;
      }

      __MATX_INLINE__ std::string get_jit_class_name() const {
        std::string symbol_name = "JITCumSumOp_R" + std::to_string(OpA::Rank());
        for (int r = 0; r < OpA::Rank(); r++) {
          symbol_name += "_" + std::to_string(out_dims_[r]);
        }
        return symbol_name;
      }

      __MATX_INLINE__ auto get_jit_op_str() const {
        const std::string class_name = get_jit_class_name();
        return cuda::std::make_tuple(
          class_name,
          std::string("template <typename OpA> struct " + class_name + "  {\n") +
          "  using input_type = typename OpA::value_type;\n" +
          "  using value_type = input_type;\n" +
          "  using matxop = bool;\n" +
          "  constexpr static int Rank_ = " + std::to_string(OpA::Rank()) + ";\n" +
          "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = { " + detail::array_to_string(out_dims_) + " };\n" +
          "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n" +
          "  template <typename CapType, typename... Is>\n" +
          "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const\n" +
          "  {\n" +
          "    return BlockScan<CapType>::Run(a_, indices...);\n" +
          "  }\n" +
          "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() { return Rank_; }\n" +
          "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const { return out_dims_[dim]; }\n" +
          "};\n"
        );
      }
#endif

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
#if defined(__CUDA_ARCH__) && defined(__CUDACC_RTC__)
        return BlockScan<CapType>::Run(a_, indices...);
#else
        return tmp_out_.template operator()<CapType>(indices...);
#endif
      };

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return this->operator()<DefaultCapabilities>(indices...);
      };

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {      
        if constexpr (Cap == OperatorCapability::BLOCK_DIM) {
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
          const int block_threads = CurrentBlockThreads();
          const auto my_cap = cuda::std::array<int, 2>{block_threads, block_threads};
          return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(a_, in));
#else
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
        }
        else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
          static_assert(std::is_same_v<InType, std::unordered_map<std::string, std::string>>, 
                        "JIT_CLASS_QUERY capability requires std::unordered_map<std::string, std::string> as input type");
          const auto [key, value] = get_jit_op_str();
          if (in.find(key) == in.end()) {
            in[key] = value;
          }
          detail::get_operator_capability<Cap>(a_, in);
          return true;
#else
          return false;
#endif
        }
        else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          static_assert(std::is_same_v<remove_cvref_t<InType>, EPTQueryInput>, "ELEMENTS_PER_THREAD capability requires EPTQueryInput as input type");
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
          if (in.jit) {
            const auto max_ept = static_cast<ElementsPerThread>(MaxJitElementsPerThread());
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, max_ept};
            return my_cap;
          }
#endif
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
          bool supported = true;
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
          supported = OpA::Rank() > 0 && !is_complex_v<value_type> &&
                      BlockSizeFitsAtMaxEPT();
#else
          supported = false;
#endif
          return combine_capabilities<Cap>(supported, detail::get_operator_capability<Cap>(a_, in));      
        }
        else if constexpr (Cap == OperatorCapability::SET_ELEMENTS_PER_THREAD) {
#ifdef MATX_EN_JIT
          current_ept_ = in.ept;
#endif
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::GROUPS_PER_BLOCK) {
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
          const int groups = MaxGroupsPerBlock();
          const auto my_cap = cuda::std::array<int, 2>{1, groups};
          return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(a_, in));
#else
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
        }
        else if constexpr (Cap == OperatorCapability::SET_GROUPS_PER_BLOCK) {
#ifdef MATX_EN_JIT
          current_groups_per_block_ = in.groups_per_block;
#endif
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::GLOBAL_KERNEL) {
#ifdef MATX_EN_JIT
          return false;
#else
          return capability_attributes<Cap>::default_value;
#endif
        }
        else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::MAX_EPT_VEC_LOAD) {
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
          return MaxJitElementsPerThread();
#else
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
        }
        else if constexpr (Cap == OperatorCapability::STATIC_SHM_SIZE) {
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
          const int block_threads = CurrentBlockThreads();
          const int self_shm = block_threads > 0 ?
            GetCubBlockShmRequired<value_type>(CubBlockAlgorithm::SCAN,
                                               current_ept_,
                                               block_threads) :
            capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_shm, detail::get_operator_capability<Cap>(a_, in));
#else
          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
        }
        else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
          const auto inner_op_jit_name = detail::get_operator_capability<Cap>(a_, in);
          return get_jit_class_name() + "<" + inner_op_jit_name + ">";
#else
          return "";
#endif
        }
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
        }
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return out_dims_[dim];
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        cumsum_impl(cuda::std::get<0>(out), a_, ex);
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
        if (prerun_done_) {
          return;
        }

        InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));      

        detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

        prerun_done_ = true;
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

  };
}

/**
 * Compute an inclusive cumulative sum (prefix sum) along the innermost dimension
 *
 * Computes an inclusive cumulative sum over the innermost dimension of a tensor.
 * For example, an input tensor of [1, 2, 3, 4] gives the output [1, 3, 6, 10].
 *
 * @tparam InputOperator
 *   Input operator type
 * @param a
 *   Input operator
 * @returns operator with cumulative sum
 */
template <typename InputOperator>
__MATX_INLINE__ auto cumsum(const InputOperator &a) {
  return detail::CumSumOp(a);
}

}
