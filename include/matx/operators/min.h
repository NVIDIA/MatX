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
#include "matx/operators/permute.h"
#include "matx/transforms/reduce.h"
#ifdef MATX_EN_JIT
  #include "matx/transforms/cub_device.h"
#endif

namespace matx {



namespace detail {
  template<typename OpA, int ORank>
  class MinOp : public BaseOp<MinOp<OpA, ORank>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      cuda::std::array<index_t, ORank> out_dims_;
      mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, ORank> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;
      mutable bool prerun_done_ = false;
      mutable ElementsPerThread current_ept_ = ElementsPerThread::ONE;
      mutable int current_groups_per_block_ = 1;

    public:
      using matxop = bool;
      using value_type = typename remove_cvref_t<OpA>::value_type;
      using matx_transform_op = bool;
      using min_xform_op = bool;
      using matx_jit_block_reduction = bool;
      using matx_jit_contains_block_reduction = cuda::std::true_type;
      static constexpr int InRank = remove_cvref_t<OpA>::Rank();

      __MATX_INLINE__ std::string str() const { return "min(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ MinOp(const OpA &a) : a_(a) {
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
        for (int r = 0; r < ORank; r++) {
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

      __MATX_INLINE__ index_t ReduceSize() const {
        index_t total = 1;
        for (int r = ORank; r < InRank; r++) {
          total *= a_.Size(r);
        }
        return total;
      }

      __MATX_INLINE__ int MaxJitElementsPerThread() const {
        return CubJitMaxReductionEPT<value_type>(ReduceSize());
      }

      __MATX_INLINE__ bool BlockSizeFitsAtMaxEPT() const {
        return CubJitReductionFitsInBlock<value_type>(ReduceSize());
      }

      __MATX_INLINE__ int CurrentBlockThreads() const {
        return CubJitReductionBlockThreads(ReduceSize(), static_cast<int>(current_ept_));
      }

      __MATX_INLINE__ int MaxGroupsPerBlock() const {
        return 1;
      }

      __MATX_INLINE__ std::string get_jit_class_name() const {
        std::string symbol_name = "JITMinOp_R" + std::to_string(ORank) + "_I" + std::to_string(InRank) + "_N" + std::to_string(ReduceSize());
        for (int r = 0; r < ORank; r++) {
          symbol_name += "_" + std::to_string(out_dims_[r]);
        }
        return symbol_name;
      }

	      __MATX_INLINE__ auto get_jit_op_str() const {
	        const std::string class_name = get_jit_class_name();
	        const bool scalar_loads = detail::get_operator_capability<OperatorCapability::GLOBAL_KERNEL>(a_);
	        return cuda::std::make_tuple(
          class_name,
          std::string("template <typename OpA> struct " + class_name + "  {\n") +
          "  using input_type = typename OpA::value_type;\n" +
          "  using value_type = input_type;\n" +
          "  using matxop = bool;\n" +
          "  using matx_jit_block_reduction = bool;\n" +
          "  using matx_jit_contains_block_reduction = cuda::std::true_type;\n" +
          "  constexpr static int Rank_ = " + std::to_string(ORank) + ";\n" +
          "  constexpr static int ReduceSize_ = " + std::to_string(ReduceSize()) + ";\n" +
          "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = { " + detail::array_to_string(out_dims_) + " };\n" +
          "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n" +
          "  template <typename CapType, typename... Is>\n" +
          "  __MATX_INLINE__ __MATX_DEVICE__ value_type operator()(Is... indices) const\n" +
          "  {\n" +
	          "    return BlockReduce<CapType, BlockReduceType::MIN, ReduceSize_, " + std::string(scalar_loads ? "true" : "false") + ">::RunLastDim(a_, indices...);\n" +
          "  }\n" +
          "  template <typename CapType, typename Out, typename... Is>\n" +
          "  __MATX_INLINE__ __MATX_DEVICE__ value_type Store(Out &out, Is... indices) const\n" +
          "  {\n" +
          "    auto aggregate = this->template operator()<CapType>(indices...);\n" +
          "    if (threadIdx.x == 0) {\n" +
          "      using ScalarCap = CapabilityParams<ElementsPerThread::ONE, true>;\n" +
          "      out.template operator()<ScalarCap>(indices...) = aggregate;\n" +
          "    }\n" +
          "    return aggregate;\n" +
          "  }\n" +
          "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() { return Rank_; }\n" +
          "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const\n" +
          "  {\n" +
          "    if constexpr (Rank_ == 0) { return 1; }\n" +
          "    else { return out_dims_[dim]; }\n" +
          "  }\n" +
          "};\n"
        );
      }
#endif

      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return tmp_out_.template operator()<CapType>(indices...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        return this->operator()<DefaultCapabilities>(indices...);
      }

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
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
	          if (in.jit) {
	            const auto max_ept = static_cast<ElementsPerThread>(MaxJitElementsPerThread());
	            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, max_ept};
	            if (detail::get_operator_capability<OperatorCapability::GLOBAL_KERNEL>(a_)) {
	              return my_cap;
	            }
	            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(a_, in));
	          }
#endif
	          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
	        }
        else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
          bool supported = true;
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
          supported = ((InRank - ORank) >= 1) && !is_complex_v<value_type> &&
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
        else if constexpr (Cap == OperatorCapability::BLOCK_REDUCES_RANK) {
#ifdef MATX_EN_JIT
          return true;
#else
          return capability_attributes<Cap>::default_value;
#endif
        }
        else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
	          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
	        }
	        else if constexpr (Cap == OperatorCapability::MAX_EPT_VEC_LOAD) {
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
	          if (detail::get_operator_capability<OperatorCapability::GLOBAL_KERNEL>(a_)) {
	            return MaxJitElementsPerThread();
	          }
#endif
	          return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
	        }
	        else if constexpr (Cap == OperatorCapability::STATIC_SHM_SIZE) {
#if defined(MATX_EN_JIT) && defined(__CUDACC__)
          const int block_threads = CurrentBlockThreads();
          const int self_shm = block_threads > 0 ?
            GetCubBlockShmRequired<value_type>(CubBlockAlgorithm::REDUCE,
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

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        min_impl(cuda::std::get<0>(out), a_, ex);
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

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return out_dims_[dim];
      }

  };
}

/**
 * Compute min reduction of an operator along axes
 *
 * Returns an operator representing the min of all numbers in the reduction
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
 * @returns Operator with reduced values of min-reduce computed
 */
template <typename InType, int D>
__MATX_INLINE__ auto min(const InType &in, const int (&dims)[D])
{
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::MinOp<decltype(permop), InType::Rank() - D>(permop);
}

template <typename InType, int D>
[[deprecated("Use min() instead of rmin() for reductions")]]
__MATX_INLINE__ auto rmin(const InType &in, const int (&dims)[D])
{
  static_assert(D < InType::Rank(), "reduction dimensions must be <= Rank of input");
  auto perm = detail::getPermuteDims<InType::Rank()>(dims);
  auto permop = permute(in, perm);

  return detail::MinOp<decltype(permop), InType::Rank() - D>(permop);
}

/**
 * Compute min reduction of an operator
 *
 * Returns an operator representing the min of all numbers in the reduction
 *
 * @tparam InType
 *   Input data type
 * @tparam D
 *   Number of right-most dimensions to reduce over
 *
 * @param in
 *   Input data to reduce
 * @returns Operator with reduced values of min-reduce computed
 */
#ifdef DOXYGEN_ONLY
template <typename InType>
#else
template <typename InType, int D = InType::Rank()>
#endif
__MATX_INLINE__ auto min(const InType &in)
{
  return detail::MinOp<decltype(in), InType::Rank() - D>(in);
}

#ifdef DOXYGEN_ONLY
template <typename InType>
#else
template <typename InType, int D = InType::Rank()>
#endif
[[deprecated("Use min() instead of rmin() for reductions")]]
__MATX_INLINE__ auto rmin(const InType &in)
{
  return detail::MinOp<decltype(in), InType::Rank() - D>(in);
}

}
