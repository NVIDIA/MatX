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


#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"
#include "matx/transforms/inverse.h"
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
  #include "matx/transforms/solver_cusolverdx.h"
#endif

namespace matx
{

namespace detail {
  template<typename OpA>
  class InvOp : public BaseOp<InvOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, OpA::Rank()> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;
      mutable bool prerun_done_ = false; 
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      mutable cuSolverDxHelper<typename OpA::value_type> dx_gesv_helper_;
#endif

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using inv_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "inv()"; }
      __MATX_INLINE__ InvOp(const OpA &a) : a_(a) {
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
        int major = 0;
        int minor = 0;
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        const int cc = major * 100 + minor * 10;

        if constexpr (OpA::Rank() >= 2) {
          const index_t n = a_.Size(OpA::Rank() - 1);
          dx_gesv_helper_.set_m(n);
          dx_gesv_helper_.set_n(n);
          dx_gesv_helper_.set_k(n);
        }
        dx_gesv_helper_.set_cc(cc);
        dx_gesv_helper_.set_function(CUSOLVERDX_FUNCTION_GESV_PARTIAL_PIVOT);
#endif
      };


      template <typename CapType, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        return tmp_out_.template operator()<CapType>(indices...);
      }

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        return this->operator()<DefaultCapabilities>(indices...);
      }

#ifdef MATX_EN_JIT
      struct JIT_Storage {
        mutable typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;
      };

      JIT_Storage ToJITStorage() const {
        return JIT_Storage{detail::to_jit_storage(a_)};
      }

      __MATX_INLINE__ std::string get_jit_class_name() const {
        return "JITInvOp";
      }

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      __MATX_INLINE__ auto get_jit_op_str() const {
        const std::string class_name = get_jit_class_name();
        const std::string solver_func_name = std::string(SOLVER_DX_FUNC_PREFIX) + "_" + dx_gesv_helper_.GetSymbolName();
        return cuda::std::make_tuple(
          class_name,
          std::string(
            " extern \"C\" __device__ void " + solver_func_name + "(" + detail::type_to_string<value_type>() + "*, int*, " + detail::type_to_string<value_type>() + "*, int*);\n" +
            " template <typename OpA> struct " + class_name + "  {\n" +
            "  using value_type = typename OpA::value_type;\n" +
            "  using matxop = bool;\n" +
            "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n" +
            "  template <typename CapType, typename... Is>\n" +
            "  __MATX_INLINE__ __MATX_DEVICE__ value_type operator()(Is... indices) const\n" +
            "  {\n" +
            "    " + dx_gesv_helper_.GetGesvInverseFuncStr(solver_func_name) + "\n" +
            "  }\n" +
            "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n" +
            "  {\n" +
            "    return " + std::to_string(Rank()) + ";\n" +
            "  }\n" +
            "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const\n" +
            "  {\n" +
            "    return a_.Size(dim);\n" +
            "  }\n" +
            "};\n")
        );
      }
#endif
#endif

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
        if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
          return combine_capabilities<Cap>(dx_gesv_helper_.GetShmRequired(), detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
          const bool supported = (OpA::Rank() >= 2) &&
                                 (a_.Size(OpA::Rank() - 2) == a_.Size(OpA::Rank() - 1)) &&
                                 dx_gesv_helper_.IsSupported();
          return combine_capabilities<Cap>(supported, detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
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
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::GLOBAL_KERNEL) {
          return false;
        }
        else if constexpr (Cap == OperatorCapability::PASS_THROUGH_THREADS) {
          return true;
        }
        else if constexpr (Cap == OperatorCapability::BLOCK_DIM) {
          const auto block_dim = dx_gesv_helper_.GetBlockDim();
          const auto threads = block_dim[0] * cuda::std::max(block_dim[1], 1) * cuda::std::max(block_dim[2], 1);
          const auto my_block = cuda::std::array<int, 2>{threads, threads};
          return combine_capabilities<Cap>(my_block, detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::GENERATE_LTOIR) {
          return combine_capabilities<Cap>(dx_gesv_helper_.GenerateLTOIR(in.ltoir_symbols),
                                           detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
          const auto inner_op_jit_name = detail::get_operator_capability<Cap>(a_, in);
          return get_jit_class_name() + "<" + inner_op_jit_name + ">";
#else
          return std::string{};
#endif
        }
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
        }
#else
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
#endif
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "inv() only supports the CUDA executor currently");
        inv_impl(cuda::std::get<0>(out), a_, ex);
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

        detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), a_.Shape(), &ptr);

        prerun_done_ = true;
        Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
      }      

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }  

        matxFree(ptr);
      }
  };
}

/**
 * Performs a matrix inverse on a square matrix. The inverse API currently uses
 * cuBLAS as a backend with the `cublas<t>matinvBatched()` family of functions
 * for `N <= 32` and `getri/getrf` functions otherwise.
 * 
 * If rank > 2, operations are batched.
 * 
 * @tparam OpA
 *   Data type of input a tensor or operator
 * @tparam ALGO
 *   Algorithm to use for matrix inversion. Currently only suport MAT_INVERSE_ALGO_LU
 * 
 * @param a
 *   Input tensor or operator of shape `... x n x n`
 * 
 * @return
 *   Operator that produces the inverse tensor of shape `... x n x n`.
 * 
 */
template<typename OpA, MatInverseAlgo_t ALGO = MAT_INVERSE_ALGO_LU>
__MATX_INLINE__ auto inv(const OpA &a) {
  return detail::InvOp(a);
}

}
