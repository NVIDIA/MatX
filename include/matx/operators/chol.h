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
#include "matx/core/operator_options.h"
#include "matx/transforms/chol/chol_cuda.h"
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
  #include "matx/transforms/solver_cusolverdx.h"
#endif
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/chol/chol_lapack.h"
#endif

namespace matx {
namespace detail {
  template<typename OpA>
  class CholOp : public BaseOp<CholOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      SolverFillMode uplo_;
      mutable detail::tensor_impl_t<typename OpA::value_type, OpA::Rank()> tmp_out_;
      mutable typename OpA::value_type *ptr = nullptr;
      mutable bool prerun_done_ = false;      
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      mutable cuSolverDxHelper<typename OpA::value_type> dx_potrf_helper_;
#endif

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using chol_xform_op = bool;
      using can_alias = bool; // Chol is allowed to use the same input/output memory

      __MATX_INLINE__ std::string str() const { return "chol()"; }
      __MATX_INLINE__ CholOp(const OpA &a, SolverFillMode uplo) : a_(a), uplo_(uplo) {
        MATX_LOG_TRACE("{} constructor: uplo={}", str(), static_cast<int>(uplo));
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
        int major = 0;
        int minor = 0;
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        const int cc = major * 100 + minor * 10;

        if constexpr (OpA::Rank() >= 2) {
          dx_potrf_helper_.set_m(a_.Size(OpA::Rank() - 1));
          dx_potrf_helper_.set_n(a_.Size(OpA::Rank() - 1));
        }
        dx_potrf_helper_.set_cc(cc);
        dx_potrf_helper_.set_function(CUSOLVERDX_FUNCTION_POTRF);
        dx_potrf_helper_.set_fill_mode(uplo_);
#endif
      }

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
        return "JITCholOp";
      }

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      __MATX_INLINE__ auto get_jit_op_str() const {
        const std::string class_name = get_jit_class_name();
        const std::string solver_func_name = std::string(SOLVER_DX_FUNC_PREFIX) + "_" + dx_potrf_helper_.GetSymbolName();
        return cuda::std::make_tuple(
          class_name,
          std::string(
            " extern \"C\" __device__ void " + solver_func_name + "(" + detail::type_to_string<value_type>() + "*, int*);\n" +
            " template <typename OpA> struct " + class_name + "  {\n" +
            "  using value_type = typename OpA::value_type;\n" +
            "  using matxop = bool;\n" +
            "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n" +
            "  template <typename CapType, typename... Is>\n" +
            "  __MATX_INLINE__ __MATX_DEVICE__ value_type operator()(Is... indices) const\n" +
            "  {\n" +
            "    " + dx_potrf_helper_.GetPotrfFuncStr(solver_func_name) + "\n" +
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
          auto result = combine_capabilities<Cap>(dx_potrf_helper_.GetShmRequired(), detail::get_operator_capability<Cap>(a_, in));
          MATX_LOG_DEBUG("cuSolverDx POTRF DYN_SHM_SIZE: {}", result);
          return result;
        }
        else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
          const bool supported = (OpA::Rank() >= 2) &&
                                 (a_.Size(OpA::Rank() - 2) == a_.Size(OpA::Rank() - 1)) &&
                                 dx_potrf_helper_.IsSupported();
          auto result = combine_capabilities<Cap>(supported, detail::get_operator_capability<Cap>(a_, in));
          MATX_LOG_DEBUG("cuSolverDx POTRF SUPPORTS_JIT: {}", result);
          return result;
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
          const auto block_dim = dx_potrf_helper_.GetBlockDim();
          const auto threads = block_dim[0] * cuda::std::max(block_dim[1], 1) * cuda::std::max(block_dim[2], 1);
          const auto my_block = cuda::std::array<int, 2>{threads, threads};
          return combine_capabilities<Cap>(my_block, detail::get_operator_capability<Cap>(a_, in));
        }
        else if constexpr (Cap == OperatorCapability::GENERATE_LTOIR) {
          auto result = combine_capabilities<Cap>(dx_potrf_helper_.GenerateLTOIR(in.ltoir_symbols),
                                                  detail::get_operator_capability<Cap>(a_, in));
          MATX_LOG_DEBUG("cuSolverDx POTRF GENERATE_LTOIR: {}", result);
          return result;
        }
        else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
          const auto inner_op_jit_name = detail::get_operator_capability<Cap>(a_, in);
          auto result = get_jit_class_name() + "<" + inner_op_jit_name + ">";
          MATX_LOG_DEBUG("cuSolverDx POTRF JIT_TYPE_QUERY: {}", result);
          return result;
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

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        chol_impl(cuda::std::get<0>(out),  a_, ex, uplo_);  
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

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
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        matxFree(ptr); 
      }        

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
      }
  };
}

/**
 * Performs a Cholesky factorization, saving the result in either the upper or
 * lower triangle of the output. 
 * 
 * If rank > 2, operations are batched.
 * 
 * @tparam OpA
 *   Data type of input a tensor or operator
 * 
 * @param a
 *   Input tensor or operator of shape `... x n x n`
 * @param uplo
 *   Part of matrix to fill
 * 
 * @return
 *   Operator that produces the factorization output of shape `... x n x n`.
 * 
 */
template<typename OpA>
__MATX_INLINE__ auto chol(const OpA &a, SolverFillMode uplo = SolverFillMode::UPPER) {
  return detail::CholOp(a, uplo);
}

}
