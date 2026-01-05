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
#include "matx/core/log.h"
#include "matx/transforms/matmul/matmul_cuda.h"
#include "matx/transforms/matmul/matmul_cusparse.h"
#ifdef MATX_EN_CPU_MATMUL
  #include "matx/transforms/matmul/matmul_cblas.h"
#endif
#include <cuda/std/__algorithm/max.h>

#if defined(MATX_EN_MATHDX) && defined (__CUDACC__)
  #include "matx/transforms/matmul/matmul_cublasdx.h"
#endif

namespace matx
{
  namespace detail {
    template <typename OpA, typename OpB, typename PermDims>
    class MatMulOp : public BaseOp<MatMulOp<OpA, OpB, PermDims>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        typename detail::base_type_t<OpB> b_;
        float alpha_;
        float beta_;
        PermDims perm_; 
        static constexpr int out_rank = cuda::std::max(OpA::Rank(), OpB::Rank());
        cuda::std::array<index_t, out_rank> out_dims_;
        // This should be tensor_impl_t, but need to work around issues with temp types returned in matmul
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, out_rank> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;
        mutable bool prerun_done_ = false;
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
        mutable cuBLASDxHelper<typename OpA::value_type> dx_gemm_helper_;
#endif 

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;
        using matx_transform_op = bool;
        using matmul_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
            return "matmul(" + get_type_str(a_) + "," + get_type_str(b_) + ")";
        }

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<OpB>> b_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(a_), detail::to_jit_storage(b_)};
        }
#endif           

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string symbol_name = "JITMatMulOp_";
          symbol_name += std::to_string(dx_gemm_helper_.get_m());
          symbol_name += "_";
          symbol_name += std::to_string(dx_gemm_helper_.get_n());
          symbol_name += "_";
          symbol_name += std::to_string(dx_gemm_helper_.get_k());
          symbol_name += dx_gemm_helper_.get_is_complex() ? "_C" : "_R";
          return symbol_name;
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          const std::string class_name = get_jit_class_name();
          const std::string gemm_func_name = std::string(GEMM_DX_FUNC_PREFIX) + "_" + dx_gemm_helper_.GetSymbolName();
          
          return cuda::std::make_tuple(
             class_name, 
             std::string(
                 " extern \"C\" __device__ void " + gemm_func_name + "(" + 
                 detail::type_to_string<typename OpA::value_type>() + "*, " +
                 detail::type_to_string<typename OpA::value_type>() + "*, " +
                 detail::type_to_string<typename OpA::value_type>() + "*, " +
                 detail::type_to_string<typename OpA::value_type>() + "*, " +
                 detail::type_to_string<typename OpA::value_type>() + "*);\n" +
                 " template <typename OpA, typename OpB> struct " + class_name + " {\n" +
                 "  using input_type = typename OpA::value_type;\n" +
                 "  using matxop = bool;\n" +
                 "  using value_type = input_type;\n" +
                 "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n" +
                 "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpB>> b_;\n" +
                 "  constexpr static cuda::std::array<index_t, " + std::to_string(Rank()) + "> out_dims_ = { " + 
                 detail::array_to_string(out_dims_) + " };\n" +
                 "  static constexpr index_t m_ = " + std::to_string(dx_gemm_helper_.get_m()) + ";\n" +
                 "  static constexpr index_t n_ = " + std::to_string(dx_gemm_helper_.get_n()) + ";\n" +
                 "  static constexpr index_t k_ = " + std::to_string(dx_gemm_helper_.get_k()) + ";\n" +
                 "  template <typename CapType, typename... Is>\n" +
                 "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n" +
                 "  {\n" +
                 "    " + dx_gemm_helper_.GetFuncStr(gemm_func_name, alpha_, beta_) + "\n" +
                 "  }\n" +
                 "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n" +
                 "  {\n" +
                 "    return " + std::to_string(Rank()) + ";\n" +
                 "  }\n" +
                 "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const\n" +
                 "  {\n" +
                 "    return out_dims_[dim];\n" +
                 "  }\n" +    
                 "};\n")
          );
        }
#endif

        __MATX_INLINE__ MatMulOp(const OpA &a, const OpB &b, float alpha, float beta, PermDims perm) : 
              a_(a), b_(b), alpha_(alpha), beta_(beta), perm_(perm) {
          MATX_LOG_TRACE("{} constructor: alpha={}, beta={}", str(), alpha, beta);
          if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
            for (int r = 0; r < Rank(); r++) {
              if (r == Rank() - 2) {
                out_dims_[perm_[r]] = a_.Size(r);
              }
              else if (r == Rank() - 1) {
                out_dims_[perm_[r]] = b_.Size(r);
              }
              else {
                out_dims_[perm_[r]] = OpA::Rank() > OpB::Rank() ? a_.Size(r) : b_.Size(r);
              }
            }
          }
          else {
            for (int r = 0; r < Rank() - 2; r++) {
              out_dims_[(size_t)r] = OpA::Rank() > OpB::Rank() ? a_.Size(r) : b_.Size(r);
            }

            out_dims_[Rank() - 2] = a_.Size(OpA::Rank() - 2);
            out_dims_[Rank() - 1] = b_.Size(OpB::Rank() - 1);
          }

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
          // Initialize cuBLASDx helper with matrix dimensions
          // For GEMM: C(m x n) = A(m x k) * B(k x n)
          // m = rows of output (from A's second-to-last dim)
          // n = cols of output (from B's last dim)
          // k = inner dimension (A's last dim = B's second-to-last dim)
          int major = 0;
          int minor = 0;
          int device;
          cudaGetDevice(&device);
          cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
          cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
          int cc = major * 100 + minor;  // Compute capability as three digits (e.g., 890 for SM 8.9)
          
          dx_gemm_helper_.set_m(a_.Size(OpA::Rank() - 2));
          dx_gemm_helper_.set_n(b_.Size(OpB::Rank() - 1));
          dx_gemm_helper_.set_k(a_.Size(OpA::Rank() - 1));
          dx_gemm_helper_.set_cc(cc);
          dx_gemm_helper_.set_is_complex(is_complex_v<typename OpA::value_type>);
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

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
          // Branch with cuBLASDx support
          if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(a_, in_copy), detail::get_operator_capability<Cap>(b_, in_copy));
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            auto result = combine_capabilities<Cap>(dx_gemm_helper_.GetShmRequired(), 
                                                    detail::get_operator_capability<Cap>(a_, in),
                                                    detail::get_operator_capability<Cap>(b_, in));
            MATX_LOG_DEBUG("cuBLASDx DYN_SHM_SIZE: {}", result);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
            bool supported = dx_gemm_helper_.template CheckJITSizeAndTypeRequirements<OpA, OpB>() && 
                             dx_gemm_helper_.IsSupported();

            auto result = combine_capabilities<Cap>(supported, 
                                                    detail::get_operator_capability<Cap>(a_, in),
                                                    detail::get_operator_capability<Cap>(b_, in));
            MATX_LOG_DEBUG("cuBLASDx SUPPORTS_JIT: {}", result);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
            // Get the capability string and add to map
            const auto [key, value] = get_jit_op_str();
      
            // Insert into the map if the key doesn't exist
            if (in.find(key) == in.end()) {
              in[key] = value;
            }
            
            // Also handle child operators
            detail::get_operator_capability<Cap>(a_, in);
            detail::get_operator_capability<Cap>(b_, in);

            MATX_LOG_DEBUG("cuBLASDx JIT_CLASS_QUERY: true");
            return true;
          }
          else if constexpr (Cap == OperatorCapability::BLOCK_DIM) {
            auto block_dims = dx_gemm_helper_.GetBlockDim();
            MATX_LOG_DEBUG("cuBLASDx block dim: {} {} {}", block_dims[0], block_dims[1], block_dims[2]);
            // Use the first dimension as the primary block size (similar to FFT)
            const auto my_block = cuda::std::array<int, 2>{block_dims[0], block_dims[0]};
            return combine_capabilities<Cap>(my_block, 
                                             detail::get_operator_capability<Cap>(a_, in),
                                             detail::get_operator_capability<Cap>(b_, in));
          }
          else if constexpr (Cap == OperatorCapability::GENERATE_LTOIR) {
            auto result = combine_capabilities<Cap>(
                dx_gemm_helper_.GenerateLTOIR(in.ltoir_symbols), 
                detail::get_operator_capability<Cap>(a_, in),
                detail::get_operator_capability<Cap>(b_, in));
            MATX_LOG_DEBUG("cuBLASDx GENERATE_LTOIR: {}", result);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
            // No need to use combine_capabilities here since we're just returning a string.
            const auto inner_op_a_jit_name = detail::get_operator_capability<Cap>(a_, in);
            const auto inner_op_b_jit_name = detail::get_operator_capability<Cap>(b_, in);
            auto result = get_jit_class_name() + "<" + inner_op_a_jit_name + ", " + inner_op_b_jit_name + ">";
            MATX_LOG_DEBUG("cuBLASDx JIT_TYPE_QUERY: {}", result);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::GLOBAL_KERNEL) {
            // If MathDx is enabled we always return false. Other checks on size and type may prevent JIT compilation.
            MATX_LOG_DEBUG("cuBLASDx GLOBAL_KERNEL: false");
            return false;
          }
          else if constexpr (Cap == OperatorCapability::PASS_THROUGH_THREADS) {
            // cuBLASDx needs all threads to call operator() for block-level cooperation
            MATX_LOG_DEBUG("cuBLASDx PASS_THROUGH_THREADS: true");
            return true;
          }
          else if constexpr (Cap == OperatorCapability::GROUPS_PER_BLOCK) {
            // 2D block operators only support one group per block
            const auto my_cap = cuda::std::array<int, 2>{1, 1};
            return combine_capabilities<Cap>(my_cap, 
                                             detail::get_operator_capability<Cap>(a_, in),
                                             detail::get_operator_capability<Cap>(b_, in));
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, 
                                             detail::get_operator_capability<Cap>(a_, in),
                                             detail::get_operator_capability<Cap>(b_, in));
          }
#else
          // Branch without cuBLASDx support
          if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(a_, in_copy), detail::get_operator_capability<Cap>(b_, in_copy));
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
            bool supported = false;
            auto result = combine_capabilities<Cap>(supported, 
                                                    detail::get_operator_capability<Cap>(a_, in),
                                                    detail::get_operator_capability<Cap>(b_, in));
            MATX_LOG_DEBUG("SUPPORTS_JIT (no cuBLASDx): {}", result);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
            MATX_LOG_DEBUG("JIT_TYPE_QUERY (no cuBLASDx): \"\"");
            return "";
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, 
                                             detail::get_operator_capability<Cap>(a_, in),
                                             detail::get_operator_capability<Cap>(b_, in));
          }
#endif
        }
   
        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return out_rank;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          // Perform SpMM or otherwise GEMM.
          static_assert(!is_sparse_tensor_v<OpB>, "sparse rhs not implemented");
          if constexpr (is_sparse_tensor_v<OpA>) {
            if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
              sparse_matmul_impl(permute(cuda::std::get<0>(out), perm_), a_, b_, ex, alpha_, beta_);
            }
            else {
              sparse_matmul_impl(cuda::std::get<0>(out), a_, b_, ex, alpha_, beta_);
            }
          }
          else if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
            matmul_impl(permute(cuda::std::get<0>(out), perm_), a_, b_, ex, alpha_, beta_);
          }
          else {
            matmul_impl(cuda::std::get<0>(out), a_, b_, ex, alpha_, beta_);
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }     

          if constexpr (is_matx_op<OpB>()) {
            b_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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

          if constexpr (is_matx_op<OpB>()) {
            b_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          matxFree(ptr);         
        }
    };
  }


  /**
   * Run a GEMM (generic matrix multiply))
   *
   * Creates a new GEMM plan in the cache if none exists, and uses that to execute
   * the GEMM. This function is preferred over creating a plan directly for both
   * efficiency and simpler code. Since it only uses the signature of the GEMM to
   * decide if a plan is cached, it may be able to reused plans for different
   * A/B/C matrices as long as they were configured with the same dimensions.
   *
   * @tparam OpA
   *    Data type of A tensor or operator
   * @tparam OpB
   *    Data type of B tensor or operator
   *
   * @param A
   *   A Tensor or Operator of shape `... x m x k`
   * @param B
   *   B Tensor or Operator of shape `... x k x n`
   * @param alpha
   *   Scalar multiplier to apply to operator A
   * @param beta
   *   Scalar multiplier to apply to operator C on input
   * 
   * @return 
   *   Operator that produces the output tensor C of shape `... x m x n`
   */
  template<typename OpA, typename OpB>
  __MATX_INLINE__ auto matmul(const OpA &A, const OpB &B, float alpha = 1.0, float beta = 0.0) {
    return detail::MatMulOp(A, B, alpha, beta, detail::no_permute_t{});
  }

  /**
   * Run a GEMM (generic matrix multiply))
   *
   * Creates a new GEMM plan in the cache if none exists, and uses that to execute
   * the GEMM. This function is preferred over creating a plan directly for both
   * efficiency and simpler code. Since it only uses the signature of the GEMM to
   * decide if a plan is cached, it may be able to reused plans for different
   * A/B/C matrices as long as they were configured with the same dimensions.
   *
   * @tparam OpA
   *    Data type of A tensor or operator
   * @tparam OpB
   *    Data type of B tensor or operator
   *
   * @param A
   *   A Tensor or Operator of shape `... x m x k`
   * @param B
   *   B Tensor or Operator of shape `... x k x n`
   * @param axis
   *   the axis of the tensor or operator to perform the gemm along
   * @param alpha
   *   Scalar multiplier to apply to operator A
   * @param beta
   *   Scalar multiplier to apply to operator C on input
   * 
   * @return 
   *   Operator that produces the output tensor C of shape `... x m x n`
   */
  template<typename OpA, typename OpB>
  __MATX_INLINE__ auto matmul(const OpA &A, const OpB &B, const int32_t (&axis)[2], float alpha = 1.0, float beta = 0.0) {
    MATX_STATIC_ASSERT(OpA::Rank() == OpB::Rank(), "matmul: inputs must have same rank to use matmul with axis parameter");
    MATX_STATIC_ASSERT(OpA::Rank() == OpB::Rank(), "matmul: inputs and outputs must have same rank to use matmul with axis parameter");

    auto perm = detail::getPermuteDims<OpA::Rank()>(axis);
    auto in1 = permute(A, perm);
    auto in2 = permute(B, perm);

    return detail::MatMulOp(in1, in2, alpha, beta, perm);
  }  
}
