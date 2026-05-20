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
#include "matx/operators/solver_projection.h"
#include "matx/transforms/lu/lu_cuda.h"
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
  #include "matx/transforms/solver_cusolverdx.h"
#endif
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/lu/lu_lapack.h"
#endif

namespace matx {
namespace detail {
  enum LUComponents : int {
    LU_FACTORS = 0,
    LU_PIV = 1
  };

  template<typename OpA>
  class LUState
  {
    private:
      static_assert(OpA::Rank() >= 2, "lu() requires input rank 2 or higher");
      using a_value_type = typename OpA::value_type;
      using piv_value_type = int64_t;
      static constexpr int RANK = OpA::Rank();

      typename detail::base_type_t<OpA> a_;
      cuda::std::array<index_t, RANK> factors_shape_;
      cuda::std::array<index_t, RANK - 1> piv_shape_;
      mutable detail::tensor_impl_t<a_value_type, RANK> factors_;
      mutable detail::tensor_impl_t<piv_value_type, RANK - 1> piv_;
      mutable a_value_type *factors_ptr_ = nullptr;
      mutable piv_value_type *piv_ptr_ = nullptr;
      mutable bool materialized_ = false;
      mutable int materialize_count_ = 0;
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      mutable cuSolverDxHelper<a_value_type> dx_lu_helper_;
#endif

    public:
      using input_type = OpA;

      LUState(const OpA &a) : a_(a)
      {
        factors_shape_ = SolverShapeFromInput<RANK>(a_);
        piv_shape_ = SolverVectorShapeFromMatrixShape<RANK>(factors_shape_);
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
        int major = 0;
        int minor = 0;
        int device = 0;
        MATX_CUDA_CHECK(cudaGetDevice(&device));
        MATX_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        MATX_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        const int cc = major * 100 + minor * 10;

        dx_lu_helper_.set_m(factors_shape_[RANK - 2]);
        dx_lu_helper_.set_n(factors_shape_[RANK - 1]);
        dx_lu_helper_.set_cc(cc);
        dx_lu_helper_.set_function(CUSOLVERDX_FUNCTION_GETRF_PARTIAL_PIVOT);
#endif
      }

      const auto &FactorsShape() const { return factors_shape_; }
      const auto &PivShape() const { return piv_shape_; }
      const auto &Input() const { return a_; }

      template <typename Executor>
      void Materialize(Executor &&ex) const
      {
        if (materialized_) {
          materialize_count_++;
          return;
        }

        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(detail::NoShape{}, std::forward<Executor>(ex));
        }

        const auto cleanup = [&]() noexcept {
          try {
            if (factors_ptr_ != nullptr) {
              if constexpr (is_cuda_executor_v<Executor>) {
                matxFree(factors_ptr_, ex.getStream());
              }
              else {
                matxFree(factors_ptr_);
              }
              factors_ptr_ = nullptr;
            }

            if (piv_ptr_ != nullptr) {
              if constexpr (is_cuda_executor_v<Executor>) {
                matxFree(piv_ptr_, ex.getStream());
              }
              else {
                matxFree(piv_ptr_);
              }
              piv_ptr_ = nullptr;
            }

            if constexpr (is_matx_op<OpA>()) {
              a_.PostRun(detail::NoShape{}, std::forward<Executor>(ex));
            }
          }
          catch (...) {
          }
          materialized_ = false;
          materialize_count_ = 0;
        };

        try {
          detail::AllocateTempTensor(factors_, std::forward<Executor>(ex), factors_shape_, &factors_ptr_);
          detail::AllocateTempTensor(piv_, std::forward<Executor>(ex), piv_shape_, &piv_ptr_);
          if constexpr (is_cuda_executor_v<Executor>) {
            lu_impl(factors_, piv_, a_, std::forward<Executor>(ex));
          }
          else {
#if MATX_EN_CPU_SOLVER
            if constexpr (std::is_same_v<piv_value_type, lapack_int_t>) {
              lu_impl(factors_, piv_, a_, std::forward<Executor>(ex));
            }
            else {
              detail::tensor_impl_t<lapack_int_t, RANK - 1> lapack_piv;
              lapack_int_t *lapack_piv_ptr = nullptr;
              try {
                detail::AllocateTempTensor(lapack_piv, std::forward<Executor>(ex), piv_shape_, &lapack_piv_ptr);
                lu_impl(factors_, lapack_piv, a_, std::forward<Executor>(ex));
                for (index_t i = 0; i < lapack_piv.TotalSize(); i++) {
                  piv_.Data()[i] = static_cast<piv_value_type>(lapack_piv.Data()[i]);
                }
                matxFree(lapack_piv_ptr);
                lapack_piv_ptr = nullptr;
              }
              catch (...) {
                if (lapack_piv_ptr != nullptr) {
                  matxFree(lapack_piv_ptr);
                }
                throw;
              }
            }
#else
            lu_impl(factors_, piv_, a_, std::forward<Executor>(ex));
#endif
          }
        }
        catch (...) {
          cleanup();
          throw;
        }
        materialized_ = true;
        materialize_count_ = 1;
      }

      template <typename Executor>
      void Release(Executor &&ex) const
      {
        if (!materialized_) {
          return;
        }
        if (materialize_count_ > 1) {
          materialize_count_--;
          return;
        }

        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(detail::NoShape{}, std::forward<Executor>(ex));
        }

        if (factors_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(factors_ptr_, ex.getStream());
          }
          else {
            matxFree(factors_ptr_);
          }
          factors_ptr_ = nullptr;
        }

        if (piv_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(piv_ptr_, ex.getStream());
          }
          else {
            matxFree(piv_ptr_);
          }
          piv_ptr_ = nullptr;
        }

        materialized_ = false;
        materialize_count_ = 0;
      }

      template <int Component>
      auto Tensor() const
      {
        if constexpr (Component == LU_FACTORS) {
          return factors_;
        }
        else {
          return piv_;
        }
      }

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      template <int Component>
      bool SupportsJITProjection() const
      {
        return (RANK >= 2) && (RANK <= 4) && dx_lu_helper_.IsSupported();
      }

      template <int Component>
      int GetJITProjectionShmRequired() const
      {
        if (!SupportsJITProjection<Component>()) {
          return 0;
        }
        return dx_lu_helper_.GetShmRequired();
      }

      template <int Component>
      cuda::std::array<int, 2> GetJITProjectionBlockDimRange() const
      {
        return dx_lu_helper_.GetBlockDimRange();
      }

      template <int Component>
      bool GenerateJITProjectionLTOIR(std::set<std::string> &ltoir_symbols) const
      {
        return dx_lu_helper_.GenerateLTOIR(ltoir_symbols);
      }

      template <int Component>
      std::string GetJITProjectionClassName() const
      {
        // LU factors and pivots deliberately share one generated class because
        // GetLuProjectionFuncStr dispatches on Component inside the body.
        return std::string("JITLUProjectionOp_R") + std::to_string(RANK) + "_" + dx_lu_helper_.GetSymbolName();
      }

      template <int Component>
      std::string GetJITProjectionTypeName(const std::string &inner_op_jit_name) const
      {
        return GetJITProjectionClassName<Component>() + "<" + inner_op_jit_name + ", " + std::to_string(Component) + ">";
      }

      template <int Component>
      JITCacheKey GetJITProjectionCacheKey() const
      {
        auto key = detail::MakeJITCacheKeyForType<LUState<OpA>>("JITLUProjection_v3");
        detail::HashJITCacheValue(key, Component);
        detail::HashJITCacheValue(key, RANK);
        for (int i = 0; i < RANK; ++i) {
          detail::HashJITCacheValue(key, factors_shape_[i]);
        }
        detail::HashJITCacheString(key, dx_lu_helper_.GetSymbolName());
        return key;
      }

      template <int Component>
      void AddJITProjectionClasses(std::unordered_map<std::string, std::string> &in) const
      {
        const std::string class_name = GetJITProjectionClassName<Component>();
        if (in.find(class_name) != in.end()) {
          return;
        }

        const std::string solver_func_name = std::string(SOLVER_DX_FUNC_PREFIX) + "_" + dx_lu_helper_.GetSymbolName();
        in[class_name] =
          " extern \"C\" __device__ void " + solver_func_name + "(" + detail::type_to_string<a_value_type>() + "*, int*, int*);\n"
          " template <typename OpA, int Component> struct " + class_name + "  {\n"
          "  using solver_value_type = typename OpA::value_type;\n"
          "  using value_type = cuda::std::conditional_t<Component == " + std::to_string(LU_PIV) + ", int64_t, solver_value_type>;\n"
          "  using matxop = bool;\n"
          "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n"
          "  template <typename CapType, typename... Is>\n"
          "  __MATX_INLINE__ __MATX_DEVICE__ value_type operator()(Is... indices) const\n"
          "  {\n"
          "    " + dx_lu_helper_.GetLuProjectionFuncStr(solver_func_name, LU_FACTORS, LU_PIV) + "\n"
          "  }\n"
          "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n"
          "  {\n"
          "    if constexpr (Component == " + std::to_string(LU_PIV) + ") { return OpA::Rank() - 1; }\n"
          "    else { return OpA::Rank(); }\n"
          "  }\n"
          "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const\n"
          "  {\n"
          "    if constexpr (Component == " + std::to_string(LU_PIV) + ") {\n"
          "      if (dim < OpA::Rank() - 2) { return a_.Size(dim); }\n"
          "      const index_t m = a_.Size(OpA::Rank() - 2);\n"
          "      const index_t n = a_.Size(OpA::Rank() - 1);\n"
          "      return m < n ? m : n;\n"
          "    }\n"
          "    else { return a_.Size(dim); }\n"
          "  }\n"
          "};\n";
      }
#endif
  };

  template<typename OpA>
  class LUOp : public BaseOp<LUOp<OpA>>
  {
    private:
      using state_type = LUState<OpA>;
      std::shared_ptr<state_type> state_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using lu_xform_op = bool;
      using factors_type = detail::tensor_impl_t<value_type, OpA::Rank()>;
      using piv_type = detail::tensor_impl_t<int64_t, OpA::Rank() - 1>;

      SolverProjectionOp<state_type, LU_FACTORS, factors_type> LU;
      SolverProjectionOp<state_type, LU_PIV, piv_type> Piv;

      __MATX_INLINE__ std::string str() const { return "lu()"; }
      __MATX_INLINE__ LUOp(const OpA &a) :
        state_(std::make_shared<state_type>(a)),
        LU(state_, state_->FactorsShape(), "lu().LU"),
        Piv(state_, state_->PivShape(), "lu().Piv")
      {
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
      };

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(state_->Input(), in));
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on lu(). ie: (mtie(O, piv) = lu(A))");     

        lu_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), state_->Input(), ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          state_->Input().PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          state_->Input().PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      // Size is not relevant in eig() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return state_->FactorsShape()[dim];
      }

  };
}

/**
 * Performs an LU factorization using partial pivoting with row interchanges.
 * The factorization has the form `A = P * L * U`.
 * 
 * The input and output tensors may be the same tensor, in which case the
 * input is overwritten.
 *
 * If rank > 2, operations are batched.
 * 
 * @tparam OpA
 *   Data type of input a tensor or operator
 * 
 * @param a
 *   Input tensor or operator of shape `... x m x n`
 * 
 * @return
 *   Operator that produces a tensor containing *L* and *U* and another containing the pivot indices.
 *   - **Out** - A tensor of shape `... x m x n` containing both *L* and *U*. *L* can be extracted
 *               from the bottom half (the unit diagonals are not stored in *Out*), and *U* can
 *               be extracted from the top half with the diagonals.
 *   - **Piv** - The tensor of pivot indices with shape `... x min(m, n)`. For
 *               \f$ 0 \leq i < \min(m, n) \f$, row i was interchanged with row 
 *               \f$ Piv(..., i) - 1 \f$. It must be of type `int64_t` for cuda
 *               `matx::lapack_int_t` for host.
 */
template<typename OpA>
__MATX_INLINE__ auto lu(const OpA &a) {
  return detail::LUOp(a);
}

}
