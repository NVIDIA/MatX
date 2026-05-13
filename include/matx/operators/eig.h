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
#include "matx/transforms/eig/eig_cuda.h"
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
  #include "matx/transforms/solver_cusolverdx.h"
#endif
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/eig/eig_lapack.h"
#endif

namespace matx {



namespace detail {
  enum EigComponents : int {
    EIG_VECTORS = 0,
    EIG_VALUES = 1
  };

  template<typename OpA>
  class EigState
  {
    private:
      static_assert(OpA::Rank() >= 2, "eig() requires input rank 2 or higher");
      using a_value_type = typename OpA::value_type;
      using w_value_type = typename inner_op_type_t<a_value_type>::type;
      static constexpr int RANK = OpA::Rank();

      typename detail::base_type_t<OpA> a_;
      EigenMode jobz_;
      SolverFillMode uplo_;
      cuda::std::array<index_t, RANK> vectors_shape_;
      cuda::std::array<index_t, RANK - 1> values_shape_;
      mutable detail::tensor_impl_t<a_value_type, RANK> vectors_;
      mutable detail::tensor_impl_t<w_value_type, RANK - 1> values_;
      mutable a_value_type *vectors_ptr_ = nullptr;
      mutable w_value_type *values_ptr_ = nullptr;
      mutable bool materialized_ = false;
      mutable int materialize_count_ = 0;
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      mutable cuSolverDxHelper<a_value_type> dx_heev_values_helper_;
      mutable cuSolverDxHelper<a_value_type> dx_heev_vectors_helper_;
#endif

    public:
      using input_type = OpA;

      EigState(const OpA &a, EigenMode jobz, SolverFillMode uplo) : a_(a), jobz_(jobz), uplo_(uplo)
      {
        vectors_shape_ = SolverShapeFromInput<RANK>(a_);
        values_shape_ = SolverVectorShapeFromMatrixShape<RANK>(vectors_shape_);
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
        int major = 0;
        int minor = 0;
        int device = 0;
        MATX_CUDA_CHECK(cudaGetDevice(&device));
        MATX_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        MATX_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        const int cc = major * 100 + minor * 10;

        const index_t n = vectors_shape_[RANK - 1];
        dx_heev_values_helper_.set_m(n);
        dx_heev_values_helper_.set_n(n);
        dx_heev_values_helper_.set_cc(cc);
        dx_heev_values_helper_.set_function(CUSOLVERDX_FUNCTION_HEEV);
        dx_heev_values_helper_.set_fill_mode(uplo_);
        dx_heev_values_helper_.set_job(CUSOLVERDX_JOB_NO_VECTORS);

        dx_heev_vectors_helper_.set_m(n);
        dx_heev_vectors_helper_.set_n(n);
        dx_heev_vectors_helper_.set_cc(cc);
        dx_heev_vectors_helper_.set_function(CUSOLVERDX_FUNCTION_HEEV);
        dx_heev_vectors_helper_.set_fill_mode(uplo_);
        dx_heev_vectors_helper_.set_job(CUSOLVERDX_JOB_OVERWRITE_VECTORS);
#endif
      }

      const auto &Input() const { return a_; }
      const auto &VectorsShape() const { return vectors_shape_; }
      const auto &ValuesShape() const { return values_shape_; }
      EigenMode Jobz() const { return jobz_; }
      SolverFillMode Uplo() const { return uplo_; }

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
            if (vectors_ptr_ != nullptr) {
              if constexpr (is_cuda_executor_v<Executor>) {
                matxFree(vectors_ptr_, ex.getStream());
              }
              else {
                matxFree(vectors_ptr_);
              }
              vectors_ptr_ = nullptr;
            }
            if (values_ptr_ != nullptr) {
              if constexpr (is_cuda_executor_v<Executor>) {
                matxFree(values_ptr_, ex.getStream());
              }
              else {
                matxFree(values_ptr_);
              }
              values_ptr_ = nullptr;
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
          detail::AllocateTempTensor(vectors_, std::forward<Executor>(ex), vectors_shape_, &vectors_ptr_);
          detail::AllocateTempTensor(values_, std::forward<Executor>(ex), values_shape_, &values_ptr_);
          eig_impl(vectors_, values_, a_, std::forward<Executor>(ex), jobz_, uplo_);
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

        if (vectors_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(vectors_ptr_, ex.getStream());
          }
          else {
            matxFree(vectors_ptr_);
          }
          vectors_ptr_ = nullptr;
        }

        if (values_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(values_ptr_, ex.getStream());
          }
          else {
            matxFree(values_ptr_);
          }
          values_ptr_ = nullptr;
        }

        materialized_ = false;
        materialize_count_ = 0;
      }

      template <int Component>
      auto Tensor() const
      {
        if constexpr (Component == EIG_VECTORS) {
          return vectors_;
        }
        else {
          return values_;
        }
      }

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
      template <int Component>
      bool SupportsJITProjection() const
      {
        const bool square = vectors_shape_[RANK - 2] == vectors_shape_[RANK - 1];
        if constexpr (Component == EIG_VECTORS) {
          return (RANK >= 2) && (RANK <= 4) && square &&
                 jobz_ == EigenMode::VECTOR &&
                 dx_heev_vectors_helper_.IsSupported();
        }
        else {
          return (RANK >= 2) && (RANK <= 4) && square &&
                 dx_heev_values_helper_.IsSupported();
        }
      }

      template <int Component>
      int GetJITProjectionShmRequired() const
      {
        if constexpr (Component == EIG_VECTORS) {
          return dx_heev_vectors_helper_.GetShmRequired();
        }
        else {
          return dx_heev_values_helper_.GetShmRequired();
        }
      }

      template <int Component>
      cuda::std::array<int, 2> GetJITProjectionBlockDimRange() const
      {
        if constexpr (Component == EIG_VECTORS) {
          return dx_heev_vectors_helper_.GetBlockDimRange();
        }
        else {
          return dx_heev_values_helper_.GetBlockDimRange();
        }
      }

      template <int Component>
      bool GenerateJITProjectionLTOIR(std::set<std::string> &ltoir_symbols) const
      {
        if constexpr (Component == EIG_VECTORS) {
          return dx_heev_vectors_helper_.GenerateLTOIR(ltoir_symbols);
        }
        else {
          return dx_heev_values_helper_.GenerateLTOIR(ltoir_symbols);
        }
      }

      template <int Component>
      std::string GetJITProjectionClassName() const
      {
        return std::string("JITEigProjectionOp_R") + std::to_string(RANK) + "_" +
               (Component == EIG_VALUES ? dx_heev_values_helper_.GetSymbolName() : dx_heev_vectors_helper_.GetSymbolName()) +
               "_C" + std::to_string(Component);
      }

      template <int Component>
      std::string GetJITProjectionTypeName(const std::string &inner_op_jit_name) const
      {
        return GetJITProjectionClassName<Component>() + "<" + inner_op_jit_name + ", " + std::to_string(Component) + ">";
      }

      template <int Component>
      JITCacheKey GetJITProjectionCacheKey() const
      {
        auto key = detail::MakeJITCacheKeyForType<EigState<OpA>>("JITEigProjection_v2");
        detail::HashJITCacheValue(key, Component);
        detail::HashJITCacheValue(key, RANK);
        if constexpr (Component == EIG_VECTORS) {
          detail::HashJITCacheValue(key, static_cast<int>(jobz_));
        }
        detail::HashJITCacheValue(key, static_cast<int>(uplo_));
        for (int i = 0; i < RANK; ++i) {
          detail::HashJITCacheValue(key, vectors_shape_[i]);
        }
        detail::HashJITCacheString(key, dx_heev_values_helper_.GetSymbolName());
        detail::HashJITCacheString(key, dx_heev_vectors_helper_.GetSymbolName());
        return key;
      }

      template <int Component>
      void AddJITProjectionClasses(std::unordered_map<std::string, std::string> &in) const
      {
        const std::string class_name = GetJITProjectionClassName<Component>();
        if (in.find(class_name) != in.end()) {
          return;
        }

        const std::string values_func_name = std::string(SOLVER_DX_FUNC_PREFIX) + "_" + dx_heev_values_helper_.GetSymbolName();
        const std::string vectors_func_name = std::string(SOLVER_DX_FUNC_PREFIX) + "_" + dx_heev_vectors_helper_.GetSymbolName();
        std::string externs;
        std::string projection_body;
        if constexpr (Component == EIG_VALUES) {
          externs =
            " extern \"C\" __device__ void " + values_func_name + "(" + detail::type_to_string<a_value_type>() + "*, " + detail::type_to_string<w_value_type>() + "*, " + detail::type_to_string<a_value_type>() + "*, int*);\n";
          projection_body = dx_heev_values_helper_.GetHeevProjectionFuncStr(values_func_name, EIG_VECTORS, EIG_VALUES);
        }
        else {
          externs =
            " extern \"C\" __device__ void " + vectors_func_name + "(" + detail::type_to_string<a_value_type>() + "*, " + detail::type_to_string<w_value_type>() + "*, " + detail::type_to_string<a_value_type>() + "*, int*);\n";
          projection_body = dx_heev_vectors_helper_.GetHeevProjectionFuncStr(vectors_func_name, EIG_VECTORS, EIG_VALUES);
        }
        in[class_name] =
          externs +
          " template <typename OpA, int Component> struct " + class_name + "  {\n"
          "  using solver_value_type = typename OpA::value_type;\n"
          "  using precision_type = typename inner_op_type_t<solver_value_type>::type;\n"
          "  using value_type = cuda::std::conditional_t<Component == " + std::to_string(EIG_VALUES) + ", precision_type, solver_value_type>;\n"
          "  using matxop = bool;\n"
          "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n"
          "  template <typename CapType, typename... Is>\n"
          "  __MATX_INLINE__ __MATX_DEVICE__ value_type operator()(Is... indices) const\n"
          "  {\n"
          "    " + projection_body + "\n"
          "  }\n"
          "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n"
          "  {\n"
          "    if constexpr (Component == " + std::to_string(EIG_VALUES) + ") { return OpA::Rank() - 1; }\n"
          "    else { return OpA::Rank(); }\n"
          "  }\n"
          "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const\n"
          "  {\n"
          "    if constexpr (Component == " + std::to_string(EIG_VALUES) + ") {\n"
          "      if (dim < OpA::Rank() - 2) { return a_.Size(dim); }\n"
          "      return a_.Size(OpA::Rank() - 1);\n"
          "    }\n"
          "    else { return a_.Size(dim); }\n"
          "  }\n"
          "};\n";
      }
#endif
  };

  template<typename OpA>
  class EigOp : public BaseOp<EigOp<OpA>>
  {
    private:
      using state_type = EigState<OpA>;
      std::shared_ptr<state_type> state_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using eig_xform_op = bool;
      using values_value_type = typename inner_op_type_t<value_type>::type;
      using vectors_type = detail::tensor_impl_t<value_type, OpA::Rank()>;
      using values_type = detail::tensor_impl_t<values_value_type, OpA::Rank() - 1>;

      SolverProjectionOp<state_type, EIG_VECTORS, vectors_type> Vectors;
      SolverProjectionOp<state_type, EIG_VALUES, values_type> Values;

      __MATX_INLINE__ std::string str() const { return "eig()"; }
      __MATX_INLINE__ EigOp(const OpA &a, EigenMode jobz, SolverFillMode uplo) :
        state_(std::make_shared<state_type>(a, jobz, uplo)),
        Vectors(state_, state_->VectorsShape(), "eig().Vectors"),
        Values(state_, state_->ValuesShape(), "eig().Values")
      {
        MATX_LOG_TRACE("{} constructor: jobz={}, uplo={}", str(), static_cast<int>(jobz), static_cast<int>(uplo));
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
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on eig(). ie: (mtie(O, w) = eig(A))");     

        eig_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), state_->Input(), ex, state_->Jobz(), state_->Uplo());
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
        return state_->Input().Size(dim);
      }

  };
}


/**
 * Performs an eigenvalue decomposition, computing the eigenvalues, and
 * optionally the eigenvectors, for a Hermitian or real symmetric matrix.
 * 
 * If rank > 2, operations are batched.
 * 
 * @tparam OpA
 *   Data type of input a tensor or operator
 * 
 * @param a
 *   Input Hermitian/symmetric tensor or operator of shape `... x n x n`
 * @param jobz
 *   Whether to compute eigenvectors.
 * @param uplo
 *   Part of matrix to fill
 * 
 * @return 
 *   Operator that produces eigenvectors and eigenvalues tensors. Regardless of jobz,
 *   both tensors must be correctly setup for the operation and used with `mtie()`.
 *   - **Eigenvectors** - The eigenvectors tensor of shape `... x n x n` where each column
 *       contains the normalized eigenvectors.
 *   - **Eigenvalues** - The eigenvalues tensor of shape `... x n`. This must be real
 *       and match the inner type of the input/output tensors.
 */
template<typename OpA>
__MATX_INLINE__ auto eig(const OpA &a,
                          EigenMode jobz = EigenMode::VECTOR, 
                          SolverFillMode uplo  = SolverFillMode::UPPER) {
  return detail::EigOp(a, jobz, uplo);
}

}
