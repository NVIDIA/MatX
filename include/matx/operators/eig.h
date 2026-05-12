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

    public:
      EigState(const OpA &a, EigenMode jobz, SolverFillMode uplo) : a_(a), jobz_(jobz), uplo_(uplo)
      {
        vectors_shape_ = SolverShapeFromInput<RANK>(a_);
        values_shape_ = SolverVectorShapeFromMatrixShape<RANK>(vectors_shape_);
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

        detail::AllocateTempTensor(vectors_, std::forward<Executor>(ex), vectors_shape_, &vectors_ptr_);
        detail::AllocateTempTensor(values_, std::forward<Executor>(ex), values_shape_, &values_ptr_);
        eig_impl(vectors_, values_, a_, std::forward<Executor>(ex), jobz_, uplo_);
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
        Vectors(state_.get(), state_->VectorsShape(), "eig().Vectors"),
        Values(state_.get(), state_->ValuesShape(), "eig().Values")
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
