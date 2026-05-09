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
#include "matx/transforms/qr/qr_cuda.h"
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/qr/qr_lapack.h"
#endif

namespace matx {

namespace detail {
  enum QRComponents : int {
    QR_Q = 0,
    QR_R = 1,
    QR_SOLVER_OUT = 2,
    QR_SOLVER_TAU = 3,
    QR_ECON_Q = 4,
    QR_ECON_R = 5
  };

  template<typename OpA>
  class QRState
  {
    private:
      static_assert(OpA::Rank() >= 2, "qr() requires input rank 2 or higher");
      using value_type = typename OpA::value_type;
      static constexpr int RANK = OpA::Rank();

      typename detail::base_type_t<OpA> a_;
      cuda::std::array<index_t, RANK> q_shape_;
      cuda::std::array<index_t, RANK> r_shape_;
      mutable detail::tensor_impl_t<value_type, RANK> q_;
      mutable detail::tensor_impl_t<value_type, RANK> r_;
      mutable value_type *q_ptr_ = nullptr;
      mutable value_type *r_ptr_ = nullptr;
      mutable bool materialized_ = false;

    public:
      QRState(const OpA &a) : a_(a)
      {
        q_shape_ = SolverShapeFromInput<RANK>(a_);
        r_shape_ = q_shape_;
        q_shape_[RANK - 1] = q_shape_[RANK - 2];
      }

      const auto &Input() const { return a_; }
      const auto &QShape() const { return q_shape_; }
      const auto &RShape() const { return r_shape_; }

      template <typename Executor>
      void Materialize(Executor &&ex) const
      {
        if (materialized_) {
          return;
        }

        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(detail::NoShape{}, std::forward<Executor>(ex));
        }

        detail::AllocateTempTensor(q_, std::forward<Executor>(ex), q_shape_, &q_ptr_);
        detail::AllocateTempTensor(r_, std::forward<Executor>(ex), r_shape_, &r_ptr_);
        qr_impl(q_, r_, a_, std::forward<Executor>(ex));
        materialized_ = true;
      }

      template <typename Executor>
      void Release(Executor &&ex) const
      {
        if (!materialized_) {
          return;
        }

        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(detail::NoShape{}, std::forward<Executor>(ex));
        }

        if (q_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(q_ptr_, ex.getStream());
          }
          else {
            matxFree(q_ptr_);
          }
          q_ptr_ = nullptr;
        }

        if (r_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(r_ptr_, ex.getStream());
          }
          else {
            matxFree(r_ptr_);
          }
          r_ptr_ = nullptr;
        }

        materialized_ = false;
      }

      template <int Component>
      auto Tensor() const
      {
        if constexpr (Component == QR_Q) {
          return q_;
        }
        else {
          return r_;
        }
      }
  };

  template<typename OpA>
  class QROp : public BaseOp<QROp<OpA>>
  {
    private:
      using state_type = QRState<OpA>;
      std::shared_ptr<state_type> state_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using qr_xform_op = bool;
      using tensor_type = detail::tensor_impl_t<value_type, OpA::Rank()>;

      SolverProjectionOp<state_type, QR_Q, tensor_type> Q;
      SolverProjectionOp<state_type, QR_R, tensor_type> R;

      __MATX_INLINE__ std::string str() const { return "qr(" + get_type_str(state_->Input()) + ")"; }
      __MATX_INLINE__ QROp(const OpA &a) :
        state_(std::make_shared<state_type>(a)),
        Q(state_.get(), state_->QShape(), "qr().Q"),
        R(state_.get(), state_->RShape(), "qr().R")
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
        static_assert(is_cuda_executor_v<Executor>, "qr() only supports the CUDA executor currently");
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 3 outputs on qr(). ie: (mtie(Q, R) = qr(A))");

        qr_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), state_->Input(), ex);
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
      // Size is not relevant in qr() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return state_->Input().Size(dim);
      }

  };
}


/**
 * Perform QR decomposition on a matrix using housholders reflections.
 * 
 * If rank > 2, operations are batched.
 *
 * @tparam AType
 *   Tensor or operator type for output of A input tensors.
 *
 * @param A
 *   Input tensor or operator for tensor A input.
 * @returns Operator to generate Q/R outputs
 */
template<typename AType>
__MATX_INLINE__ auto qr(const AType &A) {
  return detail::QROp(A);
}


namespace detail {
  template<typename OpA>
  class SolverQRState
  {
    private:
      static_assert(OpA::Rank() >= 2, "qr_solver() requires input rank 2 or higher");
      using value_type = typename OpA::value_type;
      static constexpr int RANK = OpA::Rank();

      typename detail::base_type_t<OpA> a_;
      cuda::std::array<index_t, RANK> out_shape_;
      cuda::std::array<index_t, RANK - 1> tau_shape_;
      mutable detail::tensor_impl_t<value_type, RANK> out_;
      mutable detail::tensor_impl_t<value_type, RANK - 1> tau_;
      mutable value_type *out_ptr_ = nullptr;
      mutable value_type *tau_ptr_ = nullptr;
      mutable bool materialized_ = false;

    public:
      SolverQRState(const OpA &a) : a_(a)
      {
        out_shape_ = SolverShapeFromInput<RANK>(a_);
        tau_shape_ = SolverVectorShapeFromMatrixShape<RANK>(out_shape_);
      }

      const auto &Input() const { return a_; }
      const auto &OutShape() const { return out_shape_; }
      const auto &TauShape() const { return tau_shape_; }

      template <typename Executor>
      void Materialize(Executor &&ex) const
      {
        if (materialized_) {
          return;
        }

        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(detail::NoShape{}, std::forward<Executor>(ex));
        }

        detail::AllocateTempTensor(out_, std::forward<Executor>(ex), out_shape_, &out_ptr_);
        detail::AllocateTempTensor(tau_, std::forward<Executor>(ex), tau_shape_, &tau_ptr_);
        qr_solver_impl(out_, tau_, a_, std::forward<Executor>(ex));
        materialized_ = true;
      }

      template <typename Executor>
      void Release(Executor &&ex) const
      {
        if (!materialized_) {
          return;
        }

        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(detail::NoShape{}, std::forward<Executor>(ex));
        }

        if (out_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(out_ptr_, ex.getStream());
          }
          else {
            matxFree(out_ptr_);
          }
          out_ptr_ = nullptr;
        }

        if (tau_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(tau_ptr_, ex.getStream());
          }
          else {
            matxFree(tau_ptr_);
          }
          tau_ptr_ = nullptr;
        }

        materialized_ = false;
      }

      template <int Component>
      auto Tensor() const
      {
        if constexpr (Component == QR_SOLVER_OUT) {
          return out_;
        }
        else {
          return tau_;
        }
      }
  };

  template<typename OpA>
  class SolverQROp : public BaseOp<SolverQROp<OpA>>
  {
    private:
      using state_type = SolverQRState<OpA>;
      std::shared_ptr<state_type> state_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using qr_solver_xform_op = bool;
      using out_type = detail::tensor_impl_t<value_type, OpA::Rank()>;
      using tau_type = detail::tensor_impl_t<value_type, OpA::Rank() - 1>;

      SolverProjectionOp<state_type, QR_SOLVER_OUT, out_type> Out;
      SolverProjectionOp<state_type, QR_SOLVER_TAU, tau_type> Tau;

      __MATX_INLINE__ std::string str() const { return "qr_solver()"; }
      __MATX_INLINE__ SolverQROp(const OpA &a) :
        state_(std::make_shared<state_type>(a)),
        Out(state_.get(), state_->OutShape(), "qr_solver().Out"),
        Tau(state_.get(), state_->TauShape(), "qr_solver().Tau")
      {
        MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
      }

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(state_->Input(), in));
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on qr_solver(). ie: (mtie(A, tau) = qr_solver(A))");     

        qr_solver_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), state_->Input(), ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          state_->Input().PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      // Size is not relevant in qr_solver() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return state_->Input().Size(dim);
      }

  };
}

/**
 * Perform a QR decomposition on a matrix using cuSolver or a LAPACK host library.
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
 *   Operator that produces R/householder vectors and tau tensor outputs.
 *   - **Out** - Of shape `... x m x n`. The householder vectors are returned in the
 *               bottom half and *R* is returned in the top half.
 *   - **Tau** - The scalar factors *tau* of shape `... x min(m, n)`.
 */
template<typename OpA>
__MATX_INLINE__ auto qr_solver(const OpA &a) {
  return detail::SolverQROp(a);
}


namespace detail {
  template<typename OpA>
  class EconQRState
  {
    private:
      static_assert(OpA::Rank() >= 2, "qr_econ() requires input rank 2 or higher");
      using value_type = typename OpA::value_type;
      static constexpr int RANK = OpA::Rank();

      typename detail::base_type_t<OpA> a_;
      cuda::std::array<index_t, RANK> q_shape_;
      cuda::std::array<index_t, RANK> r_shape_;
      mutable detail::tensor_impl_t<value_type, RANK> q_;
      mutable detail::tensor_impl_t<value_type, RANK> r_;
      mutable value_type *q_ptr_ = nullptr;
      mutable value_type *r_ptr_ = nullptr;
      mutable bool materialized_ = false;

    public:
      EconQRState(const OpA &a) : a_(a)
      {
        q_shape_ = SolverShapeFromInput<RANK>(a_);
        r_shape_ = q_shape_;
        const index_t k = q_shape_[RANK - 2] < q_shape_[RANK - 1] ? q_shape_[RANK - 2] : q_shape_[RANK - 1];
        q_shape_[RANK - 1] = k;
        r_shape_[RANK - 2] = k;
      }

      const auto &Input() const { return a_; }
      const auto &QShape() const { return q_shape_; }
      const auto &RShape() const { return r_shape_; }

      template <typename Executor>
      void Materialize(Executor &&ex) const
      {
        if (materialized_) {
          return;
        }

        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(detail::NoShape{}, std::forward<Executor>(ex));
        }

        detail::AllocateTempTensor(q_, std::forward<Executor>(ex), q_shape_, &q_ptr_);
        detail::AllocateTempTensor(r_, std::forward<Executor>(ex), r_shape_, &r_ptr_);
        qr_econ_impl(q_, r_, a_, std::forward<Executor>(ex));
        materialized_ = true;
      }

      template <typename Executor>
      void Release(Executor &&ex) const
      {
        if (!materialized_) {
          return;
        }

        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(detail::NoShape{}, std::forward<Executor>(ex));
        }

        if (q_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(q_ptr_, ex.getStream());
          }
          else {
            matxFree(q_ptr_);
          }
          q_ptr_ = nullptr;
        }

        if (r_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(r_ptr_, ex.getStream());
          }
          else {
            matxFree(r_ptr_);
          }
          r_ptr_ = nullptr;
        }

        materialized_ = false;
      }

      template <int Component>
      auto Tensor() const
      {
        if constexpr (Component == QR_ECON_Q) {
          return q_;
        }
        else {
          return r_;
        }
      }
  };

  template<typename OpA>
  class EconQROp : public BaseOp<EconQROp<OpA>>
  {
    private:
      using state_type = EconQRState<OpA>;
      std::shared_ptr<state_type> state_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using qr_solver_xform_op = bool;
      using tensor_type = detail::tensor_impl_t<value_type, OpA::Rank()>;

      SolverProjectionOp<state_type, QR_ECON_Q, tensor_type> Q;
      SolverProjectionOp<state_type, QR_ECON_R, tensor_type> R;

      __MATX_INLINE__ std::string str() const { return "qr_econ()"; }
      __MATX_INLINE__ EconQROp(const OpA &a) :
        state_(std::make_shared<state_type>(a)),
        Q(state_.get(), state_->QShape(), "qr_econ().Q"),
        R(state_.get(), state_->RShape(), "qr_econ().R")
      {
      }

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(state_->Input(), in));
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on qr_econ(). ie: (mtie(Q, R) = qr_econ(A))");     

        qr_econ_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), state_->Input(), ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          state_->Input().PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      // Size is not relevant in qr_solver() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return state_->Input().Size(dim);
      }

  };
}

/**
 * Perform an economic QR decomposition on a matrix using cuSolver.
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
 *   Operator that produces QR outputs.
 *   - **Q** - Of shape `... x m x min(m, n)`, the reduced orthonormal basis for the span of A.
 *   - **R** - Upper triangular matrix of shape  `... x min(m, n) x n`.
 */
template<typename OpA>
__MATX_INLINE__ auto qr_econ(const OpA &a) {
  return detail::EconQROp(a);
}

}
