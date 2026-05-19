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
#include "matx/transforms/svd/svd_cuda.h"
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/svd/svd_lapack.h"
#endif
namespace matx {


namespace detail {
  enum SVDComponents : int {
    SVD_U = 0,
    SVD_S = 1,
    SVD_VT = 2
  };

  template<typename OpA>
  class SVDState
  {
    private:
      static_assert(OpA::Rank() >= 2, "svd() requires input rank 2 or higher");
      using a_value_type = typename OpA::value_type;
      using s_value_type = typename inner_op_type_t<a_value_type>::type;
      static constexpr int RANK = OpA::Rank();

      typename detail::base_type_t<OpA> a_;
      SVDMode jobz_;
      SVDHostAlgo algo_;
      cuda::std::array<index_t, RANK> u_shape_;
      cuda::std::array<index_t, RANK - 1> s_shape_;
      cuda::std::array<index_t, RANK> vt_shape_;
      mutable detail::tensor_impl_t<a_value_type, RANK> u_;
      mutable detail::tensor_impl_t<s_value_type, RANK - 1> s_;
      mutable detail::tensor_impl_t<a_value_type, RANK> vt_;
      mutable a_value_type *u_ptr_ = nullptr;
      mutable s_value_type *s_ptr_ = nullptr;
      mutable a_value_type *vt_ptr_ = nullptr;
      mutable bool materialized_ = false;
      mutable int materialize_count_ = 0;

    public:
      using input_type = OpA;

      SVDState(const OpA &a, const SVDMode jobz, const SVDHostAlgo algo) : a_(a), jobz_(jobz), algo_(algo)
      {
        auto a_shape = SolverShapeFromInput<RANK>(a_);
        u_shape_ = a_shape;
        s_shape_ = SolverVectorShapeFromMatrixShape<RANK>(a_shape);
        vt_shape_ = a_shape;

        const index_t m = a_shape[RANK - 2];
        const index_t n = a_shape[RANK - 1];
        const index_t k = m < n ? m : n;
        u_shape_[RANK - 1] = jobz_ == SVDMode::REDUCED ? k : m;
        vt_shape_[RANK - 2] = jobz_ == SVDMode::REDUCED ? k : n;
      }

      const auto &Input() const { return a_; }
      const auto &UShape() const { return u_shape_; }
      const auto &SShape() const { return s_shape_; }
      const auto &VTShape() const { return vt_shape_; }
      SVDMode Jobz() const { return jobz_; }
      SVDHostAlgo Algo() const { return algo_; }

      template <int Component>
      void ValidateProjection() const
      {
        if constexpr (Component == SVD_S) {
          // SVDMode::NONE maps to cuSolver jobz='N': singular values are still computed,
          // while U and VT are intentionally omitted.
        }
        else if constexpr (Component == SVD_U) {
          if (jobz_ == SVDMode::NONE) {
            MATX_THROW(matxInvalidParameter, "svd().U cannot be used when SVDMode::NONE is selected");
          }
        }
        else if constexpr (Component == SVD_VT) {
          if (jobz_ == SVDMode::NONE) {
            MATX_THROW(matxInvalidParameter, "svd().VT cannot be used when SVDMode::NONE is selected");
          }
        }
      }

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
            if (u_ptr_ != nullptr) {
              if constexpr (is_cuda_executor_v<Executor>) {
                matxFree(u_ptr_, ex.getStream());
              }
              else {
                matxFree(u_ptr_);
              }
              u_ptr_ = nullptr;
            }
            if (s_ptr_ != nullptr) {
              if constexpr (is_cuda_executor_v<Executor>) {
                matxFree(s_ptr_, ex.getStream());
              }
              else {
                matxFree(s_ptr_);
              }
              s_ptr_ = nullptr;
            }
            if (vt_ptr_ != nullptr) {
              if constexpr (is_cuda_executor_v<Executor>) {
                matxFree(vt_ptr_, ex.getStream());
              }
              else {
                matxFree(vt_ptr_);
              }
              vt_ptr_ = nullptr;
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
          detail::AllocateTempTensor(s_, std::forward<Executor>(ex), s_shape_, &s_ptr_);
          if (jobz_ == SVDMode::NONE) {
            a_value_type *u_dummy_ptr = nullptr;
            a_value_type *vt_dummy_ptr = nullptr;
            const auto free_dummy = [&]() noexcept {
              try {
                if (u_dummy_ptr != nullptr) {
                  if constexpr (is_cuda_executor_v<Executor>) {
                    matxFree(u_dummy_ptr, ex.getStream());
                  }
                  else {
                    matxFree(u_dummy_ptr);
                  }
                  u_dummy_ptr = nullptr;
                }
                if (vt_dummy_ptr != nullptr) {
                  if constexpr (is_cuda_executor_v<Executor>) {
                    matxFree(vt_dummy_ptr, ex.getStream());
                  }
                  else {
                    matxFree(vt_dummy_ptr);
                  }
                  vt_dummy_ptr = nullptr;
                }
              }
              catch (...) {
              }
            };

            try {
              if constexpr (is_cuda_executor_v<Executor>) {
                matxAlloc(reinterpret_cast<void **>(&u_dummy_ptr), sizeof(a_value_type), MATX_ASYNC_DEVICE_MEMORY, ex.getStream());
                matxAlloc(reinterpret_cast<void **>(&vt_dummy_ptr), sizeof(a_value_type), MATX_ASYNC_DEVICE_MEMORY, ex.getStream());
              }
              else {
                matxAlloc(reinterpret_cast<void **>(&u_dummy_ptr), sizeof(a_value_type), MATX_HOST_MALLOC_MEMORY);
                matxAlloc(reinterpret_cast<void **>(&vt_dummy_ptr), sizeof(a_value_type), MATX_HOST_MALLOC_MEMORY);
              }

              auto u_dummy = make_tensor(u_dummy_ptr, u_shape_);
              auto vt_dummy = make_tensor(vt_dummy_ptr, vt_shape_);
              if constexpr (is_cuda_executor_v<Executor>) {
                svd_impl(u_dummy, s_, vt_dummy, a_, std::forward<Executor>(ex), jobz_);
              }
              else {
                svd_impl(u_dummy, s_, vt_dummy, a_, std::forward<Executor>(ex), jobz_, algo_);
              }
            }
            catch (...) {
              free_dummy();
              throw;
            }
            free_dummy();
          }
          else {
            detail::AllocateTempTensor(u_, std::forward<Executor>(ex), u_shape_, &u_ptr_);
            detail::AllocateTempTensor(vt_, std::forward<Executor>(ex), vt_shape_, &vt_ptr_);
            if constexpr (is_cuda_executor_v<Executor>) {
              svd_impl(u_, s_, vt_, a_, std::forward<Executor>(ex), jobz_);
            }
            else {
              svd_impl(u_, s_, vt_, a_, std::forward<Executor>(ex), jobz_, algo_);
            }
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

        if (u_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(u_ptr_, ex.getStream());
          }
          else {
            matxFree(u_ptr_);
          }
          u_ptr_ = nullptr;
        }

        if (s_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(s_ptr_, ex.getStream());
          }
          else {
            matxFree(s_ptr_);
          }
          s_ptr_ = nullptr;
        }

        if (vt_ptr_ != nullptr) {
          if constexpr (is_cuda_executor_v<Executor>) {
            matxFree(vt_ptr_, ex.getStream());
          }
          else {
            matxFree(vt_ptr_);
          }
          vt_ptr_ = nullptr;
        }

        materialized_ = false;
        materialize_count_ = 0;
      }

      template <int Component>
      auto Tensor() const
      {
        ValidateProjection<Component>();

        if constexpr (Component == SVD_U) {
          return u_;
        }
        else if constexpr (Component == SVD_S) {
          return s_;
        }
        else {
          return vt_;
        }
      }
  };

  template<typename OpA>
  class SVDOp : public BaseOp<SVDOp<OpA>>
  {
    private:
      using state_type = SVDState<OpA>;
      std::shared_ptr<state_type> state_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using svd_xform_op = bool;
      using u_type = detail::tensor_impl_t<value_type, OpA::Rank()>;
      using s_value_type = typename inner_op_type_t<value_type>::type;
      using s_type = detail::tensor_impl_t<s_value_type, OpA::Rank() - 1>;
      using vt_type = detail::tensor_impl_t<value_type, OpA::Rank()>;

      SolverProjectionOp<state_type, SVD_U, u_type> U;
      SolverProjectionOp<state_type, SVD_S, s_type> S;
      SolverProjectionOp<state_type, SVD_VT, vt_type> VT;

      __MATX_INLINE__ std::string str() const { return "svd(" + get_type_str(state_->Input()) + ")"; }
      __MATX_INLINE__ SVDOp(const OpA &a, const SVDMode jobz, const SVDHostAlgo algo) :
        state_(std::make_shared<state_type>(a, jobz, algo)),
        U(state_, state_->UShape(), "svd().U"),
        S(state_, state_->SShape(), "svd().S"),
        VT(state_, state_->VTShape(), "svd().VT")
      {
        MATX_LOG_TRACE("{} constructor: jobz={}, algo={}", str(), static_cast<int>(jobz), static_cast<int>(algo));
      };

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(state_->Input(), in));
        }
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(state_->Input(), in));
        }
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 4, "Must use mtie with 3 outputs on svd(). ie: (mtie(U, S, VT) = svd(A))");
        if constexpr (is_cuda_executor_v<Executor>) {
          svd_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), state_->Input(), ex, state_->Jobz());
        } else {
          svd_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), state_->Input(), ex, state_->Jobz(), state_->Algo());
        }
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

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }
      // Size is not relevant in svd() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return state_->Input().Size(dim);
      }

  };
}

/**
 * Perform a singular value decomposition (SVD) using cuSolver or a LAPACK host
 * library.
 * 
 * The singular values within each vector are sorted in descending order.
 * 
 * If rank > 2, operations are batched.
 *
 * @tparam OpA
 *   Operator input type
 *
 * @param a
 *   Input operator of shape `... x m x n`
 * @param jobz
 *   Compute all, part, or none of matrices *U* and *VT*
 * @param algo
 *   For Host SVD calls, whether to use more efficient divide-and-conquer based
 *   `gesdd` routine or the QR factorization based `gesvd` routine. `gesdd`
 *   can run significantly faster, especially for large matrices. However, `gesdd`
 *   requires \f$ O(\min(M,N) ^ 2) \f$ memory as compared to \f$ O(\max(M,N)) \f$ for
 *   `gesvd`, and it can have poorer accuracy in some cases.
 *   Ignored for CUDA SVD calls.
 * 
 * @return 
 *   Operator that produces *U*, *S*, and *VT* tensors. Regardless of jobz, all 3 tensors
 *   must be correctly setup for the operation and used with `mtie()`. `k = min(m, n)`
 *   - **U** - The unitary matrix containing the left singular vectors. A tensor of
 *             shape `... x m x k` for `SVDMode::REDUCED` and `... x m x m` otherwise.
 *   - **S** - A tensor of shape `... x k` containing the singular values in
 *             descending order. It must be of real type and match the inner type of
 *             the other tensors.
 *   - **VT** - The unitary matrix containing the right singular vectors. A tensor of
 *             shape `... x k x n` for `SVDMode::REDUCED` and `... x n x n` otherwise.
 */
template<typename OpA>
__MATX_INLINE__ auto svd(const OpA &a, const SVDMode jobz = SVDMode::ALL,
                        const SVDHostAlgo algo = SVDHostAlgo::DC) {
  return detail::SVDOp(a, jobz, algo);
}


namespace detail {
  template<typename OpA, typename OpX>
  class SVDPIOp : public BaseOp<SVDPIOp<OpA,OpX>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      typename detail::base_type_t<OpX> x_;
      int iterations_;
      index_t k_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using svd_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "svdpi(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ SVDPIOp(const OpA &a, const OpX &x, int iterations, index_t k) : a_(a), x_(x), iterations_(iterations), k_(k)
      {
        MATX_LOG_TRACE("{} constructor: iterations={}, k={}", str(), iterations, k); }

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, 
                                           detail::get_operator_capability<Cap>(a_, in),
                                           detail::get_operator_capability<Cap>(x_, in));
        }
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, 
                                           detail::get_operator_capability<Cap>(a_, in),
                                           detail::get_operator_capability<Cap>(x_, in));
        }
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) {
        static_assert(is_cuda_executor_v<Executor>, "svdpi() only supports the CUDA executor currently");
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 4, "Must use mtie with 3 outputs on svdpi(). ie: (mtie(U, S, VT) = svdpi(A))");

        svdpi_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), a_, x_, iterations_, ex, k_);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return matxNoRank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
      {
        return 0;
      }

  };
}

/**
 * Perform a SVD decomposition using the power iteration.  This version of
 * SVD works well on small n/m with large batch.
 *
 * @tparam AType
 *   Tensor or operator type for output of A input tensors.
 * @tparam X0Type
 *   Tensor or operator type for X0 initial guess in power iteration.
 *
 * @param A
 *   Input tensor or operator for tensor A input with size `batches x m x n`
 * @param x0
 *   Input tensor or operator signaling the initial guess for x0 at each power iteration.  A
 *   Random tensor of size `batches x min(n,m)` is suggested.
 * @param iterations
 *   The number of power iterations to perform for each singular value.  
 * @param k
 *    The number of singular values to find.  Default is all singular values: min(m,n).
 */
template<typename AType, typename X0Type>
__MATX_INLINE__ auto svdpi(const AType &A, const X0Type &x0, int iterations, index_t k=-1) {
  return detail::SVDPIOp(A, x0, iterations, k);
}




namespace detail {
  template<typename OpA>
  class SVDBPIOp : public BaseOp<SVDBPIOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      int max_iters_;
      float tol_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using svd_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "svdpi(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ SVDBPIOp(const OpA &a, int max_iters, float tol) : a_(a), max_iters_(max_iters), tol_(tol)
      { }

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(a_, in));
        }
        else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
        }
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) {
        static_assert(is_cuda_executor_v<Executor>, "svdbpi() only supports the CUDA executor currently");
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 4, "Must use mtie with 3 outputs on svdbpi(). ie: (mtie(U, S, VT) = svdbpi(A))");

        svdbpi_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), a_, max_iters_, tol_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return matxNoRank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
      {
        return 0;
      }



  };
}

/**
 * Perform a SVD decomposition using the block power iteration.  This version of
 * SVD works well on small n/m with large batch.
 *
 * @tparam AType
 *   Tensor or operator type for output of A input tensors.
 *
 * @param A
 *   Input tensor or operator for tensor A input with size `batches x m x n`
 * @param max_iters
 *   The approximate maximum number of QR iterations to perform. 
 * @param tol
 *   The termination tolerance for the QR iteration. Setting this to 0 will skip the tolerance check.
 */
template<typename AType>
__MATX_INLINE__ auto svdbpi(const AType &A, int max_iters=10, float tol=0.0f) {
  return detail::SVDBPIOp(A, max_iters, tol);
}

}
