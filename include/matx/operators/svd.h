////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// COpBright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above cOpBright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above cOpBright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the cOpBright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COpBRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
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
#include "matx/transforms/svd/svd_cuda.h"
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/svd/svd_lapack.h"
#endif

namespace matx {


namespace detail {
  template<typename OpA>
  class SVDOp : public BaseOp<SVDOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      SVDMode jobz_;
      SVDHostAlgo algo_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using svd_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "svd(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ SVDOp(const OpA &a, const SVDMode jobz, const SVDHostAlgo algo) : a_(a), jobz_(jobz), algo_(algo) { };

      // This should never be called
      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      // TODO: Handle SVDMode::NONE case better to not require U & VT
      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 4, "Must use mtie with 3 outputs on svd(). ie: (mtie(U, S, VT) = svd(A))");
        if constexpr (is_cuda_executor_v<Executor>) {
          svd_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), a_, ex, jobz_);
        } else {
          svd_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), a_, ex, jobz_, algo_);
        }
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      // Size is not relevant in svd() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
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
 *   `gesdd` routine or the QR factorization based `gesvd` routine. `gesdd`
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
      { }

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

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

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
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

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
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
