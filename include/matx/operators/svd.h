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
#include "matx/transforms/solver.h"
#include "matx/transforms/svd.h"

namespace matx {


namespace detail {
  template<typename OpA>
  class SVDOp : public BaseOp<SVDOp<OpA>>
  {
    private:
      OpA a_;
      char jobu_;
      char jobv_;
      mutable bool init_ = false;

    public:
      using matxop = bool;
      using scalar_type = typename OpA::scalar_type;
      using matx_transform_op = bool;
      using svd_xform_op = bool;
      using matx_multi_return_op = bool;

      __MATX_INLINE__ std::string str() const { return "svd(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ SVDOp(OpA a, const char jobu, const char jobvt) : a_(a), jobu_(jobu), jobv_(jobvt) { };

      bool Initialized() const { return init_; }

      // This should never be called
      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "svd() only supports the CUDA executor currently");
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 4, "Must use mtie with 3 outputs on svd(). ie: (mtie(U, S, V) = svd(A))");

        svd_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), a_, ex.getStream(), jobu_, jobv_);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        MATX_ASSERT_STR(false, matxNotSupported, "svd() must only be called with a single assignment");
      }

      // Size is not relevant in svd() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
      }

  };
}

template<typename OpA>
__MATX_INLINE__ auto svd(const OpA &a, const char jobu = 'A', const char jobvt = 'A') {
  return detail::SVDOp(a, jobu, jobvt);
}


namespace detail {
  template<typename OpA, typename OpX>
  class SVDPIOp : public BaseOp<SVDPIOp<OpA,OpX>>
  {
    private:
      OpA a_;
      OpX x_;
      int iterations_;
      index_t k_;
      mutable bool init_ = false;

    public:
      using matxop = bool;
      using scalar_type = typename OpA::scalar_type;
      using matx_transform_op = bool;
      using svd_xform_op = bool;
      using matx_multi_return_op = bool;

      __MATX_INLINE__ std::string str() const { return "svdpi(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ SVDPIOp(const OpA &a, const OpX &x, int iterations, index_t k) : a_(a), x_(x), iterations_(iterations), k_(k)
      { }

      bool Initialized() const { return init_; }

      // This should never be called
      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) {
        static_assert(is_cuda_executor_v<Executor>, "svdpi() only supports the CUDA executor currently");
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 4, "Must use mtie with 3 outputs on svdpi(). ie: (mtie(U, S, V) = svdpi(A))");

        svdpi_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), a_, x_, iterations_, ex.getStream(), k_);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return matxNoRank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) noexcept
      {
        MATX_ASSERT_STR(false, matxNotSupported, "svdpi() must only be called with a single assignment");
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
 *   Input tensor or operator for tensor A input with size "batches by m by n"
 * @param x0
 *   Input tensor or operator signaling the initial guess for x0 at each power iteration.  A
 *   Random tensor of size batches x min(n,m) is suggested.
 * @param iterations
 *   The number of power iterations to perform for each singular value.
 * @param k
 *    The number of singular values to find.  Default is all singular values: min(m,n).
 */
template<typename AType, typename X0Type>
__MATX_INLINE__ auto svdpi(AType &A, X0Type &x0, int iterations, index_t k=-1) {
  return detail::SVDPIOp(A, x0, iterations, k);
}




namespace detail {
  template<typename OpA>
  class SVDBPIOp : public BaseOp<SVDBPIOp<OpA>>
  {
    private:
      OpA a_;
      int max_iters_;
      float tol_;
      mutable bool init_ = false;

    public:
      using matxop = bool;
      using scalar_type = typename OpA::scalar_type;
      using matx_transform_op = bool;
      using svd_xform_op = bool;
      using matx_multi_return_op = bool;

      __MATX_INLINE__ std::string str() const { return "svdpi(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ SVDBPIOp(const OpA &a, int max_iters, float tol) : a_(a), max_iters_(max_iters), tol_(tol)
      { }

      bool Initialized() const { return init_; }

      // This should never be called
      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) {
        static_assert(is_cuda_executor_v<Executor>, "svdbpi() only supports the CUDA executor currently");
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 4, "Must use mtie with 3 outputs on svdbpi(). ie: (mtie(U, S, V) = svdbpi(A))");

        svdbpi_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), cuda::std::get<2>(out), a_, max_iters_, tol_, ex.getStream());
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return matxNoRank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) noexcept
      {
        MATX_ASSERT_STR(false, matxNotSupported, "svdbpi() must only be called with a single assignment");
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
 *   Input tensor or operator for tensor A input with size "batches by m by n"
 * @param max_iters
 *   The approximate maximum number of QR iterations to perform.
 * @param tol
 *   The termination tolerance for the QR iteration. Setting this to 0 will skip the tolerance check.
 */
template<typename AType>
__MATX_INLINE__ auto svdbpi(AType &A, int max_iters=10, float tol=0.0f) {
  return detail::SVDBPIOp(A, max_iters, tol);
}

}
