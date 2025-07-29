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
#ifndef __CUDACC_RTC__
  #include "matx/transforms/qr/qr_cuda.h"
  #ifdef MATX_EN_CPU_SOLVER
    #include "matx/transforms/qr/qr_lapack.h"
  #endif
#endif

namespace matx {

namespace detail {
  template<typename OpA>
  class QROp : public BaseOp<QROp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using qr_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "qr(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ QROp(const OpA &a) : a_(a) { };

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] const InType& in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
      }

#ifndef __CUDACC_RTC__
      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "qr() only supports the CUDA executor currently");
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 3 outputs on qr(). ie: (mtie(Q, R) = qr(A))");

        qr_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), a_, ex);
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
#endif
      // Size is not relevant in qr() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
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
  class SolverQROp : public BaseOp<SolverQROp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using qr_solver_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "qr_solver()"; }
      __MATX_INLINE__ SolverQROp(const OpA &a) : a_(a) { }    

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] const InType& in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on qr_solver(). ie: (mtie(A, tau) = eig(A))");     

        qr_solver_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), a_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      // Size is not relevant in qr_solver() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
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
  class EconQROp : public BaseOp<EconQROp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using qr_solver_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "qr_econ()"; }
      __MATX_INLINE__ EconQROp(const OpA &a) : a_(a) { }    

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] const InType& in) const {
        auto self_has_cap = capability_attributes<Cap>::default_value;
        return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
      }

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on qr_econ(). ie: (mtie(Q, R) = qr_econ(A))");     

        qr_econ_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), a_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      // Size is not relevant in qr_solver() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
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