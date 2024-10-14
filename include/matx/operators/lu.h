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
#include "matx/transforms/lu/lu_cuda.h"
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/lu/lu_lapack.h"
#endif


namespace matx {
namespace detail {
  template<typename OpA>
  class LUOp : public BaseOp<LUOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using lu_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "lu()"; }
      __MATX_INLINE__ LUOp(const OpA &a) : a_(a) { };

      // This should never be called
      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on lu(). ie: (mtie(O, piv) = lu(A))");     

        lu_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), a_, ex);        
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

      // Size is not relevant in eig() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
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