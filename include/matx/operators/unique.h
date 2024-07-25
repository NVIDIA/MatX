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
#include "matx/transforms/cub.h"

namespace matx {



namespace detail {
  template<typename OpA>
  class UniqueOp : public BaseOp<UniqueOp<OpA>>
  {
    private:
      OpA a_;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using matx_transform_op = bool;
      using unique_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "unique()"; }
      __MATX_INLINE__ UniqueOp(OpA a) : a_(a) { };

      // This should never be called
      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(cuda::std::tuple_size_v<remove_cvref_t<Out>> == 3, "Must use mtie with 2 outputs on unique(). ie: (mtie(O, num_found) = unique(A))");     

        unique_impl(cuda::std::get<0>(out), cuda::std::get<1>(out), a_, ex);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpA::Rank();
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
      {
        MATX_ASSERT_STR(false, matxNotSupported, "unique() must only be called with a single assignment since it has multiple return types");
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      // Size is not relevant in unique() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return a_.Size(dim);
      }

  };
}

/**
 * Reduce to unique values
 *
 * Reduces the input to only unique values saved into the output. Output tensor must be large enough to
 * hold unique entries. To be safe, this can be the same size as the input, but if something is known about
 * the data to indicate not as many entries are needed, the output can be smaller.
 *
 * @tparam InputOperator
 *   Input type
 * @param a
 *   Input tensor
 * @returns Two operators, the first representing the unique items, and the second representing how many items were found
 */
template<typename OpA>
__MATX_INLINE__ auto unique(const OpA &a) {
  return detail::UniqueOp(a);
}

}
