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

namespace matx
{
  /**
   * Returns elements on the diagonal
   *
   * Returns elements on the diagonal of a 2D tensor. Any dimensions above 2 will
   * be considered batch dimension and the size of those match the size of the
   * input operator. The last dimension is always sized to be the minimum of the
   * last two dimension of the input operator
   */
  namespace detail {
    template <typename T1, int RANK>
      class DiagOp : public BaseOp<DiagOp<T1, RANK>>
    {
      private:
        typename detail::base_type_t<T1> op_;
        index_t k_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ std::string str() const { return "diag(" + op_.str() + ")"; }

        __MATX_INLINE__ DiagOp(const T1 &op, index_t k) : op_(op), k_(k) { }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          if constexpr (EPT == ElementsPerThread::ONE) {
            static_assert(RANK != 0, "Cannot make get diagonals from 0D tensor");
            using tt = typename cuda::std::common_type_t<Is...>;

            if constexpr (RANK == 1) {
              static_assert(sizeof...(Is) == 2, "Indexing of diag() on a 1D input must be 2 indices");
              if (((pp_get<0>(indices...) == indices) && ...)) {
                return get_value<EPT>(op_, pp_get<0>(indices...));
              }
              else {
                return (value_type)(0);
              }
            }
            else {
              static_assert(sizeof...(Is) == RANK - 1, "Diagonal operator must have one fewer op() index than rank of operator");

              // Offset either the rows or columns by k_, depending on if it's negative
              if (k_ < 0) {
                cuda::std::array<tt, sizeof...(Is) + 1> tmp{indices...};
                tmp[RANK - 1] = pp_get<RANK-2>(indices...);
                //cuda::std::get<RANK - 1>(tup) = pp_get<RANK-2>(indices...) ;
  MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
                tmp[RANK - 2] -= k_;
                //cuda::std::get<RANK - 2>(tup) = cuda::std::get<RANK - 2>(tup) - k_;
  MATX_IGNORE_WARNING_POP_GCC
                return get_value<EPT>(op_, tmp);
              }
              else {
                cuda::std::array<tt, sizeof...(Is) + 1> tmp{indices...};
                //auto tup = cuda::std::make_tuple(indices..., static_cast<tt>(0));
  MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
                tmp[RANK - 1] = pp_get<RANK-2>(indices...) + k_;
                //cuda::std::get<RANK - 1>(tup) = pp_get<RANK-2>(indices...) + k_;
  MATX_IGNORE_WARNING_POP_GCC
                return get_value<EPT>(op_, tmp);
              }
            }
          }
          else {
            return Vector<value_type, static_cast<index_t>(EPT)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            return ElementsPerThread::ONE;
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          if constexpr (RANK == 1) {
            return 2;
          }
          else {
            return RANK - 1;
          }
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          if constexpr (RANK == 1) {
            return op_.Size(0);
          }
          else {
            if (dim < RANK - 2) {
              return op_.Size(dim);
            }
            else {
              if (k_ == 0) {
                return cuda::std::min(op_.Size(RANK - 1), op_.Size(RANK-2));
              }
              else {
                // If k is off the main diagonal we need to adjust the sizes
                if (k_ > 0) {
                  return cuda::std::min(op_.Size(RANK - 1), op_.Size(RANK-2) - k_);
                }
                else {
                  return cuda::std::min(op_.Size(RANK - 1) + k_, op_.Size(RANK-2));
                }
              }
            }
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }

  /**
   * Get the elements on the diagonal (2D inputs and above), or generate a diagonal matrix (1D input)
   *
   * @param t
   *   Input operator
   * @param k
   *   Diagonal to pull (0 is the main diagonal). Only used for 2D tensors and above
   */
#ifdef DOXYGEN_ONLY
  auto __MATX_INLINE__ diag(const T1 &t, index_t k = 0) {
#else
  template <typename T1, std::enable_if_t<is_matx_op<T1>(), bool> = true>
    auto __MATX_INLINE__ diag(T1 t, index_t k = 0) {
#endif
      MATX_ASSERT_STR(T1::Rank() != 1 || k == 0, matxInvalidParameter,
          "k parameter in diag() can only be used for 2D tensors and above");
      return detail::DiagOp<T1, T1::Rank()>(t, k);
    }
} // end namespace matx
