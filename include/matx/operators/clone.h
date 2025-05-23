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
  namespace detail {
    template <std::size_t CRank, typename T>
      class CloneOp : public BaseOp<CloneOp<CRank, T>>
    {
      static_assert(CRank > T::Rank(), "Clone rank must be higher than input rank");
      private:
        mutable typename detail::base_type_t<T> op_;
        cuda::std::array<index_t, CRank> sizes_;         // size of each dimension after cloning
        cuda::std::array<index_t, T::Rank()> dims_;      // gather map for computing operator() indices
      public:
        using matxop = bool;

        using value_type = typename T::value_type;

        __MATX_INLINE__ std::string str() const { return "clone(" + op_.str() + ")"; }

        __MATX_INLINE__ CloneOp(const T &op, cuda::std::array<index_t, CRank> shape) : op_(op) {
          static_assert(T::Rank() < CRank, "Cloning rank must be higher than input operator rank");

          [[maybe_unused]] const index_t num_keep = static_cast<index_t>(
			  std::count_if(shape.begin(), shape.end(), [](index_t i) { return i == matxKeepDim; }));
          MATX_ASSERT_STR(num_keep == T::Rank(), matxInvalidParameter,
            "Number of matxKeepDim in a clone must match input operator rank");

          // create gather list
          int d = 0;
          for(int i = 0; i < Rank(); i++) {
            if constexpr (T::Rank() > 0) { // This is needed since the compiler can be fooled
              if(shape[i] == matxKeepDim) {
                sizes_[i] = op_.Size(d);
                // gcc incorrectly shows an invalid access to array element [1] in a unit test here. This is not
                // possible based on runtime checks we have. Disable the warning temporarily.
MATX_IGNORE_WARNING_PUSH_GCC("-Warray-bounds")
                dims_[d++] = i;
MATX_IGNORE_WARNING_POP_GCC
              } else {
                sizes_[i] = shape[i];
              }
            }
            else {
              MATX_ASSERT(shape[i] != matxKeepDim, matxInvalidDim);
              sizes_[i] = shape[i];
            }
          }
          MATX_ASSERT(d == T::Rank(), matxInvalidDim);

        }

        template <ElementsPerThread EPT, typename Op, typename Dims, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, const Dims &dims, Is... indices)
        {
          if constexpr (EPT == ElementsPerThread::ONE) {
  MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
            cuda::std::array<index_t, Rank()> sind{indices...};
            cuda::std::array<index_t, T::Rank()> gind;
  MATX_IGNORE_WARNING_POP_GCC

            // gather indices
            for(int i = 0; i < T::Rank(); i++) {
              auto idx = dims[i];
              gind[i] = sind[idx];
            }

            return get_value<EPT>(cuda::std::forward<Op>(op), gind);
          }
          else {
            return Vector<value_type, static_cast<index_t>(EPT)>{};
          }
        }

        template <typename Op, typename Dims, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, const Dims &dims, Is... indices)
        {
          return get_impl<detail::ElementsPerThread::ONE>(cuda::std::forward<Op>(op), dims, indices...);
        }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<EPT>(cuda::std::as_const(op_), dims_, indices...);
        }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<EPT>(cuda::std::forward<decltype(op_)>(op_), dims_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return CRank;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return sizes_[dim];
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            return ElementsPerThread::ONE;
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_));
          }
        }
    };
  }


  /**
   * @brief Operator to clone an operator or tensor across dimensions
   *
   * @tparam Rank the rank of the cloned operator
   * @tparam T source operator/tensor type
   * @param t source operator/tensor
   * @param shape the shape of the cloned operator/tensor.
   * Each element is either the size of the cloned dimension or `matxKeepDim` to be from the source tensor
   * @return operator to compute the cloned value
   */
  template <std::size_t Rank, typename Op>
  auto __MATX_INLINE__ clone(const Op &t, const cuda::std::array<index_t, Rank> &shape)
  {
    static_assert(Rank >= Op::Rank(), "Cloning rank must be >= input operator rank");

    if constexpr (Op::Rank() == Rank) {
      return t; // No-op to same rank
    }
    else if constexpr (is_tensor_view_v<Op>) {
      return t.template Clone<static_cast<int>(Rank)>(shape);
    } else {
      return detail::CloneOp<static_cast<int>(Rank), Op>(t, shape);

    }
  };

  template <int Rank, typename Op>
  auto __MATX_INLINE__ clone(const Op &t, const index_t (&shape)[Rank])
  {
    return clone<Rank, Op>(t, detail::to_array(shape));
  };


} // end namespace matx
