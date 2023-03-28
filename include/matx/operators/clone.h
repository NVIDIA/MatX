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
      private:
        mutable typename base_type<T>::type op_;
        std::array<index_t, CRank> sizes_;         // size of each dimension after cloning
        std::array<index_t, T::Rank()> dims_;      // gather map for computing operator() indices
      public:
        using matxop = bool;

        using scalar_type = typename T::scalar_type;

        __MATX_INLINE__ std::string str() const { return "clone(" + op_.str() + ")"; }

        __MATX_INLINE__ CloneOp(T op, std::array<index_t, CRank> shape) : op_(op) {
          // create gather list
          int d = 0;
          for(int i = 0; i < Rank(); i++) {
            if constexpr (T::Rank() > 0) { // This is needed since the compiler can be fooled
              if(shape[i] == matxKeepDim) {
                sizes_[i] = op_.Size(d);
                dims_[d++] = i;
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

        };

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
          {

            // convert variadic type to tuple so we can read/update
            std::array<index_t, Rank()> sind{indices...};
            std::array<index_t, T::Rank()> gind;

            // gather indices
            for(int i = 0; i < T::Rank(); i++) {
              auto idx = dims_[i];
              gind[i] = sind[idx];
            }

            return mapply(op_, gind);
          }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return CRank;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return sizes_[dim];
        }

    };
  }


  /**
   * @brief Operator to clone an operator or tensor acorss dimensions
   *
   * @tparam Rank the rank of the cloned operator
   * @tparam T source operator/tensor type
   * @param t source operator/tensor
   * @param shape the shape of the cloned operator/tensor.  
   * Each element is either the size of the cloned dimension or matxKeepDim to be from the source tensor
   * @return operator to compute the cloned value
   */
  template <std::size_t Rank, typename Op>
    auto __MATX_INLINE__ clone(Op t, const std::array<index_t, Rank> &shape)
    {
      if constexpr (is_tensor_view_v<Op>) {
        return t.template Clone<static_cast<int>(Rank)>(shape);
      } else {
        return detail::CloneOp<static_cast<int>(Rank), Op>(t, shape);

      }
    };  

  template <int Rank, typename Op>
    auto __MATX_INLINE__ clone(Op t, const index_t (&shape)[Rank])
    {
      return clone<Rank, Op>(t, detail::to_array(shape));
    };   
  

} // end namespace matx
