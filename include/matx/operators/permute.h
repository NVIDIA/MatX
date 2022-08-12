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


#include "matx_type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx
{
  /**
   * permutes dimensions of a tensor/operator
   */
  namespace detail {
    template <typename T>
      class PermuteOp : public BaseOp<PermuteOp<T>>
    {
      public: 
        using scalar_type = typename T::scalar_type;
        using shape_type = typename T::shape_type; 

      private:
        typename base_type<T>::type op_;
        std::array<int32_t, T::Rank()> dims_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T::Rank();
        }

        static_assert(Rank() > 0, "PermuteOp: Rank of operator must be greater than 0.");

        __MATX_INLINE__ PermuteOp(T op, const int32_t (&dims)[Rank()]) : op_(op) {

          bool selected[Rank()] = {0};

          for(int32_t i = 0; i < Rank(); i++) {
            int32_t dim = dims[i];
            MATX_ASSERT_STR(dim < Rank() && dim >= 0, matxInvalidDim, "PermuteOp:  Invalid permute index.");
            MATX_ASSERT_STR(selected[dim] == false, matxInvalidDim, "PermuteOp:  Dim selected more than once");
            selected[dim] = true;

            dims_[i] = dims[i];
          }
        };

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
          {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            std::array<shape_type, Rank()> inds{indices...};
            std::array<shape_type, T::Rank()> ind{indices...};

#pragma unroll 
            for(int32_t i = 0; i < Rank(); i++) {	  
              ind[dims_[i]] = inds[i];
              //ind[i] = inds[dims_[i]];
            }

            //return op_(ind);
            return mapply(op_, ind);
          }

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto& operator()(Is... indices)
          {
            static_assert(sizeof...(Is)==Rank());
            //          static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            std::array<shape_type, Rank()> inds{indices...};
            std::array<shape_type, T::Rank()> ind{indices...};

#pragma unroll 
            for(int i = 0; i < Rank(); i++) {	  
              ind[dims_[i]] = inds[i];
              //ind[i] = inds[dims_[i]];
            }

            //return op_(ind);
            return mapply(op_, ind);
          }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ shape_type Size(int32_t dim) const
        {
          return op_.Size(dims_[dim]);
        }

        template<typename R> __MATX_INLINE__ auto operator=(const R &rhs) { return set(*this, rhs); }
    };
  }

  /**
   * @brief Operator to permute the dimensions of a tensor or operator.
   *
   * The each dimension must appear in the dims array once.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam T Input operator/tensor type
   * @param Op Input operator
   * @param dims the reordered dimensions of the operator.
   * @return permuted operator
   */
  template <typename T>
    __MATX_INLINE__ auto permute( const T op, 
        const int32_t (&dims)[T::Rank()]) {
      return detail::PermuteOp<T>(op, dims);
    }

} // end namespace matx
