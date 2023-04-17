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
   * Remaps elements an operator according to an index array/operator.
   */
  namespace detail {
    template <int DIM, typename T, typename IdxType>
      class RemapOp : public BaseOp<RemapOp<DIM, T, IdxType>>
    {
      private:
        //mutable typename base_type<T>::type op_;
        typename base_type<T>::type op_;
        typename base_type<IdxType>::type idx_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        using scalar_type = typename T::scalar_type;
        using shape_type = typename T::shape_type; 
        using index_type = typename IdxType::scalar_type;
        static_assert(std::is_integral<index_type>::value, "RemapOp: Type for index operator must be integral");
        static_assert(IdxType::Rank() <= 1, "RemapOp: Rank of index operator must be 0 or 1");
        static_assert(DIM<T::Rank(), "RemapOp: DIM must be less than Rank of tensor");

        __MATX_INLINE__ std::string str() const { return "remap(" + op_.str() + ")"; }

	__MATX_INLINE__ RemapOp(T op, IdxType idx) : op_(op), idx_(idx) {};

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
          {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            std::array<index_t, Rank()> ind{indices...};
            // get current index for dim
            auto i = ind[DIM];
            // remap current index for dim
            if constexpr (IdxType::Rank() == 0) {
              ind[DIM] = idx_();
            } else {
              ind[DIM] = idx_(i);
            }
            //return op_(ind);
            return mapply(op_, ind);
          }

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto& operator()(Is... indices)
          {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            std::array<index_t, Rank()> ind{indices...};
            // get current index for dim
            auto i = ind[DIM];
            // remap current index for dim
            if constexpr (IdxType::Rank() == 0) {
              ind[DIM] = idx_();
            } else {
              ind[DIM] = idx_(i);
            }
            //return op_(ind);
            return mapply(op_, ind);
          }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int32_t dim) const
        {
          if(dim == DIM) {
            if constexpr (IdxType::Rank() == 0) {
              return 1;
            } else {
              return idx_.Size(0);
            }
          } else {
            return op_.Size(dim);
          }
        }

        template<typename R> __MATX_INLINE__ auto operator=(const R &rhs) { return set(*this, rhs); }
    };
  }

  /**
   * @brief Operator to logically remap elements of an operator based on an index array/operator.
   *
   * The rank of the output tensor is equal to the rank of the input tensor.
   * The rank of the index tensor must be 0 or 1.
   *
   * The size of the output tensor is the same as the input tensor except in the applied dimenions.
   * In the applied dimension the size of the output tensor is equal to the size of the index tensor.
   * In the case of a 0-rank index tensor, the size of the output tensor in the corresponding
   * dimension is always 1.
   * 
   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam DIM Dimension to apply the remap 
   * @tparam T Input operator/tensor type
   * @tparam Ind Input index Operator type
   * @param t Input operator
   * @param idx Index operator/tensor
   * @return Value in t from each location in idx
   */
  template <int DIM, typename Op, typename Ind>
    auto __MATX_INLINE__ remap(Op t, Ind idx)
    {
      return detail::RemapOp<DIM, Op, Ind>(t, idx);
    };   

  /**
   * @brief Operator to logically remap elements of an operator based on an index array/operator.
   *
   * The rank of the output tensor is equal to the rank of the input tensor.
   * The rank of the index tensor must be 0 or 1.
   * The number of DIMS and the number of Inds provided must be the same.
   *
   * The size of the output tensor is the same as the input tensor except in the applied dimenions.
   * In the applied dimension the size of the output tensor is equal to the size of the index tensor.
   * In the case of a 0-rank index tensor, the size of the output tensor in the corresponding
   * dimension is always 1.
   * 
   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam DIM Dimension to apply the remap 
   * @tparam DIMS... list of multiple dimensions to remap along
   * @tparam T Input operator/tensor type
   * @tparam Ind Input index Operator type
   * @tparam Inds... list of multiple index operators to remap along
   * @param t Input operator
   * @param idx Index operator/tensor
   * @param inds list of multiple index operators to remap along
   * @return Value in t from each location in idx
   */
  template <int DIM, int... DIMS, typename Op, typename Ind, typename... Inds>
    auto __MATX_INLINE__ remap(Op t, Ind idx, Inds... inds)
    {
      static_assert(sizeof...(DIMS) == sizeof...(Inds), "remap number of DIMs must match number of index arrays");

      // recursively call remap on remaining bits
      auto op = remap<DIMS...>(t, inds...);

      // construct remap op
      return detail::RemapOp<DIM, decltype(op) , Ind>(op, idx);
    };   
} // end namespace matx
