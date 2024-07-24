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
   * permutes dimensions of a tensor/operator
   */
  namespace detail {
    template <typename T>
      class PermuteOp : public BaseOp<PermuteOp<T>>
    {
      public: 
        using value_type = typename T::value_type;
        using self_type = PermuteOp<T>;

      private:
        typename base_type<T>::type op_;
        cuda::std::array<int32_t, T::Rank()> dims_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;
        
        __MATX_INLINE__ std::string str() const { return "permute(" + op_.str() + ")"; }
 
        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T::Rank();
        }

        static_assert(Rank() > 0, "PermuteOp: Rank of operator must be greater than 0.");

	      __MATX_INLINE__ PermuteOp(T op, const cuda::std::array<int32_t, T::Rank()> &dims) : op_(op) {
            
          for(int32_t i = 0; i < Rank(); i++) {
            [[maybe_unused]] int32_t dim = dims[i];
            MATX_ASSERT_STR(dim < Rank() && dim >= 0, matxInvalidDim, "PermuteOp:  Invalid permute index.");

            dims_[i] = dims[i];
          }
        };


        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
          {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            cuda::std::array<index_t, Rank()> inds{indices...};
            cuda::std::array<index_t, T::Rank()> ind{indices...};

#pragma unroll 
            for(int32_t i = 0; i < Rank(); i++) {	  
              ind[dims_[i]] = inds[i];
              //ind[i] = inds[dims_[i]];
            }

            //return op_(ind);
            return cuda::std::apply(op_, ind);
          }

        template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
          {
            static_assert(sizeof...(Is)==Rank());
            //          static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            cuda::std::array<index_t, Rank()> inds{indices...};
            cuda::std::array<index_t, T::Rank()> ind{indices...};

#pragma unroll 
            for(int i = 0; i < Rank(); i++) {	  
              ind[dims_[i]] = inds[i];
            }

            return cuda::std::apply(op_, ind);
          }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int32_t dim) const
        {
          return op_.Size(dims_[dim]);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }   

        ~PermuteOp() = default;
        PermuteOp(const PermuteOp &rhs) = default;
        __MATX_INLINE__ auto operator=(const self_type &rhs) { 
          return set(*this, rhs); 
        }                      

        template<typename R> 
        __MATX_INLINE__ auto operator=(const R &rhs) { 
          if constexpr (is_matx_transform_op<R>()) {
            return mtie(*this, rhs);
          }
          else {          
            return set(*this, rhs); 
          }
        }
    };
  }
  
  /**
   * @brief Operator to permute the dimensions of a tensor or operator.
   *
   * The each dimension must appear in the dims array once.
   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param dims the reordered dimensions of the operator.
   * @return permuted operator
   */
  template <typename T>
    __MATX_INLINE__ auto permute( const T &op, 
        const cuda::std::array<int32_t, T::Rank()> &dims) {
      if constexpr (is_tensor_view_v<T>) {
        return op.Permute(dims);
      } else {
        return detail::PermuteOp<T>(op, dims);
      }
    }
  

  /**
   * @brief Operator to permute the dimensions of a tensor or operator.
   *
   * The each dimension must appear in the dims array once.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param dims the reordered dimensions of the operator.
   * @return permuted operator
   */
  template <typename T>
    __MATX_INLINE__ auto permute( const T &op, 
        const int32_t (&dims)[T::Rank()]) {
      return permute(op, detail::to_array(dims));
    }


} // end namespace matx
