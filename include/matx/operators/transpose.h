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
#include "matx/operators/permute.h"
#include "matx/transforms/transpose.h"

namespace matx {


namespace detail {
  template<typename OpA>
  class TransposeMatrixOp : public BaseOp<TransposeMatrixOp<OpA>>
  {
    private:
      typename detail::base_type_t<OpA> a_;
      cuda::std::array<index_t, OpA::Rank()> out_dims_;
      mutable detail::tensor_impl_t<typename OpA::value_type, OpA::Rank()> tmp_out_;
      mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;

    public:
      using matxop = bool;
      using value_type = typename OpA::value_type;
      using shape_type = std::conditional_t<has_shape_type_v<OpA>, typename OpA::shape_type, index_t>; 
      using matx_transform_op = bool;
      using matxoplvalue = bool;
      using transpose_xform_op = bool;
      using self_type = TransposeMatrixOp<OpA>;

      __MATX_INLINE__ std::string str() const { return "transpose_matrix(" + get_type_str(a_) + ")"; }
      __MATX_INLINE__ TransposeMatrixOp(const OpA &a) : a_(a) {
        for (int r = 0; r < Rank(); r++) {
          if (r >= Rank() - 2) {
            out_dims_[r] = (r == Rank() - 1) ? a_.Size(Rank() - 2) : a_.Size(Rank() - 1);
          }
          else {
            out_dims_[r] = a_.Size(r);
          }
        }        
      }

      __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
      {
        return tmp_out_.template operator()<InWidth, OutWidth>(indices...);
      }

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        return tmp_out_.template operator()<InWidth, OutWidth>(indices...);
      }     

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        transpose_matrix_impl(cuda::std::get<0>(out), a_, ex);
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

        detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

        Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<OpA>()) {
          a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        matxFree(ptr);
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return out_dims_[dim];
      }

      TransposeMatrixOp(const TransposeMatrixOp &rhs) = default;
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
 * @brief Operator to transpose the dimensions of a tensor or operator.
 *
 * The each dimension must appear in the dims array once.

  * This operator can appear as an rvalue or lvalue. 
  *
  * @tparam T Input operator/tensor type
  * @param op Input operator
  * @return transposed operator
  */
  template <typename T>
  __MATX_INLINE__ auto transpose(const T &op) {
  
    static_assert(T::Rank() >= 2, "transpose operator must be on rank 2 or greater");
    int32_t dims[T::Rank()];
    for (int r = 0; r < T::Rank(); r++) {
      dims[r] = T::Rank() - r - 1;
    }

    return permute(op, dims);
  }


  /**
   * @brief Operator to transpose the last two dimensions of a tensor or operator.
   *
   * The each dimension must appear in the dims array once.

   * This operator can appear as an rvalue or lvalue. 
   *
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @return permuted operator
   */
  template <typename T>
  __MATX_INLINE__ auto transpose_matrix(const T &op) {
    static_assert(T::Rank() >= 2, "transpose operator must be on rank 2 or greater");

    int32_t dims[T::Rank()];
    for(int i = 0; i < T::Rank(); i++) 
      dims[i] = i;
    int32_t dim1 = T::Rank() - 1;
    int32_t dim2 = T::Rank() - 2;

    std::swap(dims[dim1],dims[dim2]);
    return permute(op, detail::to_array(dims));
  }  

}
